from io import BytesIO
import requests
from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from flask_cors import CORS
from flask_compress import Compress
import os
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import AutoImageProcessor, TFViTModel
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import cachetools
import math

# Configuración servidor Flask para comunicar con el front-ed (Javascript)
app = Flask(__name__, static_url_path='/static')
Compress(app)
CORS(app)

temp_dir = tempfile.mkdtemp()
model = keras.models.load_model('models/model_vit_128_6classes.keras')
mask_cache = cachetools.LRUCache(maxsize=1000)

@app.route('/')
def index():
    # Print the path to main_mapa_cultivos.css
    css_path = os.path.join(app.root_path, 'static', 'main_mapa_cultivos.css')
    print(f'CSS Path: {css_path}')

    # Print the path to main_mapa_cultivos.js
    js_path = os.path.join(app.root_path, 'static', 'main_mapa_cultivos.js')
    print(f'JS Path: {js_path}')
    return render_template('main_mapa_cultivos.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/get_tile', methods=['GET'])
def get_tile():
    tile_url = request.args.get('url', '')
    if not tile_url:
        return Response(status=400)

    response = requests.get(tile_url)

    if response.status_code == 200:
        # Serve the tile image
        return Response(response.content, content_type=response.headers['content-Type'])
    else:
        return Response(status=404)

@app.route('/tiles', methods=['POST'])
def process_tile():
    if request.method == 'POST':
        try:
            data = request.get_json()
            tile_url = data.get('tileURL', '')
            tilePosition = data.get('tilePosition', {})

            sw_latlng = tilePosition.get('swLatLng')
            ne_latlng = tilePosition.get('neLatLng')

            if tile_url:
                response = requests.get(tile_url)

                if response.status_code == 200:
                    image_bytes = response.content
                    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), -1)

                    if image is not None:
                        print(f"Original image shape: {image.shape}")
                        preprocessed_sub_tiles, unprocessed_sub_tiles = split_image_subtiles(image)
                        sub_tile_size = preprocessed_sub_tiles[0].shape
                        print(f"Sub_tile_size: {sub_tile_size}")
                        sub_tiles_emb = emb_extraction(preprocessed_sub_tiles)
                        sub_tile_masks = []

                        # Classification for each sub-tile
                        with ThreadPoolExecutor() as executor:
                            sub_tile_labels = list(executor.map(classify_image, sub_tiles_emb))

                        # Generate masks concurrently and cache them
                        for i, class_label in enumerate(sub_tile_labels):
                            # Check if mask exists in cache
                            mask_cache_key = f'{tilePosition["zoom"]}-{tilePosition["x"]}-{tilePosition["y"]}-{i}'
                            cached_mask = mask_cache.get(mask_cache_key)

                            if cached_mask is not None:
                                sub_tile_mask = cached_mask
                            else:
                                if class_label == 'Agriculture':
                                    dominant_color = max_intensity_avg(unprocessed_sub_tiles[i])
                                    if dominant_color == 'Red':
                                        sub_tile_mask = generate_dry_mask(unprocessed_sub_tiles[i].shape)
                                    elif dominant_color == 'Green':
                                        sub_tile_mask = generate_green_mask(unprocessed_sub_tiles[i].shape)

                                elif class_label == 'Non agriculture':
                                    sub_tile_mask = generate_not_agriculture_mask(sub_tile_size)


                                # Store generated mask in cache with unique key
                                mask_cache[mask_cache_key] = sub_tile_mask

                            sub_tile_masks.append(sub_tile_mask)

                        # Combine sub-tile masks into one mask image
                        num_sub_tiles = len(sub_tile_masks)
                        print(f"num_sub_tiles {num_sub_tiles}")
                        num_rows = int(math.sqrt(num_sub_tiles))
                        num_cols = int(np.ceil(num_sub_tiles / num_rows))
                        print(f"num_rows {num_rows}")
                        print(f"num_cols {num_cols}")

                        tile_mask_height = num_rows * sub_tile_size[0]
                        print(f"tile_mask_height {tile_mask_height}")
                        tile_mask_width = num_cols * sub_tile_size[0]
                        print(f"tile_mask_width {tile_mask_width}")
                        tile_mask = np.zeros((tile_mask_height, tile_mask_width, 3), dtype=np.uint8)
                        for i, sub_tile_mask in enumerate(sub_tile_masks):
                            row = i // num_cols
                            col = i % num_cols
                            print(f"row * sub_tile_size[0]:0 {row * sub_tile_size[0]}")
                            print(f"(row + 1) * sub_tile_size[0] {(row + 1) * sub_tile_size[0]}")
                            print(f"col * sub_tile_size[0] {col * sub_tile_size[0]}")
                            print(f"(col + 1) * sub_tile_size[0] {(col + 1) * sub_tile_size[0]}")
                            tile_mask[row * sub_tile_size[0]:(row + 1) * sub_tile_size[0],
                                      col * sub_tile_size[0]:(col + 1) * sub_tile_size[0], :] = sub_tile_mask

                        # Convert array to PIL Image
                        mask_image = Image.fromarray(tile_mask)
                        print(f"tile_mask shape: {tile_mask.shape}")
                        buffer = BytesIO()
                        mask_image.save(buffer, format='PNG')

                        buffer.seek(0)

                        print("Mask generated successfully.")

                        response = Response(buffer.getvalue(), content_type='image/png')
                        response.headers['Tile-Position-Zoom'] = str(tilePosition['zoom'])
                        response.headers['Tile-Position-X'] = str(tilePosition['x'])
                        response.headers['Tile-Position-Y'] = str(tilePosition['y'])

                        return response

                    else:
                        print("Invalid image or None")
                        return jsonify({'error': 'Invalid Image', 'success': False})
                else:
                    print("Tile data missing")
                    return jsonify({'error': 'Tile Data Missing', 'success': False})
            else:
                print("Tile URL missing")
                return jsonify({'error': 'Tile URL Missing', 'success': False})

        except Exception as e:
            print(f'Error processing tiles: {str(e)}')
            return jsonify({'error': str(e), 'success': False})


def preprocess_image(image, sub_tile_size):
    # Create a new image with shape (128, 128, 4) and fill the alpha channel with 255 (fully opaque)
    image = cv2.resize(image, (sub_tile_size, sub_tile_size))
    image = image.astype(np.float32) / 255.

    return image

def split_image_subtiles(image):
    sub_tile_size = 64  # Cambiar para hacer sub tiles más pequeños
    final_sub_tile_size = 64
    preprocessed_sub_tiles = []
    unprocessed_sub_tiles = []

    for y in range(0, image.shape[0], sub_tile_size):
        for x in range(0, image.shape[1], sub_tile_size):
            sub_tile = image[y:y + sub_tile_size, x:x + sub_tile_size, :]
            preprocessed_tile = preprocess_image(sub_tile, final_sub_tile_size)
            preprocessed_sub_tiles.append(preprocessed_tile)
            unprocessed_sub_tiles.append(sub_tile)

    return preprocessed_sub_tiles, unprocessed_sub_tiles

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
modelViTClasif = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

def emb_extraction(sub_tiles):
    embeddings = []
    for sub_tile in sub_tiles:
        if np.isnan(sub_tile).any() or np.isinf(sub_tile).any():
            print(f"Skipping image due to NaN or Infinity values.")
            continue
        inputs = image_processor(sub_tile, return_tensors='tf', do_rescale=False)
        outputs = modelViTClasif(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = tf.reduce_mean(last_hidden_states, axis=1)
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            print(f"Skipping image due to NaN or Infinity values in embedding.")
            continue
        embeddings.append(embedding.numpy())

    return np.vstack(embeddings)

def classify_image(image):
    try:
        predictions = model.predict(np.expand_dims(image, axis=0))
        if predictions[0][1] > 0.55: # [0][0] means not agriculture. [0][1] means agriculture
            return 'Agriculture'
        else:
            return 'Non agriculture'
    except Exception as e:
        print(f"Error classifying image: {str(e)}")
        return "Error"

def generate_green_mask(shape): # Green
    print(f"Genero máscara agricultura con shape: {shape}")
    return np.full(shape, [0, 128, 0], dtype=np.uint8)

def generate_dry_mask(shape): # Red/Yellow
    print(f"Genero máscara agricultura con shape: {shape}")
    return np.full(shape, [255, 255, 0], dtype=np.uint8)

def generate_not_agriculture_mask(shape): # Blue
    print(f"Genero máscara NO agricultura: {shape}")
    return np.full(shape, [128, 128, 128], dtype=np.uint8)


def max_intensity_avg(image):
    b, g, r = cv2.split(image)
    avg_intensity_b = b.mean()
    avg_intensity_g = g.mean()
    avg_intensity_r = r.mean()

    dominant_channel_avg = 'Blue'
    max_intensity = avg_intensity_b

    if avg_intensity_g > max_intensity:
        dominant_channel_avg = 'Green'
        max_intensity = avg_intensity_g
        coef_rg = avg_intensity_r / avg_intensity_g
        if coef_rg > 1.08:
            dominant_channel_avg = 'Red'

    return dominant_channel_avg

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
