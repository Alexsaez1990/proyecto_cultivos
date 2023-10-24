var map1 = L.map('mapa1');

map1.setMaxBounds([[-35.0, -71.0], [-36.5, -73.0], [-56, -79], [-56, -67.8], [-14, -79]]); // North, south, west, east, north
map1.fitBounds([[-14, -79], [-14, -63], [-56, -79], [-56, -63], [-14, -79]]);
map1.setView([-35, -71]);

// PROYECCIÓN EPSG:4326
var proyeccion = new L.Proj.CRS('EPSG:4326', '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs', {
    resolutions: [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5],
});

var base = L.tileLayer('/get_tile?url=https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
    minZoom: 17,
    maxZoom: 17,
    crs: proyeccion,
    zoomControl: false,
    ext: 'png'
}).addTo(map1);

var maskLayer = L.layerGroup().addTo(map1);
var maskImages = {};

if(screen.width >= 712){ // To future mobile adaptable screen version
    map1.setZoom(17); //16
}else {
    map1.setZoom(17);
}

const TILE_SIZE = 256;

// Function to capture and send a loaded tile
function captureAndSendTile(e) {
    var tile = e.tile;
    //capturedTileElement = tile;
    console.log("Loaded tile: " + tile.src);
    var tilePosition = calculateTilePosition(tile.src);
    if (tilePosition) {
        //console.log("Tile position:", tilePosition);
        // Calculate the tile bounds
        var x = tilePosition.x;
        var y = tilePosition.y;
        var zoom = tilePosition.zoom;

        var mapBounds = map1.getBounds();
        var swPoint = map1.project(mapBounds.getSouthWest(), zoom);
        var nePoint = L.point(swPoint.x + TILE_SIZE * (x + 1), swPoint.y - (TILE_SIZE * y));
        var x1 = swPoint.x + (x * TILE_SIZE);
        var y1 = nePoint.y - ((y+1) * TILE_SIZE);
        var x2 = x1 + TILE_SIZE;
        var y2 = y1 + TILE_SIZE;
        
        // Send the captured tile to the backend
        sendTileToBackend(tile.src, tilePosition, tile.style.transform);
    }
}

// Listen for the 'tileload' event on the tile layer
base.on('tileload', captureAndSendTile);

function createMaskImageBatch(tileBounds, tilePosition, tileTransform) {
    const batchSize = 4;
    const tileURLs = [];

    for (let i = 0; i < batchSize; i++) {
        const x = tilePosition.x + i % 2;
        const y = tilePosition.y + Math.floor(i/2);
        const tileURL = `/get_tile?url=https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${tilePosition.zoom}/${y}/${x}`;
        tileURLs.push(tileURL);
    }

    batchLoadTiles(tileURLs, tilePosition, tileTransform);
}

const debouncedBatchLoadTiles = debounce(batchLoadTiles, 500) // Delay

function createMaskImage(imageUrl, tilePosition, tileTransform) {
    var x = tilePosition.x;
    var y = tilePosition.y;
    var zoom = tilePosition.zoom;

    var mapBounds = map1.getBounds();
    var swPoint = map1.project(mapBounds.getSouthWest(), zoom);
    var nePoint = L.point(swPoint.x + TILE_SIZE, swPoint.y);
    var x1 = swPoint.x + (x * TILE_SIZE);
    var y1 = nePoint.y - ((y) * TILE_SIZE);
    var x2 = x1 + TILE_SIZE;
    var y2 = y1 + TILE_SIZE;
    var swLatLng = map1.unproject([x1, y1], zoom);
    var neLatLng = map1.unproject([x2, y2], zoom);
    var tileBounds = L.latLngBounds(swLatLng, neLatLng);
    console.log("Tile bounds: ", tileBounds);

    

    var maskOverlay = L.imageOverlay(imageUrl, tileBounds, { opacity: 0.2});
    maskOverlay.on('error', function(e) {
        console.error('Error loading mask image', e.error);
    });

    var imgElement = new Image();
    imgElement.classList.add('lazy-load');
    lazyLoadMaskImage(imgElement, imageUrl, tilePosition, tileTransform);
    // Add the mask image to the map
    maskLayer.addLayer(maskOverlay);
    maskOverlay.setZIndex(1000);
    maskOverlay.addTo(map1);
    var imgElement = maskOverlay.getElement();

    if (imgElement) {
        var transformValues = tileTransform.match(/translate3d\(([^,]+),\s+([^,]+),\s+([^,]+)\)/);
        if (transformValues && transformValues.length === 4) {
            var transX = transformValues[1];
            var transY = transformValues[2];
            var translateX = parseFloat(transX.replace('px', '')) - 0; // Revisar por qué con zoom 15 se desplaza 272
            var translateY = parseFloat(transY.replace('px', '')) - 0; // Revisar por qué con zoom 15 se desplaza 362
            translateX = translateX.toString();
            translateY = translateY.toString();
            console.log("translateX =", translateX);
            console.log("translateY =", translateY);

            imgElement.style.transform = `translate3d(${translateX}px, ${translateY}px, 0`;
        }
        console.log(imgElement.style.transform)
        console.log("Tile transform for mask =", tileTransform);
    }
    // Store the mask image in the maskImages object using the tile position as the key
    maskImages[zoom + '-' + x + '-' + y] = maskOverlay;
}

function lazyLoadMaskImage(imgElement, imageUrl, tilePosition, tileTransform) {
    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                // Image is in the viewport, load it
                imgElement.src = imageUrl;
                imgElement.onload = () => {
                    // Once the image is loaded, you can apply your transformations
                    applyTransformations(imgElement, tilePosition, tileTransform);
                    observer.unobserve(imgElement); // Stop observing once loaded
                };
            }
        });
    });

    // Start observing the image element
    observer.observe(imgElement);
}


function calculateTilePosition(tileURL) {
    // Split the URL by '/' and extract the zoom, x, and y values
    var parts = tileURL.split('/');
    if (parts.length >= 6) {
        var zoom = parseInt(parts[parts.length - 3]);
        var x = parseInt(parts[parts.length - 2]);
        var y = parseInt(parts[parts.length - 1]);

        // Return an object containing the extracted values
        return { zoom: zoom, x: x, y: y };
    } else {
        // Handle the case where the URL doesn't match the expected format
        console.error('Invalid tile URL format:', tileURL);
        return null;
    }
}

async function sendTileToBackend(tileURL, tilePosition, tileTransform) {
    try {
        const response = await fetch('/tiles', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ tileURL: tileURL, tilePosition: tilePosition }),
        });

        if (!response.ok) {
            console.error('Failed to send original tile');
            throw new Error('Failed to send original tile');
        }

        console.log('Original Tile sent successfully');
        const tilePositionFromBackend = {
            zoom: response.headers.get('Tile-Position-Zoom'),
            x: response.headers.get('Tile-Position-X'),
            y: response.headers.get('Tile-Position-Y'),
        };

        const data = await response.arrayBuffer();
        const blob = new Blob([data], { type: 'image/png' });
        const imageUrl = URL.createObjectURL(blob);
        //console.log("Received mask image URL ", imageUrl);

        //console.log("Tile position: ", tilePosition);
        //console.log("Tile transform property =", tileTransform);

        createMaskImage(imageUrl, tilePosition, tileTransform);
    } catch (error) {
        console.error('Error while sending tile to the backend', error);
    }
}

function batchLoadTiles(tileURLs, tilePosition, tileTransform) {
    Promise.all(
        tileURLs.map(async (tileURL) => {
            try {
                const response = await fetch(tileURL);
                if (!response.ok) {
                    console.error('Failed to fetch tile: ', tileURL);
                    return;
                }

                const data = await response.arrayBuffer();
                const blob = new Blob([data], {type: 'image/png'});
                const imageURL = URL.createObjectURL(blob);
                console.log('Loaded tile: ', tileURL);

                createMaskImage(imageURL, tilePosition, tileTransform);
            } catch (error) {
                console.error('Error while fetching tile ', tileURL, error);
            }
        })
    );
}

function debounce(func, wait) {
    let timeout;
    return function () {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            func.apply(context, args);
        }, wait);
    };
}