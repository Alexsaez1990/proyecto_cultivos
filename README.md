# Detección de estado de cultivos mediante remote sensing para mejora del despliegue de medios en incendios forestales

## Introducción
En el Departamento de Investigación y Desarrollo de la Corporación Nacional Forestal de Chile, lugar donde trabajo desde hace un año, y, en particular en la Sección de Estudios y Proyectos, que se encarga principalmente del análisis y gestión de los incendios forestales en el país, se detecta la necesidad de un sistema que sea capaz de diferenciar cultivos verdes de cultivos secos mediante imágenes por satélite. 
Esta necesidad, se fundamenta en la velocidad de respuesta y despliegue de forma precisa de medios de combate de incendios, tanto terrestres como aéreos, así como la optimización de recursos en dichos despliegues, durante la temporada de incendios, que en el hemisferio sur comprende los meses de octubre a abril.
La metodología de trabajo que se tiene en nuestro departamento, consiste en general en la utilización de herramientas ya existentes de análisis e imagen de satélite, adaptándose estas, no siempre en las mejores condiciones a las necesidades propias. Se utilizan plataformas como Google Earth Engine, ArcGis o Carto. Algunas de estas herramientas ofrecen modelos de Remote Sensing, pero siempre con las limitaciones de, o bien versiones gratuitas debido a que no se tienen los recursos necesarios o bien, no completamente adaptadas a las necesidades locales. 
En este caso, dichos modelos de detección, son capaces de detectar y diferenciar de forma eficiente en la mayoría de los casos zonas urbanizadas, bosques, cultivos, extensiones de agua, pastizales y otros, pero no existe una herramienta que diferencie cultivos secos o ya recolectados de cultivos verdes.
Este punto es muy importante en el despliegue de medios para el combate de incendios, ya que un campo seco (como por ejemplo de trigo o cebada, amarillo) antes de su recolección, o incluso después, es extremadamente inflamable y extiende rápidamente las llamas, mientras que un campo verde (por ejemplo un maizal), no extenderá las llamas, o, al menos no con la misma intensidad y velocidad. 

## Objetivo del proyecto

El objetivo es crear un modelo propio, que sea capaz de diferenciar los cultivos en función de si están secos o verdes.
A partir de esta idea central, surgen diferentes posibilidades para una tarea que ya se define como de clasificación. Una opción es la creación de un modelo multiclase, que, al igual que los ya existentes, deberá identificar los distintos tipos de suelo, y además, en una segunda fase, diferenciar qué cultivos están verdes y cuales secos. Otra opción, será la de crear un modelo binario, que sepa en una primera fase diferenciar lo que es cultivo de aquello que no lo es y en una segunda, de todo aquello que sea cultivo, diferenciar si está seco o verde, aplicándolo  como una nueva máscara en la visualización en el mapa, por encima de la máscara ya existente.
Se opta por la realización de un modelo binario, con la idea de centrarse en la calidad de la identificación de cultivos, aunque no se descarta realizar pruebas de modelado multiclase.
En ambos casos se deberá entrenar en una primera fase un modelo que diferencie los tipos de suelo y, en una segunda etapa se deberá diferenciar aquellos cultivos secos de los verdes. 
Se debe dejar claro, que, debido a la limitación de los tiempos para la primera iteración del proyecto, el objetivo es entregar un prototipo funcional y que, tras ello, quedará mucho trabajo por delante de mejora de los modelos  y desarrollo del  producto en general.

## Archivos .ipynb del proceso de selección, EDA, etiquetado y modelado


Análisis y preparación de los datos para el proceso de modelado
- Validación_muestras_deepglobe.ipynb
- DeepGlobe_TilesGeneration.ipynb
- Labeling_images_DeepGlobe.ipynb
 
Exploración y análisis
- EDA_DeepGlobe_128.ipynb

Pre-procesado
- Preprocess_Transformers_DeepGlobe_128.ipynb

Modelo multiclase de clasificación
- Multiclass_Model_Transformers_DeepGlobe_128.ipynb

Modelo binario de clasificación
- Binary_Classification_Transformers_DeepGlobe-128.ipynb

Modelo binario de segmentación
- Binary_Segmentation_128.ipynb

## La carpeta app contiene los ficheros de la app web para visualizar la demo

La carpeta app contiene los ficheros correspondientes a la web app desarrollada para aplicar el modelo binario de clasificación sobre mapa.

## Documentación y links
El PDF Documento_Proyecto_Cultivos.pdf contiene toda la información acerca del desarrollo de la pimera iteración y maqueta básica de este proyecto.
El PDF Presentación_Proyecto_Cultivos.pdf contiene la presentación del proyecto.

Se adjuntan además los siguientes links a youtube en los que se muestran tres demos de la maqueta con diferentes niveles de zoom:
- Demo zoom cercano: https://youtu.be/ZHs3ccCZQw0
- Demo zoom intermedio: https://youtu.be/m24wlfd1JVY
- Demo zoom lejano: https://youtu.be/0RL0OAIRaBY

- Video presentación: https://youtu.be/L0goZP3UrZ4


