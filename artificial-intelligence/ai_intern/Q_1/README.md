# Question #1


## Deep learning analytical thinking

### Problem description:

In 2024, several UK drivers were wrongly fined up to £105 for failing to pay the Dartford Crossing toll, despite never having used the crossing. The issue stemmed from ANPR cameras misreading vehicle licence plates, leading to incorrect penalty charge notices being issued. For instance, one driver received fines for vehicles with plates differing by a single character from her own, and another was fined for a vehicle of a similar make and color but with a slightly different registration number.

<div style="text-align: center;">
  <img src="../imagery/plates.png" alt="Plate 1" width="20%" style="display: inline-block; margin-right: 2%;">
  <img src="../imagery/ocr_plates.png" alt="Plate 2" width="20%" style="display: inline-block;">
</div>


#### Interesting Links:
- [Drivers wrongly fined £105 after number plate cameras get details wrong](https://www.inyourarea.co.uk/news/drivers-wrongly-fined-105-after-number-plate-cameras-get-details-wrong)
- [Drivers hit with Dartford Crossing fines despite never using it amid number plate camera issues](https://www.gbnews.com/lifestyle/cars/drivers-fined-dartford-crossing-never-used)


### Assumptions:
- System was being built with a Deep Learning based model (not traditional computer vision methods) 


### Questions:

Please provide a document (and extra data if any) answering each question below; please note that no practical exercises are required here (just theoretical analysis) but feel free to add any.

1. What do you think it occurred during this model development (trainning & evaluation)? 

Los errores que se describen en los artículos apuntan a problemas de generalización del modelo, probablemente causados por overfitting o por un dataset poco diverso.
Es posible que el sistema se entrenara con imágenes muy homogéneas (matrículas bien iluminadas, tipografía clara y sin obstrucciones), lo que provoca que tenga dificultades para reconocer correctamente matrículas reales en condiciones distintas (desgastadas, parcialmente tapadas, con poca iluminación, etc.).

Además, como se observa en los casos reales mencionados, existe un problema en el tratamiento de falsos positivos, como confundir una "O" con un "0" o una "B" con un "8", lo cual puede llevar a errores críticos que el modelo no penaliza adecuadamente durante la evaluación.

2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks

Opción 1: Ampliar el conjunto de datos para hacerlo más variado, lo que permitiría que el modelo generalice mejor y reduzca el sobreajuste.
Ventaja: Mejora la capacidad del modelo para adaptarse a diferentes casos reales.
Desventaja: Requiere más tiempo y recursos para el entrenamiento debido al aumento en la cantidad de datos.

Opción 2: Aplicar técnicas de data augmentation para simular condiciones reales como baja iluminación, desenfoque o rotaciones. Esto ayudaría al modelo a manejar mejor las variaciones propias de entornos reales sin necesidad de recopilar datos adicionales.
Ventaja: Incrementa la robustez del modelo frente a variaciones en los datos sin necesidad de recolectar más muestras.
Desventaja: Estas técnicas no reemplazan completamente la diversidad y complejidad de datos reales, por lo que el modelo podría no aprender algunos casos específicos.

Opción 3: Penalizar más los falsos positivos durante la fase de entrenamiento para minimizar los errores críticos, como confundir matrículas. Esto haría que el modelo sea más conservador a la hora de predecir una matrícula completa cuando no está seguro.
Ventaja: Reduce errores críticos que pueden tener un alto impacto en la aplicación.
Desventaja: Es necesario definir cuidadosamente qué errores deben ser penalizados, lo cual requiere un análisis previo y puede complicar el diseño del modelo.

3. What do you think it will occur when running this AI in a different country with different plates formats? How would you ensure system accuracy?

Uno de los principales problemas es que el sistema podría fallar al enfrentarse a matrículas con formatos diferentes a los utilizados durante el entrenamiento, ya sea por variaciones en colores, cantidad de caracteres o disposición de números y letras.

Para garantizar la precisión, se debería entrenar un modelo específico para el país objetivo, adaptándolo a las particularidades del formato local de matrículas.

Una alternativa sería partir de un modelo preentrenado con características generales comunes, como la forma rectangular de las matrículas, y luego ajustar el modelo con datos específicos del nuevo país.

Además, se podrían incorporar métodos de detección previa del país o formato de matrícula antes de procesar las imágenes, de modo que el sistema seleccione el modelo o parámetros adecuados para cada caso.

4. Do you know any OCR (Optical Character Recognition) algorithms (Deep learning based) that could be used here?

No tengo experiencia práctica con algoritmos específicos para este tipo de problema, pero teóricamente conozco la CRNN (Convolutional Recurrent Neural Network). Esta combina una CNN para extraer características visuales con una RNN para modelar secuencias, lo que la hace adecuada para tareas de reconocimiento de texto en imágenes. Sin embargo, no podría asegurar con certeza su efectividad en este caso particular.

5. Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)

En uno de los proyectos, trabajamos con una base de datos que contenía múltiples imágenes del mismo objeto, capturadas desde diferentes ángulos y posiciones. El objetivo del sistema era recibir una imagen de entrada y devolver las imágenes más similares dentro de la base de datos. Para ello, utilizamos redes neuronales convolucionales (CNNs) para extraer características de las imágenes y aplicamos métricas de similitud, como la distancia euclídea, para identificar coincidencias. Este proyecto me permitió profundizar en técnicas de extracción de características y normalización de imágenes.

Una de las dificultades fue la variabilidad de condiciones entre imágenes (cambios de iluminación, fondo o resolución), lo que afectaba la calidad de los embeddings. Para solucionarlo, ajustamos el preprocesamiento aplicando filtros de normalización para estandarizar mejor los datos de entrada.

En otro proyecto, desarrollamos un sistema para la clasificación automática de imágenes, asignándoles etiquetas como “gato”, “coche” o “montaña”. Además, el sistema debía verificar la corrección de estas etiquetas, validando así la predicción. Esto implicó trabajar tanto en la clasificación como en la verificación, utilizando técnicas de aprendizaje supervisado y métricas de evaluación como precisión y recall. Gracias a este proyecto, reforcé mis conocimientos en entrenamiento de modelos, validación de datos y detección de errores en predicciones automáticas.

Durante este proyecto, una de las principales dificultades fue que algunas clases se detectaban muy mal, debido al desequilibrio en la cantidad de ejemplos por clase y a la alta similitud visual entre ciertas categorías. Para solucionarlo, aplicamos data augmentation específicamente en las clases problemáticas y realizamos un análisis de errores para identificar confusiones recurrentes. Esto nos permitió ajustar tanto los datos como el modelo, mejorando así su rendimiento en las clases más conflictivas.

En ambos casos, mi rol consistió en diseñar la arquitectura del modelo, preparar los datos (incluyendo normalización y aumento de datos) y evaluar el rendimiento del sistema.