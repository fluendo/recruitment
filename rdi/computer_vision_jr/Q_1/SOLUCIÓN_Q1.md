# Q1

## 1. ¿Qué crees que ocurrió durante el desarrollo de este modelo (entrenamiento y evaluación)?

Se me ocurren un par de posibles problemas que ocurrieron en el entrenamiento del modelo:

- Falta de diversos escenarios: Posiblemente, el modelo no haya sido entrenado con una cantidad suficiente de datos que representen una diversidad de situaciones y condiciones de juego

- Overfitting: Es posible que el modelo se haya ajustado demasiado las características del data training, haciendo que identificara erróneamente la cabeza de la persona calva. En otras palabras, el modelo pudo haber aprendido a identificar características muy específicas presentes en los datos de entrenamiento y generalizar erróneamente estas características a todos los casos, incluso cuando no son relevantes.

Pero se me ocurre otra posibilidad para la evaluación del modelo:

- Métricas de evaluación inadecuadas: Puede ser que los evaluadores del problema se hayan centrado en métricas del modelo que no representan la complejidad y variedad de escenarios.


## 2. ¿Cómo arreglarías este comportamiento? Por favor proporciona al menos 2 opciones explicando sus pros y contras.

Se me ocurren tres opciones para arreglar los problemas anteriores:

- Aumentar y diversificar los datos de entrenamiento: El modelo será más robusto (generalizará mejor) pero recopilar dichos datos puede llevar tiempo y ser costoso.

- Aplicar técnicas de regularización: Reducirá el overfitting y el modelo será más robusto en general, pero incrementará el tiempo de entrenamiento y la complejidad del modelo

- Usar métricas de evaluación más adecuadas (Usar una variedad de métricas y no solo unas pocas): Mejor evaluación y comprensión del modelo, pero, obviamente, la evolución será más compleja de hacer y más dificil de entender.

Pero también se me ocurre una solución que si bien no arreglaría los problemas anteriores, pero sí ayudaría al modelo a identificar el balón:

- Dar contexto al modelo, se podría intentar diseñar el modelo para que tenga en cuenta información contextual. Por ejemplo, un balón de fútbol rara vez estará por encima de los jugadores durante mucho tiempo, mientras que la cabeza de un árbitro generalmente estará en una posición más alta, o se le podría dar un máximo de píxeles donde buscar el balón, ya que las personas estamos limitados en lo que es la fuerza. Esto podría ayudar a mejorar la precisión del modelo, pero podría ser más difícil de implementar, especialmente si el modelo no fue diseñado para tener en cuenta esta información desde el principio.

## 3. Extra: ¿Conoces algún algoritmo de seguimiento (basado en aprendizaje profundo) que podría ser utilizado aquí?

Utilizaría uno de estos algoritmos para detectar el balón:

- YOLO (You Only Look Once): Es un algoritmo de detección de objetos en tiempo real que puede identificar y rastrear objetos en videos con solo una mirada a la imagen. Sin embargo, tiende a tener más falsos positivos, y a veces le cuesta detectar objetos pequeños que están cerca unos de otros.

- SSD(Single-shot detector): El algoritmo SSD también examina la imagen una sola vez, pero difiere de YOLO en su manejo de las diferentes escalas de los objetos. Mientras que YOLO solo mira una cuadrícula de una cierta escala, SSD examina varias capas de la imagen para detectar objetos de diferentes tamaños. Esto hace sea mejor para detectar objetos pequeños, pero más lento (aun así se puede utilizar en tiempo real).

Pero si se quiere que el seguimiento sea más preciso, podríamos combinarlo con un algoritmo más robusto de seguimiento, como:

- Deep SORT (Deep Simple Online and Realtime Tracking): Es un robusto algoritmo utilizado para el seguimiento de objetos en videos. Es muy bueno, pero, sin embargo, es complejo computacionalmente y es sensible a la calidad de video.

La combinación de estos puede ser un poco más lenta que utilizar solo uno, pero los resultados serían bastante mejores.
