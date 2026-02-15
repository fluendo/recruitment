# AI Tech Assessment: Artificial Intelligence Intern

### Problem description:

### In 2024, several UK drivers were wrongly fined up to £105 for failing to pay the Dartford Crossing toll, despite never having used the crossing. The issue stemmed from ANPR cameras misreading vehicle licence plates, leading to incorrect penalty charge notices being issued. For instance, one driver received fines for vehicles with plates differing by a single character from her own, and another was fined for a vehicle of a similar make and color but with a slightly different registration number.

# 1. What do you think it occurred during this model development (trainning & evaluation)? 

Durante el entrenamiento se basaron únicamente en imágenes limpias para la tarea de detección (encontrar los píxeles correspondientes a la placa) y reconocimiento (identificar letras y dígitos), sin ruido ni movimiento. Como resultado, el modelo aprendió a distinguir caracteres con bordes demasiado ideales y no toleró variaciones. En la evaluación, con el coche en movimiento y un entorno cambiante (vibraciones, desenfoque, contraste variable), la placa detectada y reconocida fallaba con frecuencia. Además, no se incorporaron datos V2I (comunicación directa entre vehículo e infraestructura), de modo que nunca se validaron lecturas ópticas contra la matrícula real transmitida inalámbricamente. Esto provocó que el modelo sobreajustara patrones irreales y la verificación V2I, de haber existido, habría identificado esos errores.

# 2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks

Se instala una RSU en el carril de peaje y cada vehículo lleva un OBU que, al pasar, emite su matrícula por DSRC/C-V2X; la infraestructura recibe esa señal y la compara con la lectura ANPR, descartando automáticamente multas si no coinciden. Esto prácticamente elimina los falsos positivos al verificar la matrícula real sin depender de la calidad de imagen ni de la iluminación. Sin embargo, exige distribuir o instalar OBU en los coches, asegurar el cifrado y la privacidad de la comunicación, desplegar y mantener las RSU, y gestionar la interoperabilidad de distintos fabricantes.

Se define un nivel mínimo de confianza (por ejemplo, 95 %) para la lectura óptica; si la lectura no alcanza ese umbral, la imagen se reenvía a un OCR secundario (como un modelo Transformer) o a revisión manual. De este modo, se filtran las lecturas dudosas y se corrigen antes de emitir multas, sin necesidad de hardware adicional, y cada caso complicado sirve como “hard example” para mejorar el modelo. No obstante, en horas punta puede saturarse el OCR secundario o el equipo de revisión, introduciendo retrasos y duplicando los recursos de cómputo necesarios para procesar cada imagen.

# 3. What do you think it will occur when running this AI in a different country with different plates formats? How would you ensure system accuracy?

En otro país, el modelo ANPR entrenado para matrículas británicas fallará al detectar formatos distintos (p. ej. “1234 ABC” en España o “浙F 88888” en China), fuentes tipográficas diferentes y variaciones de colores y relieves. Esto provocará altas tasas de errores de lectura porque los patrones de CNN/CTC no reconocerán la distribución ni el estilo de los caracteres nativos, y la validación por formato (regex) no coincidirá.

Para garantizar precisión en un nuevo país, es imprescindible entrenar modelos específicos con datasets locales de matrículas, ya que resulta muy complejo desarrollar un único modelo genérico que funcione correctamente con todos los formatos y tipografías nacionales.

# 4. Do you know any OCR (Optical Character Recognition) algorithms (Deep learning based) that could be used here?

No tengo experiencia directa en OCR, pero he investigado el tema y he seleccionado las siguientes soluciones:

- CRNN (Convolutional Recurrent Neural Network + CTC): Frecuentemente implementado en dispositivos IoT que se conectan por redes 4G/5G, procesando OCR en el borde para minimizar la latencia y aprovechar la capacidad de cómputo distribuido típico de entornos teleco.

- Transformer-based OCR (por ejemplo, TrOCR): Usable en arquitecturas de baja latencia como MEC en 5G, donde el modelo puede recibir y procesar flujos de video en tiempo real, beneficiándose de la infraestructura de red y la cercanía al usuario para obtener resultados rápidos.

# 5. Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)

En mi último proyecto de Computer Vision, desarrollé un sistema que detecta vehículos tras barreras verticales y segmenta granos de arroz en zonas con distinta iluminación usando ecualización, DFT para remover patrones de barras, pirámide multi-resolución y filtrado morfológico para sombras. Mi rol fue diseñar el preprocesamiento (cuantización, histogramas, DFT, diezmado/interpolación) y parametrizar filtros. Un reto fue la confusión de bordes por sombras, resuelto con umbral en HSV y apertura morfológica, y la velocidad real, solventada usando pirámide de imágenes e interpolación para refinar solo regiones críticas.