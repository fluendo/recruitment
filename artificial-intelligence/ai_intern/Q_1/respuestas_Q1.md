# Respuestas – AI Intern Test – Q_1

**Nombre:** Maria Sans Bosch 

**Fecha:** 30/05/2025

**Puesto:** AI Intern  

---

## 1. ¿Qué crees que ocurrió durante el desarrollo del modelo (entrenamiento y evaluación)?

Durante el desarrollo del modelo de reconocimiento de matrículas basado en Deep Learning, probablemente ocurrieron varios problemas que contribuyeron a errores de predicción:

1. **Datos de entrenamiento insuficientemente diversos**: Es posible que el dataset utilizado para entrenar el modelo no contuviera ejemplos suficientemente diversos, en terminos de cambios de luz, distintos ángulos de cámara, matrículas sucias o dañadas, diferentes fuentes de vehículos de países diferentes, de años diferentes (puesto que los vehículos antiguos presentan una nomenclatura diferente). Esto lleva a una baja capacidad de generalización.
Una opción para solucionar este problema sin tener que recurrir a otros datasets, podría ser aplicar técnicas de data augmentation como: rotación, flipping, ajuste del brillo, etc.

2. **Errores en el preprocesamiento o etiquetado**: Si los datos estaban mal anotados (es decir, la Ground Truth no era correcta), o si no se normalizaron adecuadamente (por ejemplo, imágenes demasiado grandes o pequeñas, deformadas, etc.), el modelo podría haber aprendido patrones incorrectos.

3. **Evaluación en condiciones ideales**: Es común que se evalúe el modelo con imágenes limpias o artificiales, lo cual no refleja las condiciones reales. Si no se validó con imágenes del entorno real (como las captadas por cámaras de tráfico), se sobreestimó el rendimiento del modelo.

4. **Elección incorrecta del modelo**: Aunque no hay un modelo correcto, podría ser que no se hayan probado modelos suficientemente buenos o eficientes para la detección de matrículas. Según lo que he estudiado en la asignatura de Procesamiento de Imagen i Visión artificial, existen métodos  eficientes para la detección de matrículas, como YOLO (You Only Look Once), SSD (Single Shot MultiBox Detector), o Faster R-CNN.

5. **Falta de verificación lógica posterior al OCR**: No parece haberse aplicado una verificación posterior para comprobar que las matrículas reconocidas sean válidas o posibles.

---

## 2. ¿Cómo solucionarías este comportamiento?  


### Opción A: Mejorar los datos de entrenamiento con augmentations y casos reales

**Descripción:**  
El problema principal del sistema podría haber sido un entrenamiento con datos artificiales o limitados. Una solución directa sería **incrementar y diversificar los datos de entrenamiento** usando imágenes reales de matrículas en condiciones adversas: iluminación desigual, suciedad, ángulos inclinados, placas parcialmente visibles, etc. Además, usar técnicas de *data augmentation* (rotación, blur, ruido, cambio de brillo/contraste) ayuda a mejorar la capacidad del modelo para generalizar.

**Ventajas:**
- Mejora la robustez del modelo ante condiciones reales.
- Permite que el modelo aprenda errores típicos de lectura OCR (por ejemplo, "O" vs "0").
- Se mantiene completamente en el ámbito de Deep Learning, sin tener que recurrir a técnicas más avanzadas de Visión Artificial.

**Inconvenientes:**
- Requiere más recursos: tiempo, capacidad de anotación manual, potencia de cómputo.
- Es muy difícil conseguir datasets reales suficientemente representativos, así como muy grandes.

---

### Opción B: Dividir el sistema en dos partes: detección y reconocimiento
**Descripción:**  
Otra opción es separar el problema en dos pasos: primero, detectar la matrícula en la imagen (por ejemplo usando un modelo como YOLO) y luego aplicar un modelo OCR solo en esa parte de la imagen. Esto permite que cada modelo esté especializado en una tarea más simple, lo que puede dar mejores resultados.

**Ventajas:**
- Es más fácil detectar una matrícula si no hay que reconocer texto al mismo tiempo.
- Muchos modelos preentrenados ya existen y se pueden usar directamente.

**Desventajas:**
- Hay que coordinar bien ambos pasos (detección y reconocimiento).
- Requiere un poco más de programación y prueba para hacer funcionar todo junto.

---

### Opción C: Usar una herramienta OCR ya entrenada con la que ya se haya demostrado que funciona correctamente

A veces, es más fácil y eficiente usar modelos ya entrenados y optimizados para tareas como el reconocimiento de texto, en lugar de intentar crear uno desde cero. Esto es especialmente útil cuando se tienen recursos limitados. 

Investigando un poco, encontré dos herramientas bastante populares que ya vienen entrenadas y listas para usar: **PaddleOCR** y **EasyOCR**. Ambas permiten reconocer texto en imágenes (incluyendo matrículas de vehículos) con muy buenos resultados, y requieren muy poco código para ponerlas en marcha.

**Ventajas:**
- Ahorra mucho tiempo y recursos (es sostenible tanto económicamente como ecológicamente)
- Compatible con muchos formatos y lenguajes.
- Se ha entrenado con datos que quizás sería imposible obtener de otra forma. 

**Desventajas:**
- No tienes control sobre el modelo ni cómo fue entrenado ni qué datos se usaron por lo que se podría convertir en una "caja negra" sobre la cuál no puedes decir nada. 
- Aunque estas herramientas son muy prácticas, si el sistema empieza a fallar o da errores en ciertos contextos, no podremos saber bien por qué, ni ajustar internamente el modelo. Esto puede ser una limitación si se quiere desplegar a gran escala. 
- Para el caso concreto que se está analizando, creo que no sería la mejor opción, ya que estamos hablando de una herramienta que se utiliza para poner multas a ciudadanos. Al estar el gobierno involucrado en el proyecto, es fundamental que tenga un control total sobre las herramientas que se usan, tanto a nivel técnico como legal.



---

## 3. ¿Qué ocurrirá al usar esta IA en otro país con distintos formatos de matrícula?  
Si se utiliza este mismo sistema de reconocimiento de matrículas en otro país, lo más probable es que **el rendimiento del modelo baje considerablemente**. Esto se debe a que los modelos de Deep Learning suelen aprender patrones específicos del conjunto de datos con el que han sido entrenados. Por tanto, si se han usado solo (o mayoritariamente) datos de matrículas de UK es seguro que el modelo va a detectar mal muchas más matrículas de otro país. 
Un aspecto muy importante a destacar es que de un país a otro no solo cambia el número de carácteres permitidos en la matrícula, sinó también las longitudes entre carácteres, la tipografía de las letras, existen diferentes alfabetos con letras diversas, el uso de símbolos o banderas regionales en la matrícula, etc. 

Como dato curioso, en Dubái es posible personalizar tu matrícula con números y letras específicas (hasta un número limitado de carácteres), e incluso pagar mucho dinero por combinaciones muy exclusivas. Esto haría que un modelo de reconocimiento de matrículas entrenado solo con formatos fijos fallara fácilmente en ese contexto, ya que no podría validar correctamente estas matrículas "outliers" que se salen del estándar. En estos casos, sería necesario hacer un sistema mucho más robusto para que acepte entradas más flexibles, o bien incluir excepciones específicas en la lógica de validación.


### ¿Cómo garantizaría la precisión?

1. **Ampliar el dataset con ejemplos internacionales**  
   Se pueden incorporar imágenes de matrículas reales de otros países al dataset de entrenamiento. 

2. **Entrenar modelos específicos por país (o región)**  
   En caso que la IA solo se usara en otro país concreto, una alternativa es crear varios modelos, cada uno especializado en un país. Por ejemplo, un modelo para matrículas europeas, otro para latinoamericanas, etc. Esto permite una mayor precisión al trabajar con estructuras fijas conocidas.

3. **Usar transfer learning con fine-tuning**
En lugar de entrenar un modelo desde cero, se puede aprovechar el modelo ya entrenado de Reino Unido (o uno que funcione un poco mejor) y hacer un **fine-tuning** con datos de otro país. Esto significa ajustar las últimas capas del modelo con un pequeño conjunto de datos locales, lo que permite adaptar el sistema a nuevos formatos de matrícula sin requerir tantos datos ni tiempo de entrenamiento.

4. **Aplicar reglas locales de validación**  
   Una vez obtenida la predicción se puede verificar si cumple con el formato válido del país (por ejemplo, cantidad de caracteres, uso de guiones, letras permitidas...). Esto ayudaría a filtrar errores comunes. Sin embargo, no funcionaría en todos los países (por ejemplo en Dubái).

En resumen, para que un sistema sea preciso a nivel internacional, es importante tener en cuenta tanto el **diseño técnico** como el **contexto local**, y adaptar el modelo a los requisitos de cada país en el que se vaya a implementar.


---

## 4. ¿Conoces algún algoritmo OCR basado en Deep Learning que pueda usarse aquí?

Por lo aprendido en clase, el único modelo que podría ser adaptable a este problema sería **CRNN (Convolutional Recurrent Neural Network)**. Este modelo combina CNNs (para extraer características visuales de la imagen) con RNNs (para entender el orden de los caracteres). Es uno de los enfoques más utilizados para tareas de OCR cuando el texto está más o menos en línea recta, como es el caso de una matrícula.

Por los demás métodos, investigando por Internet, como he mencionado anteriormente, he encontrado **PaddleOCR**, una herramienta de código abierto que ya viene entrenada con grandes datasets multilingües, y **EasyOCR**, una librería de Python que soporta muchos idiomas.

---

## 5. Explica un proyecto de Visión Artificial / Inteligencia Artificial en el que hayas participado.

Actualmente he finalizado un proyecto en colaboración con el Hospital Clínic de Barcelona dentro de la asignatura de tercero **Proyectos de Ingeniería** del grado en Ciencia e Ingeniería de Datos. 

El proyecto, llamado *PredictIA*, consistía en el desarrollo de un modelo predictivo entrenado con datos médicos reales, capaz de detectar el riesgo de que un paciente con cáncer hematológico y neutropenia sufra una bacteriemia, una infección grave en la sangre. El objetivo principal era ofrecer al personal médico una herramienta rápida y personalizada que les permitiera anticiparse a posibles infecciones y mejorar así la toma de decisiones clínicas y la atención al paciente.


Mi rol se centró, junto a una compañera, en toda la parte de **preprocesamiento y preparación de los datos clínicos**. Esta etapa fue especialmente importante, ya que trabajamos con datos reales de pacientes extraídos de diferentes fuentes hospitalarias, muchos de ellos incompletos, no estructurados o en formatos muy heterogéneos. También colaboré en la selección de variables utilizadas por el modelo y en la evaluación de su rendimiento, además de participar en sesiones de validación con profesionales del hospital.

Durante el desarrollo nos enfrentamos a varios retos: desde **valores nulos e inconsistencias** en los datos, hasta el hecho de que el conjunto estuviera desbalanceado, lo que complicaba la capacidad del modelo para generalizar correctamente. Además, trabajar con datos médicos reales supuso un gran reto técnico y de interpretación, ya que al no tener formación sanitaria previa, tuvimos que investigar mucho y mantener reuniones frecuentes con el personal del Clínic para entender correctamente el significado, estructura y rangos de las variables clínicas.

Para superar estos retos aplicamos un criterio sistemático en el tratamiento de los datos: eliminamos variables con más del 30% de valores nulos e imputamos las que tenían menos con la media. Estandarizamos y reconvertimos los archivos recientes a un formato compatible, y, para reducir la carga de entrada para los médicos sin perder precisión, aplicamos análisis con **SHAP values** y logramos mantener el 95% del poder predictivo usando solo 42 variables en lugar de las 111 iniciales. Finalmente, tras comparar varios modelos con validación cruzada mediante **GridSearch**, seleccionamos **XGBoost** por su equilibrio entre rendimiento, robustez frente al desbalanceo y baja tendencia al *overfitting*.

El resultado fue un prototipo funcional con una **interfaz web desarrollada en Streamlit**, donde el personal médico puede introducir información clínica y obtener una predicción del riesgo, acompañada de gráficos interpretables y comparativas visuales con otros pacientes. El modelo superó el 80% de precisión en test y fue validado positivamente por el equipo médico del hospital.

Este proyecto me permitió ver de cerca cómo se aplica la inteligencia artificial en un entorno clínico real y comprobar la enorme importancia que tiene la calidad de los datos en cualquier solución basada en IA. Ha sido una experiencia muy enriquecedora que ha reforzado mi interés por esta área y me ha motivado a seguir aprendiendo y aportando en proyectos reales donde la tecnología tenga un impacto positivo.


---

