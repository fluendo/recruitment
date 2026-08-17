# Question #1

## Analytical thinking

### Problem description:

With the enforcement of the new EU AI Act, AI systems that process biometric data including face detection and recognition) are now subject to stricter transparency, risk management, and privacy-preserving requirements. In many real-world video analytics scenarios (smart cities, retail, sports analytics, workplace safety, and media processing), camera streams may capture identifiable faces even when identity is not required for the system’s core purpose. Under the new regulatory framework, this creates legal and operational risk unless appropriate safeguards are applied.

To address these constraints, we have designed a **face anonymization** solutuin based on a CNN-based face detector (YOLO-style architecture) to locate faces in images and video streams and automatically applies anonymization filters (blur, pixelation, or masking). The objective is to preserve scene utility while preventing personal identification, supporting privacy-by-design and privacy-by-default principles required by the EU AI Act.

However, deploying such a system in production raises several technical challenges: selective anonymization (only specific identities), fairness and bias in detection accuracy across demographics, robustness under difficult visual conditions, computational efficiency for real-time processing, and proper MLOps governance. This test explores how to analyze, diagnose, and improve such a Deep Learning–based anonymization system from both model and infrastructure perspectives.

<div style="text-align: center;">
  <img src="../imagery/anonym_1.png" alt="Plate 1" width="60%" style="display: inline-block; margin-right: 2%;">
</div>

In this test, we will analyze real problems and dilemas related with this product in order to evaluate how would you act and reason in front of them.

### Assumptions:
- Face detection is CNN-based and runs on images and video
- The system must comply with EU AI Act privacy and risk-control expectations
- There is no constraint in which model architecture can be used for both reidentification & depth estimation
- We want to reduce computational cost so the lighter the models, the better

### Questions:

Please provide a document (and extra data if any) answering each question below; please note that no practical exercises are required here (just theoretical analysis) but feel free to add any.

1. **How can we improve face detection by adding reidentification to anonymize only specific individuals while keeping computational cost low?**

Inicialmente había llegado a la conclusión de que la clave para integrar la re-identificación sin disparar el coste está en el tracking como portador de la identidad del individuo. Es decir, ejecutando distintas etapas con diferentes frecuencias: detección cada N frames, tracking en cada frame, y reconocimiento una única vez por track.

No obstante, este planteamiento al analizarlo con más detalle puede ser problemático. Primero, saltar detecciones es arriesgado, ya que la performance de un YOLOv8-face detectando caras es mayor que un tracking del individuo, ya que el bbox puede derivar progresivamente y, sobre todo, una cara que entra en escena entre dos detecciones no existiría hasta la siguiente. En un sistema de anonimización eso sería una potencial fuga de privacidad.

Además, el ahorro real no está en la detección sino en el reconocimiento. Un detector ligero tipo SCRFD-2.5GF a 640² cuesta ~2,5 GFLOPs por frame, mientras que un ArcFace R50 a 112² cuesta ~6,3 GFLOPs por cara: con 10 personas en escena, reconocer cada frame supone ~63 GFLOPs frente a 2,5 de la detección. Por tanto lo que hay que amortizar por track es el embedding, no la detección.

Así pues, como propuesta haría lo siguiente:

**Cada frame**
- Detección de caras: bbox + landmarks (ojos, nariz, comisuras).
- Tracking de la cara conjuntamente con su ID.
- Aplicación del filtro de anonimización según el ID (blur, pixelation o masking).

**Amortizado por track**
- Face alignment: a partir de los features de la cara, normalizar la cara para obtener un embedding preciso del individuo.
- Embedding: no uno solo, sino las primeras K detecciones que superen un filtro de calidad (tamaño en píxeles y pose aproximadamente frontal), agregadas mediante media de los vectores normalizados. Un único embedding fija la identidad de todo el track a partir de un solo crop, y si ese crop es malo o hay un ID switch el error se propaga hasta el final sin corrección posible.
- Matching contra la galería de identidades registradas mediante cosine similarity, con umbral configurable y asignación global para evitar que dos caras del mismo frame reciban la misma identidad.
- Re-verificación periódica cada M frames contra la identidad ya asignada, y re-identificación completa cuando el track se pierde por oclusión o salida de escena y reaparece.
- Ante duda el sistema anonimiza por defecto (un falso negativo expone a quien había que proteger).

Por último, un punto que merece señalarse aunque no sea estrictamente técnico. Un detector que solo difumina no realiza identificación biométrica, porque no compara contra ninguna referencia; al añadir una galería de identidades y matching, sí la realiza. El EU AI Act excluye la verificación cuya única finalidad es confirmar que alguien es quien dice ser, pero aquí la comparación es contra una galería, de modo que el sistema entra como alto riesgo, incluso cuando la finalidad es la privacidad. 

En cuanto al almacenamiento, la galería es el elemento más sensible del sistema, funciona como allowlist: quien está registrado es quien no se anonimiza. Así, si el matching falla o la identidad no se resuelve, el resultado por defecto es anonimizar. Aún así, debe estar cifrado, con acceso restringido y con capacidad de borrar una identidad concreta para poder atender derechos de supresión.

2. **How can we extend an anonymization pipeline with depth-based filtering to blur only background objects efficiently?**

Según el entorno, para difuminar el fondo y mantener nítidas solo a las personas, se podría plantear un detector o segmentador de personas invirtiendo la máscara, en lugar de un filtrado por profundidad. Este approach hace perder información de la escena, ya que conservando solo las personas no habría contexto sobre sus acciones. Además, en un sistema de anonimización se debería tener control sobre a partir de qué distancia hay que enmascarar o no, y no solo sobre qué clases se enmascaran (personas o background). Hay varios métodos basados en semantic segmentation aplicables a este enfoque, como DeepLab o BiSeNet.

Otra forma de enmascarar el background, en este caso por movimiento en lugar de por clase o por distancia, sería con un background subtraction adaptativo, donde el coste computacional sería ínfimo. Eso sí, con muchos condicionantes: la cámara debería ser estática, es sensible a cambios de iluminación y es ciega a los objetos estáticos.

En cualquier caso, el filtro de profundidad es el único que da control real sobre la distancia a la que se debe anonimizar, independientemente de la clase de objeto. La implementación eficiente sería con un modelo de depth estimation monocular sobre el frame downsampleado, ya que la profundidad es una propiedad de baja frecuencia espacial. 

Puntos a tener en cuenta:

- Elección del modelo según el umbral que se necesite: Los modelos relativos (Depth Anything V2-Small, MiDaS-small) devuelven una profundidad inversa sin escala: valor alto = cerca. Con ellos el umbral debe fijarse de forma relativa al propio mapa, no en metros. Si el requisito es "anonimizar a partir de X metros" hace falta un modelo métrico como Depth Pro o UniDepth.
- Bordes: El interior de los objetos es baja frecuencia, pero las discontinuidades de profundidad no lo son. Un upsample bilineal desde 1/4 de resolución produce halos, con blur que se come el borde del sujeto o se sale de él. Se corrige usando el frame a resolución completa como guía, que es mucho más barato que inferir la profundidad a resolución nativa.
- Frecuencia de inferencia: A diferencia de la detección de caras, aquí sí se puede inferir cada N frames. 
- Coste: Como no hay constraint de arquitectura, lo más eficiente es un backbone compartido con dos heads (detección de caras + depth) en lugar de dos forward passes independientes.

Finalmente, combinar la máscara de profundidad con la de caras sería lo más adecuado, dando prioridad a las caras:

`blur = (mask_caras_denylist OR mask_profundidad) AND NOT mask_caras_allowlist`

Conviene subrayar que el blur de fondo es un efecto de escena, no una garantía de anonimización: una cara pequeña al fondo puede quedar difuminada de forma insuficiente y seguir siendo reidentificable. Por eso el filtro de profundidad complementa al de caras, pero no lo sustituye.

3. **Imagine we have this case: our face detection system does not detect properly (bad accuracy) dark-skins nor bald-people; how would you improve system accuracy?**

Ante todo cabe comentar que las pieles oscuras no solo se detectan peor por culpa del sesgo en los datos de entrenamiento, sino también por un motivo físico ya que reflejan menos luz. Por tanto, según el entorno y la posibilidad de actuación en la captura de las imagenes, recomendaría un sistema con HDR/multi-exposition (con la contrapartida de que capturar y fusionar 2-3 exposiciones por frame multiplica el ancho de banda a transmitir y el coste de memoria, lo que suele obligar a sacrificar frame rate o resolución), evitar el clipping en las zonas oscuras y en entornos de poca luminosidad valorar una iluminación adicional, ya sea con un foco con detección de presencia o con iluminación IR. Son medidas que no requieren reentrenar y que cubren la causa del problema (o como mínimo con las dark-skins).

En cualquier caso, antes de intervenir sobre el modelo me aseguraría de obtener metricas de su performance por grupos. Es decir, evaluaría por slices, midiendo separadamente por tono de piel (usando una escala como la Monk Skin Tone Scale (MST), en lugar de etiquetas raciales) y por presencia o ausencia de cabello, con intervalos de confianza. De esta forma se evita que un fallo concentrado en un subgrupo quede oculto tras la media, por ejemplo: si de 10.000 caras evaluadas solo 300 son dark-skin y el detector falla en todas ellas, el buen rendimiento sobre las 9.700 restantes seguiría dando una métrica global alta y esa pérdida no destacaría. Por eso hay que evaluar cada subgrupo por separado (además de procurar usar datasets balanceados). 
Revisaría también dos cosas: si las caras problemáticas se están detectando pero por debajo del umbral de confianza (lo que se resolvería recalibrando) y si el modelo en producción ha sido comprimido ya sea con una quantization o un pruning (ya que ha podido alterar la performance).

Con respecto a la mejora del modelo, existen varias vías:

- Datos: reentrenar con un dataset más balanceado sin perder las características anteriores. La dificultad está en conseguir esos datos. Para el tono de piel, el data augmentation es viable y directo (variaciones agresivas de exposición, gamma, temperatura de color, balance de blancos y simulación de compresión). Para la calvicie es más difícil, ya que habría que modificar el cabello; hoy es factible con edición generativa por inpainting, aunque introduce sus propios artefactos y habría que validar que el modelo no aprende esos artefactos en lugar del rasgo.
- Diagnóstico de features y fine-tuning selectivo: usar saliency maps (Grad-CAM) para comprobar si el detector se apoya efectivamente en la región capilar como contexto. Si se confirma, una opción de bajo coste es hacer fine-tuning congelando el backbone y reentrenando solo las cabezas de detección con el dataset balanceado, lo que reduce el riesgo de olvido catastrófico. Si se quiere forzar invarianza explícita a un atributo, el mecanismo adecuado sería adversarial debiasing, penalizando que los features permitan predecir ese atributo.
- Cambiar el planteamiento de la detección: detectar cabeza completa en lugar de solo cara, o detectar cuerpo y de ahí derivar a la cabeza. Un detector de cabeza es estructuralmente menos dependiente de la textura capilar y además más robusto a oclusión.

Finalmente, considero que en un anonimizador el criterio de optimización no debería ser prioritariamente maximizar el accuracy medio sino minimizar el peor miss rate entre subgrupos, ya que un falso negativo deja a una persona sin proteger. 

4. **Imagine this other case: let's assume we have added several bald people imagery to our dataset and we still are not increasing the accuracy. What may be happening? How would you fix this?** 

En resumen: Revisar las métricas -> Revisar las annotations -> Revisar el test & trainning set  -> Verificar si el rasgo es realmente la causa -> Adversarial debiasing -> Replantear el modelo

- Revisar las métricas. Si se sigue mirando el valor medio en lugar del de los subgrupos, la mejora podría no apreciarse. Y sobre todo haría una inspección visual de los casos que fallan.
- Revisar las anotaciones, y en particular las bounding boxes. Comprobar que el dataset base y el nuevo siguen las mismas convenciones. Por ejemplo, si el dataset base trunca la caja a la altura del nacimiento del pelo y las imágenes nuevas se anotaron cubriendo la cabeza completa, el modelo recibe señales contradictorias: se penaliza el IoU y la Precision no sube aunque la cara se esté detectando.
- Revisar el test y training set. Vigilar si solo se amplió el training dataset, ya que la mejora puede existir y no ser medible con pocos casos. Si la muestra de calvos en test sigue siendo pequeña, el resultado estará dominado por la varianza (también estaría bien comprobar los intervalos de confianza). Por tanto, ampliar y diversificar el test set es parte del arreglo. Revisaría además posible data leakage. Y comprobaría que las imágenes de calvos tengan suficiente peso durante el entrenamiento, ya que unas pocas sobre decenas de miles quedan diluidas, en ese caso aplicaría oversampling o stratified batching. Otra opción complementaria es generar calvos artificiales a partir de los datos existentes.
- Verificar si el rasgo es realmente la causa. La calvicie puede estar asociada a llevar gorras o cascos, a brillos sobreexpuestos, y las imágenes nuevas pueden traer sus propias condiciones de captura. Por tanto, haría un counterfactual test comparando el rendimiento sobre las mismas caras con y sin pelo: si falla solo al ocultarlo, el rasgo es causal, y si falla en ambos casos el problema es otro. Es más fiable enmascarar el pelo existente que generarlo, ya que la generación introduce artefactos que pueden confundir el resultado.
- Adversarial debiasing. Si el modelo insiste en depender del pelo, se puede penalizar explícitamente que lo use: una head auxiliar que prediga la presencia de pelo, con el signo de su gradiente invertido (Gradient Reversal Layer), de modo que el backbone se penaliza si sus features apuntan a esa información. Como alternativa más simple, fine-tuning congelando el backbone y reentrenando solo las heads con un dataset balanceado.
- Replantear el modelo, como último recurso. Por ejemplo, pasar a un detector de cabeza completa en lugar de cara.

5. **Which MLOPs infrastructure would you design to manage this system? Just provide a high-level design and which tools would you use.**

Dos condicionantes marcan el diseño: no se pueden recolectar frames de producción para reentrenar, porque son datos biométricos, y la latencia es un criterio de aceptación tan vinculante como la precisión, al tratarse de vídeo en tiempo real.

- Experimentación: Weights & Biases para el tracking de experimentos, registrando las métricas por subgrupo demográfico y no solo el mAP medio, para que quede claro si una mejora global esconde una regresión en un subgrupo. Y Ray Tune para la búsqueda de hiperparámetros.
- Versionado: Además del código en Git, hay que versionar el dataset y las anotaciones, porque un modelo solo es reproducible si se sabe con qué datos se entrenó. Los splits por subgrupo deben quedar fijados y versionados, no recalcularse en cada evaluación. Su conservación se ampara en el EU AI Act, que habilita el tratamiento de datos sensibles para detectar y corregir sesgos.
- Empaquetado y CI/CD: Docker para empaquetar el modelo con sus dependencias y que funcione igual en cualquier máquina, con imágenes separadas para entrenamiento e inferencia. Y unos tests automáticos que se pasen antes de publicar cada versión nueva, comprobando que funciona bien en todos los subgrupos, que ninguno ha empeorado respecto a la versión anterior, que sigue funcionando después de aligerar el modelo.
- Despliegue: Para procesar vídeo en tiempo real el cloud no es suficiente, porque añade latencia y además implica enviar por internet caras sin anonimizar, que es justamente lo que se quiere evitar. Así que la inferencia debería correr en local, cerca de la cámara. El inconveniente es que con muchos equipos hay que llevar el control de qué versión tiene cada uno, y poder volver atrás si una versión sale mal.
- Monitorización: Por un lado el rendimiento (FPS, latencia, uso de recursos). Por otro la calidad del modelo, que aquí no se puede medir de forma directa porque no hay ground truth ni se pueden guardar imágenes, así que se sustituye por señales indirectas como la distribución de los scores de confianza, el número de caras detectadas por frame o la tasa de identidades no resueltas.
- Cierre del bucle: Como no se puede reentrenar con datos de producción, la mejora se apoya en un conjunto de evaluación interno con consentimiento, y en feedback del cliente en forma de informes de fallo, no de imágenes.

6. **From end user POV, which techniques would you use to improve final solution user experience?** 

Lo primero sería la estabilidad temporal del efecto. El parpadeo de la máscara es el fallo más visible, cada frame en que el blur desaparece es un frame en que la cara queda expuesta. Por ejemplo se podria poner un suavizado temporal de las bounding boxes, histéresis en el umbral y manteniendo la máscara unos frames más cuando se pierde una detección. También pondría un selector de efecto, teniendo en cuenta que el blur y la pixelación son reversibles hasta cierto punto con redes entrenadas para ello, así que conviene fijar una intensidad mínima configurable y ofrecer alternativas como el enmascarado sólido o la sustitución por rostro sintético, cubriendo completamente la cabeza.

Desde el punto de vista del operador, daría control y visibilidad sobre lo que está pasando: preview en tiempo real para ajustar parámetros sin procesar todo el material, corrección manual de fallos (marcar una cara no detectada o corregir una identidad mal asignada) e indicadores de estado como el número de caras detectadas o cuántas quedan sin identidad resuelta. También puede ser interesante poder extraer valor del material sin identificar a nadie: contar personas, medir aforo o generar mapas de calor de movimiento a partir del tracking, sin necesidad de vincular esos datos a una identidad.

Además, el usuario final no es solo quien maneja la herramienta sino también la persona grabada. Así pues se debe señalizar el entorno debidamente advirtiendo de que el espacio está siendo procesado, información accesible sobre qué se hace con las imágenes, y un canal para poder pedir la supresión de la identidad del individuo, lo que implica que el sistema debe poder borrar una identidad concreta de la galería.

Un caso que conviene contemplar es el de una autoridad que necesite recuperar la identidad de una persona anonimizada, por ejemplo por una orden judicial. Aquí el sistema no debería permitirlo por diseño, porque si la anonimización es reversible entonces no es anonimización sino pseudonimización. Si el cliente necesita esa capacidad sería un producto diferente, que debería conservar el original bajo custodia y acceso auditado, y anonimizar la copia que se distribuye.

Y por último la integrabilidad, que probablemente es lo que más pesa para el cliente. Para muchos usuarios la mejor UX es no tener que cambiar de herramienta, así que el anonimizador debería ser un elemento más del pipeline multimedia que ya usan, por ejemplo como plugin de GStreamer, en lugar de una aplicación aparte que obligue a exportar, procesar e importar. Eso implica soportar los formatos y fuentes habituales (ficheros, streams en vivo, cámaras) y poder ejecutarse sin interfaz gráfica para integrarlo en procesos automáticos. La fricción de integración suele pesar más en la decisión de adoptar una herramienta que unos puntos de diferencia de precisión entre modelos.

7. **(EXTRA) Provide a code snippet (practical example) on how would you implement any of these applications: reidentification or depth estimation, for face anonymization.**

Esbozo corto del flujo para el filtrado por profundidad (respuesta 2), sin entrar en la implementación:

``` py
load_depth_model()                                  # Depth Anything V2-Small
load_face_detector()                                # YOLO-face / SCRFD
disparity = None                                    # se calcula cada N frames y se reutiliza

para cada frame:
  faces = detect_faces(frame)                       # siempre cada frame

  si es el primer frame o cada N frames:
    small = downsample(frame)
    disparity = estimate_depth(small)               # valor alto = cerca (sin escala métrica)
    disparity = upsample_guided(disparity, frame)   # bordes limpios usando el frame como guía
    disparity = temporal_smooth(disparity)          # evita flicker entre inferencias

  bg_mask = disparity < threshold                   # el fondo tiene disparidad baja
                                                    # threshold por percentil, no en metros
  deny_mask, allow_mask = build_face_masks(faces)

  blur_mask = (deny_mask OR bg_mask) AND NOT allow_mask
  frame = apply_blur(frame, blur_mask)
```

<p align="center">
    <img src="../imagery/sad-crying-bald-black-guy.gif"
</p>