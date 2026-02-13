# Fluendo AI Intern Application

**Question 1:** *What do you think occurred during this model development (training & evaluation)?*

**Answer:** I believe the model suffered from **overfitting and spurious correlations**. During training, the deep learning model likely memorized patterns specific to the UK license plate images it saw (such as colors, fonts, or backgrounds) rather than learning the true character shapes. As a result, evaluation metrics on the training-like data looked good, but the model did not generalize to slight variations in the plates. The examples (one plate misread by a single character, another by a similar-looking plate) suggest that the model wasn’t robustly recognizing each character. Instead, it may have relied on background cues or a narrow set of training examples. In short, the model learned dataset-specific features and failed to generalize when a plate differed even slightly, causing misclassification during evaluation. I would also suspect **data leakage or bias** – perhaps the training and test splits were not fully independent, giving an overly optimistic evaluation until the model was faced with truly new cases.

**Question 2:** *How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks.*

**Answer:** A few possible solutions come to mind:

* **Improve and diversify the training data:**  One approach is to collect or synthesize a much larger and more varied dataset of license plate images. For example, include plates under different lighting, angles, backgrounds, and minor variations of each character. Data augmentation (rotations, brightness changes, adding noise) can also help. *Pros:* A richer dataset would force the model to learn the actual character shapes rather than spurious cues, improving generalization. *Cons:* Gathering and labeling more data can be time-consuming and costly. It may also require simulating or acquiring images from other regions to cover all variants.

* **Modify the model/pipeline architecture:**  Another approach is to use a specialized OCR pipeline. For example, segment the plate into individual characters and classify each character separately (a CNN or CNN+LSTM with CTC loss), rather than treating the whole plate as a single image. Alternatively, use a hybrid approach combining deep learning with rule-based checks (for example verify output against expected plate formats). *Pros:* A character-level model or multi-stage pipeline can be more interpretable and might reduce errors on single digits. Rule-checking can catch obvious mistakes (like if the result isn’t a valid plate number). *Cons:* These changes add complexity. Segmentation can be error-prone, and handcrafted rules may fail on edge cases. It also requires more engineering and may slow down processing.

* **Use ensemble or fallback methods:**  A third option is to combine models or add an error-detection step. For example, if the deep model’s confidence is low or the output doesn’t match a valid pattern, fall back to a classical OCR engine (like Tesseract). *Pros:* This can catch the model’s mistakes by leveraging a second opinion. *Cons:* It complicates the system and still doesn’t fundamentally solve the model’s generalization; it just mitigates errors. 

* **Flagging as inconclusive:** Have an option to let the AI mark some results as *"inconclusive"*. In some applications, like this one, it's more important to avoid false positives than false negatives. Adding the option for the AI to flag results for review when its confidence is low partially solves this problem. However, this requires manual review of flagged data or creating a larger pipeline with another "error-deciding AI". *Pros:* Reduces the risk of false positives, ensuring higher precision in critical applications. Allows for human oversight in ambiguous cases. *Cons:* Increases the complexity of the pipeline and may slow down processing due to manual review. Requires additional resources for handling flagged cases.


**Question 3:** *What do you think will occur when running this AI in a different country with different plate formats? How would you ensure system accuracy?*

**Answer:** A model trained only on UK plates will almost certainly **fail on plates from other countries**. Different countries have different plate layouts, fonts, character sets, and colors. For example, continental European plates often have blue strips or different character spacing. The current model would likely misinterpret those, leading to many errors.

To ensure accuracy across countries, I would adapt the system in one or more of the following ways:

* **Retrain or fine-tune on new data:** Collect representative samples (or use synthetic generators) of license plates from the target country and retrain the model. This could involve transfer learning: starting from the UK-trained model and fine-tuning it on images from the new format. This helps the model learn the new style and character set.

* **Modular design:** Use a two-stage approach that first identifies the country or plate type and then applies a specialized recognizer. For example, one network could classify the plate region and country flag, then a country-specific OCR model reads the characters. This ensures each model is tuned to one format.

* **General OCR architecture:** Alternatively, use a more flexible deep OCR model (see Q4) that can handle variable input. For instance, a robust text-recognition network (like a CNN+RNN with CTC or a transformer-based OCR) can be trained on many languages and plate styles simultaneously, learning the generic task of reading text from images. Adding diverse data from multiple countries during training helps the model generalize.

* **Rule and format based checks:** Finally, I would implement format validation (for example using regex or known plate patterns) for each country. Even if the model misreads a character, a validation step could catch unlikely sequences. Ensuring accuracy is an iterative process: collect feedback from misreads in the new environment, update the model, and continuously validate against known standards. Applying the Levenshtein-distance in this case can be useful. As most codes have a pattern to follow. For example a DNI in Spain can be corrected if there's one character wrong just using RegEx and the Levenshtein-distance. I do not know if this is the case in UK plates but at least just knowing the pattern like, digit, character, "-", etc. And using RegEx can be good to check a result.


**Question 4:** *Do you know any OCR (Optical Character Recognition) algorithms (Deep learning based) that could be used here?*

**Answer:** Yes, there are several deep-learning OCR approaches that could be applied:

* **Convolutional Recurrent Neural Network (CRNN):** This is a common architecture that combines a CNN feature extractor with a recurrent layer (like an LSTM) and a Connectionist Temporal Classification (CTC) loss. CRNNs can recognize sequences of characters in an image without pre-segmentation. They’ve been widely used for scene text and license plates.

* **Transformer-based models (for example TrOCR):** Newer OCR models use transformers. For example, the TrOCR model (from Microsoft) uses a Vision Transformer as an image encoder and a Transformer-based text decoder for sequence prediction. TrOCR has achieved high accuracy on OCR tasks by leveraging pre-trained transformer architectures.

* **Segment-and-recognize pipelines:** Some systems first detect each character using object detection (for example a YOLOv5 or EAST detector) and then classify each character with a CNN. This isn’t a single end-to-end model but can be very effective in practice, especially for fixed-format text like plates.

* **Commercial/Pretrained Engines:** There are also robust OCR libraries and services that incorporate deep learning under the hood (for example, Google’s Vision API, or open-source projects like PaddleOCR and EasyOCR) which could be used out-of-the-box or fine-tuned.

**Question 5:** *Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)*

**Answer:** In my previous internship at Serimag, I worked on a document analysis project for a bank. The goal was to automatically **classify and extract information from scanned documents** (like checks, forms, invoices). My responsibilities included designing the machine learning pipeline and data processing.

Specifically, I used **SVM classifiers** with **spatial pyramid descriptors** and **Latent Semantic Indexing (LSI)** to categorize document images into types (for example contract, ID, letter). We combined visual features (to capture layout textures) with text features (from OCR output) using LSI to reduce dimensionality. One major challenge was that OCR results were often noisy: scanned text had recognition errors. I addressed this by writing custom **regex-based correction scripts** to fix common OCR mistakes (for example, correcting “0” vs “O” or “5” vs “S” in context).

Another difficulty was varying document layouts. We needed to locate key fields (like a name or number) in documents with different templates. To handle this, I applied **Hough transforms** to detect lines and align segments of the page. For instance, Hough helped find the bounding lines of table columns so we could isolate data fields. This significantly improved our ability to correctly parse information regardless of document orientation or skew.

Overall, I learned to integrate computer vision, OCR text processing, and classic machine learning into a practical solution, and I enjoy applying those skills to new AI challenges.
