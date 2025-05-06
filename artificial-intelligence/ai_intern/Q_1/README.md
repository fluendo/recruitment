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

---

## 1. What do you think it occurred during this model development (trainning & evaluation)? 

At first, several issues might have occurred during development:

- Imbalanced dataset: The training data may not have included enough examples of all plate types, leading to poor generalization.
- Lack of real-world variability: Missing data from adverse conditions such as rain, night-time, or blurred images.
- Confusion between similar characters: Characters like 'B' and '8' or 'O' and '0' can be easily misclassified if not properly represented in the dataset.
- Bias in training data: Over-representation of certain plate styles or regions might have caused the model to learn biased patterns.
- Low plate diversity: Limited variation in fonts, plate formats, and character arrangements could reduce the model’s robustness.

---

## 2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks

Option 1: Upgrade and expand the training dataset
- Pros:
  - Improves model performance without modifying the architecture.
  - Helps the model generalize better to real-world cases like blurry, dirty, angled plates, etc.
- Cons:
  - Requires more time and computational resources for training.
  - Collecting or generating high-quality annotated data can be costly and time-consuming.

Option 2: Add a verification module
- Pros:
  - Reduces the risk of unfair fines by comparing detected plates with registered plate databases using similarity measures.
  - Can reject low-confidence results or flag them for human review.
- Cons:
  - Increases system complexity and potential latency.
  - Requires access to additional data sources, which may raise privacy or compliance concerns.

---

## 3. What do you think it will occur when running this AI in a different country with different plates formats? How would you ensure system accuracy?

If the AI system is deployed in another country, it may have some potential problems:

Potential problems:
- Different plate formats: Varying lengths, character sets, layouts, and fonts.
- Non-Latin alphabets: Countries using Arabic, Asian, or other alphabet characters may not be recognized by a model trained only on Latin-based plates.
- Regional symbols: Flags, logos, or national identifiers might confuse the model.
- Legal layout variations: Some countries have two-line plates or regional codes.

To ensure system accuracy:
- Retrain or fine-tune the model with local plate data using transfer learning.
- Collect a representative dataset from the target country with diverse conditions.
- Set up a validation system: Use confidence scores or similarity checks to reject uncertain predictions.

---

## 4. Do you know any OCR (Optical Character Recognition) algorithms (Deep learning based) that could be used here?

Yes, several modern OCR algorithms based on Deep Learning are well-suited for license plate recognition tasks:

CRNN (Convolutional Recurrent Neural Network)
- It combines CNN layers for visual feature extraction with RNN layers for sequence modeling.
- This one is widely used in text recognition tasks such as license plates, signs, and documents.

Tesseract OCR with LSTM
- This one is an open-source OCR engine developed by Google.
- The latest versions incorporate LSTM which is a type of RNN to handle sequences better.

---

## 5. Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)

Project 1: Magic: The Gathering Chatbot Assistant

- Goal: Build an intelligent assistant capable of answering questions about Magic: The Gathering cards, rules, deck building, and interactions between abilities.
- My Role: I designed and implemented the core logic of the chatbot using natural language processing (NLP) techniques. I used a public database of cards and mechanics, and integrated it with a retrieval-based dialogue system.
- Difficulties:
  - Understanding domain-specific queries that involve card combinations or informal phrasing.
  - Maintaining context across multi-turn conversations.
- How I solved them:
  - Used a combination of intent classification and entity recognition to extract card names and rule components.
  - Implemented a memory mechanism to keep conversation context and follow-up logic.


Project 2: Agricultural Land Analysis via Satellite Images

- Goal: Predict whether a specific terrain can be cultivated based on satellite image data and color information.
- My Role: I developed the image analysis pipeline, performing preprocessing, segmentation, and classification.
- Difficulties:
  - Differentiating terrain types under varying lighting or seasonal conditions.
  - Limited labeled data for training the model.
- How I solved them:
  - Applied color space transformations (HSV, NDVI) to enhance vegetation patterns.
  - Used image augmentation and clustering to expand the training set and generalize better across cases.

---
