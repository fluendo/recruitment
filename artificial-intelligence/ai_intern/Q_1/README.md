# Answers

## 1. What do you think occurred during this model development (training & evaluation)?
  A number of things could explain the errors of the ANPR system. Maybe the loss function wasn't properly defined so that it avoided false positives. Maybe the model works quite well, but even though it gets an accuracy of 96%, 4% of false positives result in several incorrect penalty charges. The model should be trained with more focus on avoiding FP with metrics like precision or F1-score.

  Assuming the loss function was adequately defined, there could still be other issues with the training itself. Furthermore, there could be data leakeage and overfitting, given that "vehicles of a similar make and color but with a slightly different registration number" were confused. This makes me think that the model isn't just focusing on the license plates, even though it should since license plates are unique and don't give any information about the vehicle, or the other way around. If it's using the vehicles themselves to make predictions, that's a sign that it's trying to learn patterns that won't/shouldn't exist.

  Besides that, there's the example of the "plates differing by a single character", which could be similar letters (C-G, V-Y, D-O, ...), or numbers and letters (0-O, 1-I, 2-Z, ...) getting confused. While this is normal, it could be diminished by augmenting commonly confused characters in the training dataset, and specifically training to differentiate them among varying conditions and fonts.

  Last but not least, the issue could not be with the model itself, but with the data. As we all know, "garbage in, garbage out". For example, the real world data could be very different from the training dataset, with different angles and lighting conditions. Or maybe the images taken by the cameras were blurry, as suggested by the second article. If the data is too noisy or blurry, there's no model in the world that can fix that.

## 2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks
  ### Solution to False Positives
  The solution to a hypothetical overrepresentation of false positives in the model could be to use a different loss function that penalizes false positives more than false negatives. For example, using a weighted cross-entropy loss function where the weight for false positives is higher than for false negatives, or a simple precision or F1-score loss function.
  #### Pros:
  - We would probably get a model that has a better precision and F1-score, which would reduce the number of false positives.
  #### Cons:
  - This could lead to an increase in false negatives, which could result in some vehicles not being detected at all.

  ### Solution to Using Vehicle Features
  To make sure the model focuses on the license plates and not the vehicles in the training dataset, which could be getting confused by similar vehicles on the street, we could use a more robust data preprocessing step that isolates the license plate area from the vehicle image. To do this, we could use a bounding box around the license plate area. Pre-trained models like YOLO or Faster R-CNN could be used to detect the license plate area in the image, and then crop the image to only include that area. This would ensure that the model is only trained on the license plate area and not on the vehicle features.
  #### Pros:
  - We would ignore the vehicle features and focus on the license plate area, potentially improving the model's accuracy and training time.
  - We would reduce the risk of vehicles in the training dataset being confused with similar vehicles on the street.
  #### Cons:
  - This would require additional preprocessing steps, which could increase the training time and complexity of the model.
  - If the bounding box is not accurate, it could lead to the model not being able to detect the license plate area, and potentially missing some vehicles altogether.

  ### Solution to Commonly Confused Characters
  To address the issue of commonly confused characters, we could augment the training dataset with images of license plates that contain these characters. This could be done by generating synthetic images of license plates with these characters, or by collecting real-world images of license plates that contain these characters. We could also use a character recognition model to specifically train the model to differentiate between these characters.
  #### Pros:
  - This would help the model to learn to differentiate between commonly confused characters, potentially improving the model's accuracy and reducing the number of false positives.
  #### Cons:
  - This could require additional data collection and preprocessing steps, which could increase the training time and complexity of the model.
  - If not carefully done, it could lead to overfitting on these characters, which could result in the model not being able to generalize to other characters, or seeing them more commonly than more distinct characters.

  ### Solution to Noisy or Blurry Data
  To address the issue of noisy or blurry data, we could use data augmentation techniques to improve the quality of the training dataset. This could include techniques like adding noise to the images, blurring the images, or changing the lighting conditions. On the more practical side, we could use better cameras with faster shutter speeds and better lenses to capture clearer images of the license plates.
  #### Pros:
  - This would help the model to learn to recognize license plates in different conditions, potentially improving the model's accuracy and robustness.
  - It could also help to reduce the risk of overfitting on specific conditions, which could lead to better generalization.
  #### Cons:
  - This could require additional data collection and preprocessing steps, which could increase the training time and complexity of the model.
  - Installing better cameras could be very expensive.

## 3. What do you think it will occur when running this AI in a different country with different plates formats? How would you ensure system accuracy?
  Assuming the license plates in the UK are relatively homogeneous in terms of format, running the AI in a different country with different plate formats could lead to significant issues. The model could have learned to assume the first two characters are letters, the next two are numbers, and the last three are letters, which is the format of UK plates. If it was applied in Catalonia, for example, it might confuse the first two numbers with letters, or the last three letters with numbers, leading to a lot of false positives and negatives.

  To try to prevent this, I could preprocess the dataset with masks that hide the full plate so it learns to focus on the characters themselves. Other similar techniques like dividing the plate into segments and training each segment separately or reordering the characters in the plate could also be used. However, I would have to make sure that the labels of the masks or segments are still correct and that the preprocessing isn't introducing too much noise to the training dataset.

  Other considerations could be if the country has a different alphabet, like Arabic or Chinese characters. However, while researching I found that there's a Geneva Convention (not the one you're thinking of lol) that says that all license plates should have Latin characters and Arabic numerals, so this shouldn't be an issue. Colors and shapes do differ though, so transforming the images to black and white and using a consistent shape (adding a white margin around the plate to make it consistent) could help the model to learn the characters better and not get confused by the colors or shapes of the plates.

## 4. Do you know any OCR (Optical Character Recognition) algorithms (Deep learning based) that could be used here?
  - **CRNN (Convolutional Recurrent Neural Network)**: It combines CNNs and RNNs with a CTC (Connectionist Temporal Classification) loss function, which makes it appropriate for recognizing text in images where the length of the text is variable.
  
  Since I didn't know any others, I looked up some more and found:
  - **Tesseract**: Open-source that uses LSTM networks.
  - **TrOCR (Transformer-based OCR)**: Uses transformers for text recognition, which can be more efficient than RNNs.

## 5. Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)

  I worked with a teammate to train a Deep Learning classification model to identify the materials and techniques of art pieces based on only their high-resolution images. Typically, low-resolution images are used for this task, but since we got a chance to use one of the world's most powerful supercomputers (MareNostrum V) we used over 30k (200 GB) high-resolution images. While our shallow CNN model didn't match the performance of the SOTA models like VGG11 or ResNet50 (it was also much smaller), it was still able to achieve a 70% accuracy on the test set, which was a significant improvement over the previous models that used low-resolution images. Working with such a large dataset was a challenge, because even though we had access to a supercomputer, we still had to be careful with the memory usage and training time. We had to wait for the queue to run our jobs, and we had to optimize our code to make sure it was efficient, but we couldn't do anything locally since only a very tiny subset of the dataset fit in our local machines.
