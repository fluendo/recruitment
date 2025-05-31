# Deep learning analytical thinking

## **Question 1:**
**What do you think it occurred during this model development (trainning & evaluation)?**

To understand what may have gone wrong during the development of the model, it is important to first analyze the nature of the problem. For this, we can look at real-world cases where the system has failed, as described in the provided news articles. These reports highlight several issues:

- Blurry or low-quality images

- Misreading of visually similar characters

- Errors when processing new or unfamiliar license plates

These errors suggest three main underlying problems in the development and training phase.

First, if the model was trained primarily on high-quality, well-lit images, it is likely to perform poorly when faced with blurry, dark or partially visible plates. This lack of robustness may stem from insufficient training data representing real conditions. The model likely lacks exposure to diverse image qualities and realistic noise, which hinders its generalization ability.

Second, the misclassification of similar-looking characters (e.g., confusing "C" with "G" or "D" with "O") indicates that the model has not effectively learned to distinguish subtle visual differences. This could be due to a limitation in the dataset: if not enough difficult or borderline examples were included during training, the model may struggle in these scenarios.

Finally, if the model has mainly seen older or frequent license plate combinations, it may fail to correctly identify new or rare ones. This suggests a case of overfitting to the training data. Moreover, if the dataset was biased toward certain regions, vehicle types, or plate formats, the model might struggle when encountering inputs outside those patterns.


## **Question 2:**
**How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks**

First, synthetic versions of the images could be generated to simulate more challenging real-world conditions, such as blurry, poorly lit or partially visible license plates, as well as new character combinations or different plate styles. This approach can help improve the model’s robustness under adverse conditions and allows it to learn rare cases not well represented in real data. However, if the synthetic data is not generated properly or does not accurately reflect real scenarios, it may introduce irrelevant noise. Additionally, since this approach does not modify the model architecture, there may still be limitations in its ability to generalize.

Another option would be to incorporate a system that calculates confidence scores for each recognized character and applies additional rules when predictions have low reliability. For example, if the model detects a low-confidence prediction, it could trigger an extra verification step or reject the result entirely. This can help prevent serious errors and reduce false positives without changing the base model. However, this method requires additional validation logic, which can increase system complexity and response time, especially in real-time applications.

## **Question 3:**
**What do you think it will occur when running this AI in a different country with different plates formats? How would you ensure system accuracy?**

When deploying this AI system in a different country with different license plate formats, its performance is likely to drop significantly. This is because the model was trained on a dataset with a specific plate style and when exposed to unfamiliar formats or visual patterns, it may fail to generalize properly.

To ensure the system remains accurate when applied to new formats, I would perform fine-tuning of the model using data from the target country. Since the underlying task remains the same, the model’s learned features can still be valuable. By adapting the existing model rather than training one from scratch, we can leverage its pre-learned knowledge while reducing training time and resource consumption. Fine-tuning on a smaller, representative dataset from the new domain allows the model to adjust to the specific characteristics of local plates, such as font, color, layout and structure.

This approach balances efficiency and performance, ensuring the system adapts effectively to new environments.

## **Question 4:**
**Do you know any OCR (Optical Character Recognition) algorithms (Deep learning based) that could be used here?**

To properly apply an OCR system in the context of Automatic Number Plate Recognition (ANPR), it is first necessary to detect the location of the license plate within the image. This can be achieved using object detection algorithms such as Faster R-CNN (two-stage) or YOLO and SSD (one-stage) models.

Once the license plate’s bounding box is identified, an OCR-specific model like a CRNN (Convolutional Recurrent Neural Network) can be used to recognize the characters. This model combines a convolutional network to extract visual features from the image with a recurrent layer that processes the character sequence. A final decoding layer then converts this sequence into readable text.

CRNNs are particularly effective for recognizing unsegmented text sequences, such as license plates, and allow quick adaptability to new formats.

## **Question 5:**
**Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)**

Outside the academic environment, I have not yet participated in professional projects related to Computer Vision or Artificial Intelligence. However, throughout my studies, I have completed several courses focused on image and video processing, where I developed a wide range of practical assignments to apply the theoretical knowledge acquired.

These experiences allowed me to work with various types of neural networks, including Fully Connected networks, CNNs, RNNs and their variants, as well as more advanced architectures like Generative Adversarial Networks (GANs).

In the specific area of computer vision, I have worked with object detection and segmentation models such as YOLO, SSD, Faster R-CNN, and Mask R-CNN, as well as traditional techniques like Watershed segmentation or morphological reconstructions using erosion and dilation. I have also applied methods such as frequency analysis, geometric processing, region-based modeling, and object-based modeling.

Throughout these projects, I analyzed realistic scenarios, selected and adapted the most appropriate techniques, and evaluated model performance based on the task requirements. For instance, in one of the assignments, I used transfer learning to adapt Mask R-CNN for detecting people in a custom image dataset.

These academic projects have provided me with a strong foundation not only in implementing state-of-the-art models, but also in understanding, adapting, and assessing them under different conditions and objectives.