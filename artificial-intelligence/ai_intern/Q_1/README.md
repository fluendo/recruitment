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


  * **Insufficient Data (Quantity & Variety)**: There might not have been enough training data or it may not have adequately covered the variety of situations the cameras could capture. This includes variations such as rotations, blurring, and other distortions, which are crucial to reflect real-world conditions in which the cameras operate.
  * **Insufficient Hyperparameter Tuning**: There might not have been an exhaustive search for optimal hyperparameters. Factors like the number of epochs, the size of layers, activation functions, learning rate, batch size, or the application of regularization (such as dropout) might not have been properly optimized, which could have affected the model’s ability to learn effectively.
  * **Inadequate Model Choice**: It's possible that a too-simple or inappropriate model architecture was selected, such as a traditional deep neural network instead of a CNN (Convolutional Neural Network), which is much more efficient for extracting spatial features (feature maps) from images. CNNs are specifically designed to capture patterns in images, such as edges, textures, and shapes.
  * **Feature Transformation Issues**: The images may have needed appropriate preprocessing, such as normalization of pixel values, conversion to grayscale (if color wasn’t required), or downsampling the images to make the model more efficient. However, if downsampling wasn’t done correctly, key information could have been lost.
  * **Overfitting**: During training, the model might have learned too much from the training data and failed to generalize well to unseen data. It's crucial to have an adequate validation set that reflects the possible variations in real-world data.
  * **Inadequate or Misleading Metrics**: If the wrong metrics were used to evaluate the model's performance, misleading results could have been obtained. In this case, metrics such as precision and recall are crucial to avoid errors, but they need to be measured correctly.
  * **Insufficient Test Data (Variety of Scenarios)**: A small or overly homogeneous test set could have led to the model not seeing enough diverse situations. It is important that the test data covers as many scenarios as possible so the model can generalize correctly.
  * **Class Imbalance**: If the data was imbalanced (e.g., more images of cars than motorcycles), the model could have been biased towards the more frequent class, which would affect prediction accuracy.
  * **Noisy or Low-Quality Images**: The model might have been trained on low-resolution, noisy, or out-of-focus images, making it difficult for the ANPR (Automatic Number Plate Recognition) cameras to correctly identify license plates.
  * **Poor Data Labeling**: Poor labeling of images during data preparation could have led to the model learning incorrect patterns or non-representative features of the data.


2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks.

  * One of the most important solutions to address this type of issue lies in **improving the quality and quantity of the data** used during training. It is essential to have a dataset that is **large and diverse enough** for the model to learn to generalize effectively. If the dataset is small or not sufficiently representative of the different real-world conditions, a recommended option would be to apply **data augmentation**. This involves applying transformations to the images, such as rotations, changes in lighting, translations, etc., to increase data diversity. This can also help prevent **overfitting**, as the model is exposed to a wider range of image variations. **The main advantage of this option is that it can significantly improve the model's generalization ability, but the downside is that the data augmentation process can make training slower and may introduce noise if the transformations are not applied properly.**
  * Another crucial aspect is **hyperparameter tuning**. It is necessary to properly adjust hyperparameters, choosing the right values for elements such as the learning rate, number of epochs, layer size, and activation functions, among others. It is also important to ensure a proper **data split** so that both the training and validation sets cover a wide range of possible cases. **The advantage of this option is that it allows for improved model accuracy through better configuration, but the challenge is that it requires time and resources to experiment with different hyperparameter combinations.**
  * A third solution would be to choose **appropriate models**, such as Convolutional Neural Networks (CNNs), which are much more effective for image processing tasks. Additionally, **pretrained models** like ResNet or VGG can be used, as they have already been trained on large datasets. This not only reduces the computational cost of training but also improves result quality since these models have already learned general image patterns. They can be used as a base and then **transfer learning** or **fine-tuning** can be applied, meaning the model is adapted to a specific dataset by adding custom images and training the final layers. **The benefit of this option is that it reduces training time and improves results, but the drawback is that it may not be ideal for every case, as pretrained models do not always adapt perfectly to highly specific tasks.**
  *Last but not least, an additional solution is to work with **appropriate evaluation metrics** and apply techniques to **minimize overfitting**. To do this, it is important to have a good validation dataset and apply techniques such as **regularization**, specifically **dropout**, to prevent the model from overtraining on specific data. Additionally, analyzing the model’s errors through a **confusion matrix** and adjusting the model accordingly can be key to improving performance. **The benefit of this option is that it helps improve the robustness of the model, but the challenge lies in finding the right balance between generalization and specialization, which requires continuous monitoring of the model’s performance.**

3. What do you think it will occur when running this AI in a different country with different plates formats? How would you ensure system accuracy?

  * If this license plate recognition system were deployed in another country with **different plate formats**, it is highly likely that it would **fail or significantly lose accuracy**, especially if training did not include examples of those new formats or lacked a **sufficient variety of data**. While license plates across countries may differ in aspects like size, font, character arrangement, or even the use of symbols, many share **common structural patterns** (e.g., alphanumeric combinations arranged horizontally). However, if the model has not learned to **extract general and robust features** that capture these commonalities, it will likely perform poorly, something already observed with errors involving even slightly different plates or similar-looking vehicles within the same country.

  * To **ensure the system's accuracy** in a new country or international context, it's not enough to apply the model as-is. It’s necessary to:

      * **Expand the dataset** by including real-world images of license plates from the target country under various conditions (lighting, angles, image quality, etc.).
      * Apply **transfer learning** by reusing the base model and fine-tuning the final layers using local data.
      * Ensure **specific validation** using data that reflects the real deployment environment, to test whether the model generalizes correctly.
      * Use **appropriate metrics** to track errors like false positives and false negatives, especially in sensitive applications like automated fines.
      * In some cases, it may help to implement a **modular architecture**, where the system first detects the country or plate type and then applies the most suitable model for that specific format.

  * In summary, without **explicit adaptation** to the new environment, the system is very likely to make errors, not only with similar plates but even with **completely different elements**, potentially **failing to recognize the new country’s plates** altogether. For this reason, it is crucial to apply the previously mentioned recommendations related to **data quality, quantity, and diversity**, and to adopt a well-defined strategy for **model adaptation and contextual validation**.

4. Do you know any OCR (Optical Character Recognition) algorithms (Deep learning based) that could be used here?

  * Although I haven't explored the field in great depth before, I’m familiar with concepts like CRNN (Convolutional Recurrent Neural Network), which combines CNN layers to extract visual features and RNN layers (such as LSTM or GRU) to model character sequences. This approach is particularly effective for structured text like license plates. Given my solid understanding of the underlying mathematical, statistical, and computer science principles, as well as my awareness of current trends, I’m confident in my ability to quickly learn and apply other OCR tools as needed.

5. Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)

  * Regarding computer vision, I have completed several exercises during a machine learning course focused on neural networks, where I learned to perform image classification using a simple neural network (not CNN). However, on my own initiative, I practiced additional exercises to reinforce my skills with CNNs and related techniques such as data augmentation and dropout.

  * In terms of AI more broadly, I have participated in various machine learning and data mining projects at university involving classification and regression tasks. These projects ranged from tackling small problems with different models to more complex, real-world oriented projects where I selected appropriate datasets, performed necessary data transformations and explorations, and managed the full pipeline of training, evaluation, and model selection.

  * Additionally, I have worked on AI projects related to local search algorithms, expert rule-based systems, and planning using PDDL. I have some experience with constraint satisfaction problems (CSP) and I am currently learning about reinforcement learning.

 * Moreover, I have worked with local large language models and frameworks such as LangChain, mainly to test model capabilities using techniques like few-shot prompting, retrieval-augmented generation (RAG), and tools that extract data from sources like Wikipedia, databases, or scripting.


