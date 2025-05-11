# Answers #1

## Deep Learning Analytical Thinking

### Answers:

#### 1. **What do you think occurred during this model development (training & evaluation)?**

The problem describes a typical computer vision task, where the mission is to detect the license plate of a vehicle and recognize the characters on it.
A misclassification in such a case could be due to several reasons:

* **Lack of training data**: Deep learning models are data-driven. In a typical DL task, a large quantity and diversity of data is essential to unlock their potential and achieve precise results. The dataset should cover various scenarios to generalize well.

* **Low-quality data**: Like quantity, quality matters. The dataset should be balanced so the model performs ideally across all character classes. Incorrect labels can also misguide the learning process.

* **Model architecture**: As demonstrated in the ResNet paper, the architecture—such as the number of layers and the activation functions—has a significant impact on performance. A poorly chosen architecture can lead to suboptimal results.

* **Incorrect evaluation metrics**: In cases of imbalanced data, traditional metrics (accuracy, precision, etc.) may be misleading since each class contributes equally. More suitable metrics should be used.

#### 2. **How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks**

* **Data Augmentation**: By rotating, cropping, or altering images, we can synthetically increase dataset size.

  * *Pros*: Easy to implement and can boost performance.
  * *Cons*: Augmented data might not always reflect real-world scenarios, so quality isn't guaranteed.

* **Pre-trained Models**: Use a model trained on large datasets and fine-tune it with transfer learning.

  * *Pros*: Usually achieves better results, especially when the original dataset is small.
  * *Cons*: Requires more computational resources.

* **Tuning Model Architecture/Parameters**: Experiment with different model designs and hyperparameters.

  * *Pros*: Can significantly improve results.
  * *Cons*: Requires benchmarking and hardware capabilities.

* **Traditional CV Methods**: Instead of predicting all characters at once, segment the license plate region into individual characters and recognize them one by one.

  * *Pros*: Can improve accuracy.
  * *Cons*: Might slow down inference.

* **Macro/Micro-Averaged Metrics**: Use metrics that account for class imbalance.

  * *Pros*: Offers a more realistic performance measure.
  * *Cons*: Adds complexity to evaluation but not training.

#### 3. **What do you think will occur when running this AI in a different country with different plate formats? How would you ensure system accuracy?**

If the training dataset includes only local plates (e.g., UK), the model may not generalize well to plates from other countries due to different formats and styles. The system may fail because it's overfitted to the training distribution.
**Solutions**:

* Retrain or fine-tune the model using local data from the new country.
* Employ a region-specific model or use transfer learning to adapt the base model to new plate formats.


#### 4. **Do you know any OCR (Optical Character Recognition) algorithms (deep learning based) that could be used here?**

I haven’t studied deep learning-based OCR models in depth. However, I understand OCR as a classification task where the input is an image of a character and the output is the corresponding label.
I also see parallels with voice recognition, which also involves sequential or structured data. In that context, I studied architectures like CNNs, RNNs, and Transformers—relevant ideas for OCR too.

#### 5. **Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)**

Besides the projects listed in my CV, here are some academic projects I worked on:

* **Vegetal Condition Research:**
This was an individual project completed for the final assignment in a Computer Vision course. The dataset consisted of images of the same type of plant, and the goal was to analyze visual attributes such as color and area to assess their condition. I used traditional computer vision techniques like segmentation, morphological operations (open/close), and color histograms, mainly with MATLAB’s Computer Vision Toolbox. One of the main difficulties was segmenting the plants accurately, as lighting conditions and backgrounds varied across images. I solved this by experimenting with different thresholding strategies and applying open/close operators to refine the segmented regions and recover essential plant details.

* **Character Recognition of Book Titles:**
In this project, the dataset contained images of book covers along with separate images for each of the 26 alphabetic characters. The objective was to recognize the title text on each book cover. My approach began with image segmentation to isolate each character as an independent connected component. Then, I applied skeletonization and extracted features such as chain code, number of lines, and number of circles, which were then compared to the reference dataset. This was also part of the final Computer Vision project. The main challenge was identifying efficient and meaningful features to describe the characters, which I addressed through iterative testing and refinement.

* **Transfer Learning with YOLOv8:**
This project followed the Transfer Learning workflow described in the GitHub repository Pretrained YOLOv8 Network for Object Detection. I used the Indoor Object Detection Dataset to fine-tune the model. Even though the dataset was relatively small, training the network for just 10 epochs yielded promising results, with a mean Average Precision (mAP) close to 0.8. This experience showed me how effective transfer learning can be in scenarios where data is limited, and how pretrained models can drastically reduce training time while maintaining high performance.
