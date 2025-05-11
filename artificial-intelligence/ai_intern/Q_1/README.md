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

During training, the first main issue could be the lack of training data or class imbalance (the dataset must be as diverse as possible so the model can differentiate all features equally). Insufficient data would make it impossible for the model to learn to extract features such as small changes in characters, variations in car models, lighting conditions, colors, or the presence of noise, among many other factors. Another possible reason could be that the model is not suitable for the data it needs to represent. This could be because it is too simple, leading to an inability to learn patterns (underfitting, where some letters are classified as the same), or it might be too complex, resulting in overfitting (learning irrelevant details such as the frequency of certain characters). Another potential issue could be in the model’s loss function, where thresholds are too low, or all errors are treated equally, without considering their frequency or relevance. Finally, another possible problem is that the validation set is not representative of real world data, it must include all the different possible groups.

2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks

Assuming the input data is well partitioned and well labeled, data augmentation could be used to address class imbalance, as well as to produce more varied images to improve discrimination. The simplest approach would be data transfer from an existing, larger dataset that includes more of the cases our model struggles with, or colecting or generating synthetic data but this can be expensive and time consuming. Another useful method would be modifying the input data to create slightly altered versions. Some transformations could include noise, blur, dropout, color changes, or more complex filters. The advantage is that more data, especially for the problematic classes, usually leads to improvements. However, this also increases computational costs, such as execution time (use of GPUs), water consumption, and environmental impact. Another solution could involve modifying the loss function to give more weight to certain classes or types of errors or implementing an adversarial training network thats looks for examples that the network missclassifies. This one probably outperforms as it can learn better where the errors are and very dificult patterns but it is mathematically more complex. Also, as a drawback, it does need a generator and a discrimination which would be another big implementation. Other approaches might involve augmenting the validation data or adding new layers and performing additional tuning.

3. What do you think it will occur when running this AI in a different country with different plates formats? How would you ensure system accuracy?

Almost certainly, accuracy would drop due to the change in data. While some features may remain the same (car model characteristics), the model would likely miss critical information from the plates due to changes in typography, color, symbols, position, and many other small but important differences. To ensure the model performs well, I would make sure the input data reflects the new license plate formats and features, as well as other conditions such as the environment in which the new camera is used and the types of vehicles that pass through it. The more real data included, the better. Finally, data transfer from our original model to the new domain would be performed, and training with fine-tuning should already improve the model.

4. Do you know any OCR (Optical Character Recognition) algorithms (Deep learning based) that could be used here?

I am familiar with CRNN, which in this example could be used as the CNN extracts features from the images and the RNN ensures those features are encoded into the correct text. Additionally, transformer based models can be implemented using attention mechanisms to improve performance.
 
6. Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)

I have participated in many student-related projects across different subjects, as well as in a few datathons and some personal projects. The latest project I’m working on is a 3-month initiative for a city hall, where I am one of nine students. We are divided into backend, frontend, and two team members who are in constant contact with the final client. I am part of the backend team and am currently developing a prediction model based on timeline data, as well as analyzing its correlation with city hall social implementations. So far, we have completed data integration, preprocessing, and selection. The first main problem was the lack of a universal format among the different input datasets, which caused inconsistencies during integration so we defined a standardized structure and applied a validation code to ensure minimal errors. Now that the data has been cleaned, we are training temporal models such as ARIMA and LSTM in order to find the best predictor. The current challenge is evaluating the models, as each has its own advantages depending on factors so we are still working on it. The overall goal of the project is to implement a visualization of the effects of different social regulations, to support better decision-making in social policy.
