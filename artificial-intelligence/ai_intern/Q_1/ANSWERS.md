## 1. What do you think it occurred during this model development (trainning & evaluation)? 

The problem with similar characters being misrecognized might be due to a low number or quality of the samples in the training data.

A different issue arises with the driver that got fined with a similar car; this shows the model is getting more information into the recognition of the plates than just the plate. This behavior might be unintended and might need a review of the pipeline of the model.

## 2. How would you fix this behavior? Please provide at least 2 options explaining their pros and drawbacks
### Data enhancement and retraining

A revision of the dataset is needed, ensuring it contains a good amount of plates in different lighting conditions. Also, a different training metric should be used to reduce false positives, like precision instead of accuracy which is more common in general use cases.

### New model architecture in two stages

First, we can separate the license plate from the rest of the image, then we can pass the image of just the plate to the second stage of the model that will identify the plates more easily. Before this step, we could also apply filters to the image to try to minimize the impact of lighting conditions on the result.

## 3. What do you think it will occur when running this AI in a different country with different plates formats? How would you ensure system accuracy?

Different shapes, colors, or patterns might confuse the model, diminishing its accuracy and precision. To ensure proper recognition in the new country, a fine-tuning step is probably needed. 

## 4. Do you know any OCR (Optical Character Recognition) algorithms (Deep learning based) that could be used here?

After a bit of research, I think the best solution for this use case would be a YOLO (You Only Look Once) model for feature extraction and retrieval of the plates. Following this, we could use a CRNN (Convolutional Recurrent Neural Network) or an attention-based transformer OCR depending on the computing budget we have. 

## 5. Explain a Computer Vision / Artificial Intelligence project in which you have participated (goals, your role, difficulties you found, how they were solved, ...)

My main project in AI has been my end-of-degree project. In this project, I learned a lot of the basics in AI models, implementing a Multi-Layer Perceptron followed by a genetic algorithm to optimize a hybrid solar-gas power plant for different climate conditions. I used scikit-learn at first, then built a custom model on PyTorch to leverage GPU parallelism.

I have worked on some smaller projects training CNNs and other models on the MNIST OCR dataset.

I also participated with some colleagues in the FME datathon 2023 in which we developed a small model to classify outfits. This was a rather small project in which we all had very little experience with AI models. I led the team to clean the dataset and we brainstormed different ideas for outfit prediction. In the end, we came up with a simple neural network classifier that, given 2 objects in the dataset, predicted the most fitting other object in the sample data. Given more time, we would have liked to implement some human feedback in the loop, but we fell short on time due to our lack of experience.

In the near future, I will be doing my master's thesis on optimizations for the performance of LLMs on modern CPU architectures.