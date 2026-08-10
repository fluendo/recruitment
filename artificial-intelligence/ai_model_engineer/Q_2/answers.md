

## Question 2.1: 
The model shows strong performance on validation data but fails on new matches recorded in different clubs with different camera heights and 
angles. Hit detection accuracy drops significantly.

Explain the most likely root causes of this failure and describe how
 you would systematically investigate and isolate the problem.
 
 
### Answer

This question actually reminds me of the decade old question with computer vision, “Why does my computer vision model fail in real/production environments?”. I have heard this far too often. 

The biggest reason for this is data drift - the dataset used to train and validate the model is different from the data used to infer the predictions. Lets explore it step by step on the best ways to identify and pin point the problem. 

- Camera angles
    - Because a change in camera angle will cause the player movement to appear differently - even though the movement has not changed completely.
    - If we explore how pose estimation works (https://www.superannotate.com/blog/human-pose-estimation-with-deep-learning), 
	it identifies key-points which are basically coordinates for lets say shoulders, wrists, head etc. So the model has learned to identify wrists coordinates at a specific camera angle for a forehand - now with a different camera angle, the coordinates of the wrist can change drastically and hence it performs poorly.
    - One way to confirm this is to do the following:
        - For each shot category, extract or focus wrist/elbow coordinates for validation and then the same for new data (different clubs/camera)
        - then for same shot compare how the wrist/elbow coordinates are through the frames
        - if there is a visible difference in the two camera angles on how the model is assigning wrist/elbow coordinates then we can confirm that camera angle is a culprit.
        - I just figured we cant compare raw or average coordinates for an elbow b/w two camera angles, because players can be standing in totally different positions so pixel coordinates will not help.
        - So rather than pixels, we look at raw frames and how the model is assigning the coordinates and how confident it is in assigning.
        - It is likely that for a new camera angle, the model will be less confident in assigning elbow coordinates.
- Exploring Particular Keypoints accuracy
    - Linking it to the idea that in computer vision, either image classification or object detection, the model sometimes performs not very well on some categories.
    - Similarly, in production, we explore what keypoints is the model missing the most. Is it elbows?
    - This would allow us to focus on specific category keypoints that can narrow down the problem. This could be due to occlusion, shadows, bad lighting, different colors of shirts/skins, etc.
- I also researched about how people are fixing pose detection and the most obvious failure is occlusion - the hand movement is occluded by the shirt or torso or even by a players shooting stance 
(as there are alot of variations in this as well)



## Question 2.2: 
After reviewing some samples, you discover that different annotators 
labeled keypoints slightly differently (especially wrists and elbows), 
and hit timestamps are sometimes shifted by a few frames.

Analyze how inconsistent annotations affect both pose estimation and
 hit classification, how you would detect label-quality problems at 
scale and how you would decide whether to relabel, filter, or reweight 
parts of the dataset. Justify your decision criteria.

### Answer

This question is a classic example of how of error propagation through badly labeled labeled datasets affect model performance. 

The two problems and their impact mentioned in the question: 

- Annotators put wrists/elbows in slightly different positions for the same scene, leads to less concise and noisy keypoints.
    - and this error then propagates through the model and leads to incorrect hit classifications.
- Timestamp differences where once annotator says hit is at A seconds and another annotator says that hit is at A + 5 seconds.
    - the classifier is being taught different answers for the same hit event.
    - hence more errors

The way these two problems affect the pose estimation and hit classification are as follows: 

1. Pose Estimation: 
    1. The inconsistent labels directly affect how the target (keypoints) are learned by the model. (the coordinates for keypoints are targets that are learned during the training of pose estimation)
    2. the model when given the similar scene with two conflicting targets, calculates the loss to move the weights in favor of one target and then tried to do the same for the other target - but in total the weights are updated as a compromise between the two. 
    3. If there is no inconsistency, the model will get the loss and update the weights in the same direction (and magnitude) - as both targets are similar for the similar scene - hence better learning. 
    4. Another thing to add is that is the annotation is off, by lets say 5pixels to the left, then the model will learn to identify the wrists position slightly to the left - leading to a learned bias. 
2. Hit Timestamp: 
    1. I believe this will not directly affect the pose estimation part. But rather if the timestamps are lets say shifted by 3seconds, the pose coordinates at that timestamp might not be a HIT. 
    2. so the model learns to identify coordinates (correct or wrong keypoints) as a hit even when it was a few seconds after a hit. 
    3. so if the hit timestamp is shifted by 3 frames, the player has already made a hit and is now in relax mode (standing) and then those relax mode coordinates are associated with a HIT - which introduces a mistake and the model learns to classify even standing positions as a HIT. 

so overall:

- **Keypoint errors** → wrong wrist/elbow coordinates → wrong movement information → can cause wrong hit classification.
- **Timestamp errors** → wrong frames labeled as hits → model learns the wrong timing → can miss or shift hits.

To answer the second part of the question which is how to detect it at scale: 

- I remember labeling ID cards for an OCR problem and the first thing we found was something called Annotator agreement (found through an online tool we planned to use). It states that give the same ID card or image to the 3 annotators and check how close the annotated coordinates are - and also identify the outliers (we did found that some of the annotators were actual outliers as there coordinates were a little off from the rest).
    - so we would either teach the outliers to correct mistakes, or relabel the ID cards that were done by those outliers. (since it was a small dataset).
- Another thing that I am a strong proponent of is trying to find patterns in errors.
    - Run the model through a well curated test set and check the top errors for each category/class. It is highly likely that if the model is making mistakes on a particular class, there is something wrong with the labels/data.
    - This can be done at scale.
- Using CleanLabs approach to detect bad labels
    - its a tool that trains a model with labeled data and then runs a kfold crossval on a test set and calculates the label score for each label.
    - It does it somehow by how consistent the errors were on labels.

To answer the third part of the question is deciding either to re-label, filter or re-weight parts of the dataset:

- Re-label: only if the class we want is absolutely necessary for prediction for example re-labelling wrists and elbows perfectly since it is important in predicting the correct pose for a hit.
    - combine it with bad label scoring (cleanlabs) - so we only label instances where the labeling done is highly inaccurate. (lets say 10k out of 100k)
- Re-weigh: I think this is very helpful when combined with label scoring. So instead of relabeling low-score labels, we can just weigh them to be less relevant. This is particularly helpful is scale.
    - Lets say we have 100k samples, re-weighing them is easier.
- Filter: filter the samples where the label scoring is very low.


## Question 2.3: 
Smashes are rare in the dataset but very important for the product. The 
model often misses them while performing well on common hits.

Without changing the model architecture, explain which dataset, 
training, and evaluation strategy changes you would consider. Here, the 
idea is to discuss tradeoffs and risks of each approach and how you 
would verify that improvements are real and not overfitting artifacts.


### Answer

I cannot recall how many times I have encountered this exact problem where the rare class is important but is not performing very well. Since models, particularly deep learning based models are data hungry, they tend to perform better on categories that have more examples. 

Since the question asks to divide it into two main strategies, dataset and training, I will divide the techniques that I think work best into these two categories. 

Dataset Strategy: 

- Collect more data (obvious - but not the most feasible)
- Weighted oversampling
    - I have tried the usual oversampling so many times and have seen it fail (SMOTE) many times as well.
    - So instead of oversampling, we can use weighted oversampling so as to not oversample only a few instances but rather the most rare instances.
        - Lets say we have 100 smashes, 80 are in similar conditions or from a single player, where 20 are in different conditions and different players, so we would want to weigh these 20 more than the 80.
        - In the usual oversampling, we would get way more samples for the 80 very similar cases - which is not the best way since we risk overfitting.
- Augmentation - being careful that it does not occlude the smash. (needs to apply transformation to the coordinates as well)
    - Horizontal flipping
    - scaling a bit
    - rotating a bit

Training Strategy: 

- Class weights
    - assigning high class weights to the smash class will make it incur a higher loss if it gets a smash wrong.
    - Although we dont want to overdo it - a very high weight will decrease the precision for smash, making it predict more smash as false positives.
- Hard Negative Mining
    - Identify the instances where the model is confusing smashes and other categories and re-train the model on these.
    - The model then learns to get better on the rare class (again not to overdo it by risking it to overfit)
- Represented/balance batches
    - whilst training we make sure that each batch/mini-batch contains the rare smash class. It makes sure that the model sees smashes at each batch instead of not encountering smashes for multiple batches.
    - best done in a stratified way. (again risk of overfitting)
- Re-training
    - This is something I learned at Fast.ai (Jeremy howard) where we train a model on the original dataset (natural distribution of classes - smashes are rare) and then re-train it on a balanced dataset (smashes oversampled or other classes downsampled)
    - So in the first turn, the model learns all the classes (although the smash class is learned not very well) and then in the second turn it gets better at smashes.
    - Its like rebalancing the classifier.
    - Although this poses a risk of overfitting (retraining is prone to overfitting), effective regularization techniques can help. (dropout, weight decay)

Evaluation:

- I think the most important 2 question when doing any kind of sampling and re-training is to ask:
    1. Have we improved the rare class? 
    2. Have we improved the rare class without damaging the other classes? 
- To answer these questions, the best way is to analyze the metrics for all the classes separately.
    - Since smashes were being missed in the question, I would focus on recall - telling us how many smashes were captured. (athough a good precision-recall tradeoff is important)
- Looking at the confusion matrix to get an idea of how many smashes were correct and how many are being confused as other classes (forehand, volley)
    - This will also give us a good idea on what the model is confusing the most with a smash - so we can focus on that in hard negatives or a new dataset.
- For the test set:
    - keep the real world distribution - if smashes are rare, we keep it rare in test.
    - Keep the test set fully unseen (not even some frames from a match that was present in training set should be in the test set)
- Another thing that comes to mind is thresholds - I have experimented with this alot.
    - We can try different thresholds for which we predict smash.
    - depends on how much we are interested in smashes, we can have a lower threshold that will make sure to capture all smashes (but might add some false positives as well)



## Question 2.4: 
Two training runs with the same configuration produce noticeably 
different results. One model is stable but less accurate; another is 
more accurate but produces jittery keypoints and inconsistent hit 
timing.

Provide a reasoning to decide which model is actually better for 
production. What additional tests, metrics, and scenario-based 
evaluations would you design to support the decision?


### Answer

The two models provided in the question:

1. Stable but less accurate (less accuracy in pose + hit classification)
2. More accurate but jittery and inconsistent hit timing. 

If I have to rank my priorities with respect to the task at hand - pose estimation and hit classification: 

- Hit timing - As a user, I would prefer correct timing.
    - I mean it would be a bummer if it correctly classifies as Smash but 8 frames/seconds later. Giving the impression that its slow.
- No jitter - stable keypoints.
    - This matters because if the keypoints are jittering on the wrists and elbows, it ruins the experience.
    - This jittery and unstable keypoints can be carried forward to the final HIT classification as well. (as we discussed Q2)
- Based on the above two priority reasons, I would choose model A (stable but less accurate)

With respect to tests to support my decision: 

- The first thing to compare would be my first reasoning - better hit timing.
    - Take model A and calculate its average hit timing miss and similarly for model B as well.
    - I would want model A to be fewer frames apart from the actual than model B. So model A could be 1-2 frames apart from the actual Hit, whereas model B would be 7-8 frames apart.
    - This timing comparison will give a definite picture on which model gets the timing right for a hit.
- Another thing I have previously used is to test difficult scenarios.
    - I would also want to test each models on its competitors errors.
    - So I take the top errors (samples) from model B and test those samples on model A.
        - also dividing the top errors in categories like - hit timing error, keypoints errors.
        - given the priority is Hit timing - if model A performs better on those hit timing errors from model B then we can say A is good.
		
		
		

## Question 2.5: 
Which model would you propose for this application? Why?

### Answer

I was searching through models for pose detection and keypoint identification and found a few discussions where people suggested RTMpose and RTMO - both pose detectors that are sports oriented - I believe they have been fine tuned on sports data and designed to be used for fast sports tracking plus its light weight and fast. 

Another good thing about RTMO is one stage - which means it can detect a person and then also get the keypoints and easier to debug in a pipeline. (no need for YOLO)

For the classifier, I found people using RTMO with a Graph CNN (https://github.com/yysijie/st-gcn). 

- So the RTMO outputs the keypoints for each frame and then through a sliding window operation, we pass it to the graph cnn which learns to associate the joints in the body and their movements as it is trained.
- So its particularly well at processing poses like smashes, volley that change over frames due to its graph structure.

So overall: 

RTMO → keypoints → 15–20 frame sliding window → Graph CNN → hit type and timing.