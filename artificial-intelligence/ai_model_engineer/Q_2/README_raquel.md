# QUESTION 1:
**The model shows strong performance on validation data but fails on new matches recorded in different clubs with different camera heights and angles. Hit detection accuracy drops significantly.
<br>Explain the most likely root causes of this failure and describe how you would systematically investigate and isolate the problem.**

With pose estimation the most typical problem is distribution shift between training and validation data and inference data. The model performs well on validation because that set probably follows the same distribution as training data (same clubs, camera position, lighting, distances...) but when the model sees new clubs and new visual conditions doesn't generalize good. The model learns not where the wrist is but also where the wrist tends to appear in a particular frame, when changing the frame in inference is where the model has problems to predict correctly.
 
The main sources of this shift are: viewpoint (height, angle, zoom, distance of the player), background and lighting (the glass walls, adverts on the glass, the color of the floor) and camera quality (resolution, bllur, compression artifacts).

To solve this, I would investigate which part of the body by club has the lowest percentage of correct keypoints (PCK). This will tell me where the problem is:
- Pose estimation problem: If wrist and elbows have a low PCK but torso and hips are fine, the person is detected correctly but the keypoint regression is wrong.
- Detection problem: if torso and hip PCK is low, the pose estimator doesn't receive a good crop of the person. 

If the problem is the pose estimation, the first thing to try is a preprocessing fix that requires no retraining at all: compute a court homography from the court line markings. A padel court always has the same standardized dimensions, so we can compute a perspective transform that maps every frame to a canonical top-down view before feeding it to the model. If this recovers accuracy, the problem was purely geometric and you never needed to touch the model.

If preprocessing is not enough, I would fine-tune the model adding perspective augmentation (simulating different camera heights and angles), different backgrounds, and lighting variation. Importantly I would freeze the backbone and only retrain the regression head, because the backbone features like edges and body silhouette are probably still correct.

If the problem is detection, I would lower the confidence threshold and re-evaluate. I would consider adding a multi-scale detection if necessary to deal with small person silhouettes.

# QUESTION 2
**After reviewing some samples, you discover that different annotators labeled keypoints slightly differently (especially wrists and elbows), and hit timestamps are sometimes shifted by a few frames.
<br>Analyze how inconsistent annotations affect both pose estimation and hit classification, how you would detect label-quality problems at scale and how you would decide whether to relabel, filter, or reweight parts of the dataset. Justify your decision criteria.**

The pose estimator is trained by placing a heatmap peak at the labeled join position and penalizing any deviation from it during training. Therefore, if one keypoint is defined by two different anotators the model receives contradictory supervision signals, and will result in predicting the average of them, as it can satisfy both.

For hit classification, timestamps inconsistency breaks the temporal localization it tries to learn, as the model needs to capture when each hit happens. If the mark of the hit is at different frames it is difficult for the model to learn the temporal response, it will learn a broad window across the hit. It can be a correct hit classification but a bad timestamps, early or later than it actually is. 


# QUESTION 3
**Smashes are rare in the dataset but very important for the product. The model often misses them while performing well on common hits.
<br>Without changing the model architecture, explain which dataset, training, and evaluation strategy changes you would consider. Here, the idea is to discuss tradeoffs and risks of each approach and how you would verify that improvements are real and not overfitting artifacts.**

This is a class imbalance problem. The model has seen very few smash compared to the other common hits, so the decision it learns is biased toward the majority of hit classes. 

In the training set we need to oversample smashes. We can duplicate the exsiting ones adding a temporal agumentation to each copy, mirror the players horizontally, in order to have more samples without collecting new data. We only do that in training set. The risk of doing that is overfitting to the specific players we have in the training set. 

We can also collect videos that have similar actions of a smash but don't result to one. Such as raising the elbow or have the arm extended overhead. Label this actions as non-smash and include them in training. this helps the model to know what is not an smash. 

For the training strategies, I can use focal loss rather than manually fine tuning each class. Focal loss automatically puts a lower weight to easy classes and higher weight to more difficult ones. I like this strategy because it doesn't need manualy finetuning or hyperparameters.

After training, I can also lower the decision threshold for the smash class. This is cheap and requires no retraining, can be tuned directly on the validation set. However, the limitation is that if the model never produces high confidence for smashes this only adds false positives.

In order to evalutate correctly, the validation and test set must have realistic smash prevalence, not balance the sets artificially. Per class I would look to PR curves, look at the average precision of the smash independenly. Run inference in a club that the model has not seen doing training and checking the tracing with different camera angles and players. In order to prevent overfitting I can also perform k-fold cross validation where each fold is a different set of players.

# QUESTION 4
**Two training runs with the same configuration produce noticeably different results. One model is stable but less accurate; another is more accurate but produces jittery keypoints and inconsistent hit timing.
<br>Provide a reasoning to decide which model is actually better for production. What additional tests, metrics, and scenario-based evaluations would you design to support the decision?**

In order to decide which model is actually better I would implement these steps:
1. Keypoint trajectory smoothness metric: I would compute the acceleration of keypoints acrooss time. A good model needs low-acceleration for smooth player motion. So this is a metric I will take a look at it.
2.  Hit timing consistency test: run inference in the same video with different small perturbations (like brightness) and measure the variance in predicted hit timestamps. The stable model needs to produce identical timestamps. 
3. Difficult test examples: test both models on more difficult examples such as players overlapping, fast smashings with not 100% visibility.

# QUESTION 5
**Which model would you propose for this application? Why?
For pose estimation I would propose MoveNet from TensorFlow as it is a fast an accurate model that detects 17 keypoints of the body. The model is trained on COCO and fitness, dance and yoga poses, so closer to padel than models trained only on walking padestrians.**

For hit classification, I would try and small transformer encoder, as it can model the full temporal context of the window simultaneously, which helps distinguish hits that have similar peak positions but different lengths, sush as smash and serve. 