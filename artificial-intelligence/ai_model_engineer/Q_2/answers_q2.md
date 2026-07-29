# Pose Estimation for Padel Hits Test

---

#### Q1: The model shows strong performance on validation data but fails on new matches recorded in different clubs with different camera heights and angles. Hit detection accuracy drops significantly. Explain the most likely root causes of this failure and describe how you would systematically investigate and isolate the problem.

The most likely cause would be a domain shift respect to the data used in validation. Different camera heights and angles change how the body is projected in the image and so that the hit detector accuracy is dropping, as it may have used joint angles or motion features that were trained on a specific camera geometry. If viewpoint changes, the same real hit can produce different 2D keypoint trajectories.

The validation set should be built club-independent / camera-independent. If it comes from the same clubs, same cameras, or even same matches, the validation score may be optimistic. 

Assuming validation set correctness, the first action would be verifying whether the problem starts in the pose estimator or the hit-event classifier. In order to do so, I would separate the pipeline in both stages.  

On one hand, I would sample frames across new clubs' data with manual joint ground-truth annotations to extract pose metrics. I would manually annotate a small set of test videos from the new clubs with frame-level hit labels and keypoint GT so I can compute hit accuracy metrics. Even if pose metrics look reasonable and coherent with validation ones, they are not concluding in order to say pose estimation model is correctly predicting. To definitively confirm whether pose quality is the main cause or not, we can run two isolated tests:

- Feeding GT Poses into the hit-event classifier and computing the accuracy.
- Feeding predicted poses into the hit-event classifier and computing the accuracy.

If the first accuracy is high but the second one is low, then the problem has concluded to appear with the pose estimation model. However, if the first accuracy is already low, the classification logic would be concluded to be wrong and so that the hit-event classifier would need to be fixed.

---



#### Q2: After reviewing some samples, you discover that different annotators labeled keypoints slightly differently (especially wrists and elbows), and hit timestamps are sometimes shifted by a few frames. Analyze how inconsistent annotations affect both pose estimation and hit classification, how you would detect label-quality problems at scale and how you would decide whether to relabel, filter, or reweight parts of the dataset. Justify your decision criteria.

Wrong data will always make a system underperform and/or make wrong predictions. In our system's case:

- If keypoints are slightly differently labelled the pose estimation model can lose stability, increasing jittering and/or producing lower confidence scores. Apart from that, the classifier uses metrics derived from pose to classify. If specially wrists and elbows are wrongly annotated, these body parts' related metrics as it could be wrist speed, which are probably fed into the classifier. And for hit-event classifier being padel the sport that concerns this system, elbow and wrist are fundamental to be correctly annotated, as even small errors modify the hand-arm motion.
- If hit timestamps are shifted by a few frames, the model won't be able to learn when exactly the hit happens. Motion before and after the impact may be similar, so a few frames can confuse the model and disabling it to correctly classify the hits.

How would I detect label-quality problems at scale without inspecting frame by frame? 

- Regarding the pose estimator, I would measure inter-annotator agreement per-joint with the Mean Per Joint Position Error. That simple metric can leads us to detect bad label-quality.

- In terms of the label-quality for the hit timestamps, an informative indicator could the frame-offset diagreement between annotators. Frame-offset is just the difference in frames between two annotators for a hit being marked. Measuring the distribution of disagreement across all of our annotators, can leads us to flag who is shifted.

Decision criteria for relabelling, filtering or reweighting for each dataset part inconsistently annotated would be based on:

- Relabel is an exhausting task, so we must be sure about doing it. In view of that, I would just relabel if the class is high-valued, and the sample/s is/are recoverable.
- Filter out is the best option in noisier ambiguous samples that are difficult to annotate consistently, where relabelling would cost more than their value for our data.
- Finally, reweighting parts of our dataset should be done when samples are uncertain but not enough to be filtered out because still improve coverage diversity, so assigning them a lower training weight should solve the issue for them.

With reasons for each action exposed, for our task I would definitely prioritize relabelling wrists, elbows and hit timestamps, because they are crucial for the hit-event classification.

---

#### Q3: Smashes are rare in the dataset but very important for the product. The model often misses them while performing well on common hits. Without changing the model architecture, explain which dataset, training, and evaluation strategy changes you would consider. Here, the idea is to discuss tradeoffs and risks of each approach and how you would verify that improvements are real and not overfitting artifacts.

As long as the architecture must remain the same, the following suggestions could be taken into account for solving the stated issue:

##### 1. Dataset Changes

Given that smashes are crucial for the final product, dataset strategies may imply increasing our training samples for the model to learn the patterns accordingly and in the end detecting better the smashes. During the test, both in Q1 and Q2, data augmentation has been a discussed topic and here applies as well. Undersampling other classes to balance the dataset could be also an option. However, if the model performs accurately for other classes, we risk it doing worse for them. 

Oversampling the smash class entails the risk of the model memorizing background, angles and conditions instead of learning to classify them. To mitigate this, augmentation techniques like random erasing, changing the brightness, scaling, speed perturbations or frame dropping could be applied so oversampled clips remain diverse. Alternatively, SMOTE could be used to generate synthetic samples by interpolating in feature space.

##### 2. Training Changes

Assuming we still have low-representation for smashes' samples, and architecture must not be changed, the used loss function needs to be changed (or not the function at all but adjusted). As we have few examples for smash events, I would use a focal loss, with the adjustment of forcing the model to focus on hard examples, down-weighting easy examples. In this case, assuming we don't have a labelling problem as the stated in the previous question, no extra noise would be introduced (non-smash noisy samples would be considered difficult to classify), and it may work. The risk we are taking here is hurting our actual precision for other classes if we do not decide correctly our weights' values. 

##### 3. Evaluation Changes

Global metrics could be masking the smash-related issues of our model. That is why it is essential to compute per-class metrics. Furthermore, we could consider using a lower threshold for smashes to be detected. It has a risk of the model becoming too sensitive and increasing false positives, though. 

Besides that, an informative evaluation addition could be to inspect confusion rates between different classes and smashes. With that last addition, we could obtain insights and if the boundary between a certain event and a smash is ambiguous, make some adjustment in the training or dataset to better classify them. It is not a strategy but explainability measure.

From my point of view, there is not a correct answer as each has its pros and cons. However, for the problem stated, I would suggest adjusting the loss function, and then inspecting the confusion rates between events, as the risk is not such as oversampling or lowering the threshold. Depending on how it works, other options among the suggested (and of course others that may arise from discussing it with the team), might be considered.

Lastly, in order to verify improvements are real and the model is not overfitting I would conduct evaluation on an unseen players or scenarios only on smash events. Among the metrics we could compute from this test, the difference in recall and precision from the ones computed from last model's training, is the most informative one. If recall increases but precision drops, then the model would not have learned better features as we had expected. To end up with this question, it is important to check in each step smash classification upgrades are not degrading other classes' performance.

---



#### Q4: Two training runs with the same configuration produce noticeably different results. One model is stable but less accurate; another is more accurate but produces jittery keypoints and inconsistent hit timing. Provide a reasoning to decide which model is actually better for production. What additional tests, metrics, and scenario-based evaluations would you design to support the decision?

This decision is not as automatic as saying "the one more accurate is better in production". A more accurate model may still produce a worse experience for the user if hit timing jumps between nearer frames, or jittery keypoints can create a feeling of false motion. If the more accurate model has only a small accuracy gain but significantly worse jitter and unstable timing, I would choose the stable model for production, provided that minimum constrains are accomplished. Otherwise, if the accurate model is clearly better, then I would keep it and try to solve the instability, for instance by add some post-processing.

However, to support this decision, some additional tests could be conducted so the hypotheses confirms we are taking the right path.

- The ideal scenario for us would be the more accurate model to behave stable and reliable. To confirm in which degree this unstability is being produced, we could compute metrics related to temporal smoothness. For instance, the variance in frame-to-frame displacement or the frame-to-frame position displacement of the keypoints. I would do these both in a global metric and per-joint. As well , Percentage of Correct Keypoints is a metric that also could bring us useful information.

Besides that, we could compute when jittery keypoints between frames leads us to the stated inconsistency regarding hit timing. I would add two specific metrics for this: on one hand, I would measure the standard deviation of the frame-level hit prediction error, so the std is computed between the GT and predicted hit frame, and on another hand I would add a temporal IoU to evaluate if a hit falls within an acceptable temporal window. Which is acceptable or not should be discussed previously with the team, but I would suggest +/- 2/3 frames.

- Regarding the tests and scenario-based evaluations I would suggest to perform, I would target edge cases and conditions, so we compare both models in terms of temporal variance and stability. The following could be explicative for us:

1. Feeding static videos of an individual standing. We would expect from this an std of predicted coordinates over time approaching to zero.
2. Feeding both models with high-speed actions and directional changes. We would expect from this a stable frame to frame trajectory continuity.
3. To test occlusion and changing conditions, we would feed the model with sequences where body parts are occluded or temporary leaves the frame. We would expect the confidence to drop but not to have wrong high-confidence outputs.

---

#### Q5: Which model would you propose for this application? Why?

As for other question along this test, the lower the computational cost the better. With that assumption in mind, and some other's commented along this Q2 as the importance of wrists and elbows for classifying hit-events, I would suggest a pipeline that manages an acceptable trade-off between accuracy and computational cost.

1. A first lightweight player detector previous to the pose estimator. I would suggest YOLOv8n restricted to person detection. 

2. A small top-down pose model. The decision of using a top-down model instead of a bottom-up is that the first one could provide us with a better-quality set of keypoints specially for upper-body ones, as far as I am concerned. For this step I would suggest using a small RTMPose ([https://arxiv.org/abs/2303.07399](https://arxiv.org/abs/2303.07399)), as it can be a good balance between accuracy and speed.

3. A temporal hit-event classifier from sequences of keypoints over a sliding window. I found a good option for this step in a lightweight ST-GCN (Spatial-Temporal Graph Convolutional Network) ([https://arxiv.org/abs/1801.07455](https://arxiv.org/abs/1801.07455)), which treats the body as a skeleton graph and naturally models relationships. In order to be more specific, keypoints would be in 2D, and for the sliding window we could consider 15-20 frames.

With this pipeline, if joint jitter appears, it could be introduced a 4th step of post-processing, where we could add temporal smoothing for keypoint trajectories across consecutive frames.

Nowadays, there are YOLO-unified models that detect and predict keypoints at the same time, and, as far as I am concerned, they are optimized to work in production environments. However, the decision of splitting this pose estimation step in our pipeline is based on:

- A YOLO-Pose specific model may be less accurate than a top-down model as the one suggested on fine keypoints. Given that, and the complexity of keypoints such as elbows and wrists, makes me think a top-down model could work better, even if the efficiency is slightly lower.

- The complexity of debugging and optimizing a large end-to-end model may be higher than the suggested workflow in production.