1. The model shows strong performance on validation data but fails on new matches recorded in different clubs with different camera heights and angles. Hit detection accuracy drops significantly.
Explain the most likely root causes of this failure and describe how you would systematically investigate and isolate the problem.

This is probably related to a domain shift issue (similar to the one we had with bald-people). In this specific case of pose estimation, the training dataset correctly represents the evaluation and validation datasets (because we obtain good results). However, when it comes to production systems, the camera height, angle, and ilumination conditions change, so probably our training dataset actually does not represent the production data, and thus the system performance drops. We can further diagnose the issue by inspecting the image properties of the new footage in these clubs. Maybe, the problem is as simple as the ball color (in this club the ball is white instead of yellow). As a general solution, we can leverage the trained model and fine tune it with the new collected data in clubs, improving the model robustness.

2. After reviewing some samples, you discover that different annotators labeled keypoints slightly differently (especially wrists and elbows), and hit timestamps are sometimes shifted by a few frames.
Analyze how inconsistent annotations affect both pose estimation and hit classification, how you would detect label-quality problems at scale and how you would decide whether to relabel, filter, or reweight parts of the dataset. Justify your decision criteria.

This dataset issues may affect model performance in two ways:
- If labeled keypoints are not very far from the "common sense" wrist and elbow position of a player, these "errors" may introduce a positive data variability, forcing the model to map an elbow to a body region instead of a single point, improving the model genralization. A similar rationale applies to hit timestamp.
- If labeled keypoints are quite wrong, this can generate a negative effect on the model performance, as it may generate training noise and a jitter efect in the pose detection. Similar to the hit timestamp, we may see that actions are not detected, specially if they are fast (e.g., a smash).

To detect these errors at scale, we should analyze the training data and look for outliers. For example, we can analyze all the pose positions labeled as a smash and: (i) look for the mean properties of the keypoints and (ii) identify those samples that are two or three sigmas apart (probably misslabeled). 

3. Smashes are rare in the dataset but very important for the product. The model often misses them while performing well on common hits.
Without changing the model architecture, explain which dataset, training, and evaluation strategy changes you would consider. Here, the idea is to discuss tradeoffs and risks of each approach and how you would verify that improvements are real and not overfitting artifacts.

This issue may be related to a data imbalance. The loss function averages all the samples equally. Since the model is not penalized much for missing a smash (because it is a rare event), it may not be learning to detect them. We can try to balance the dataset by oversampling the smash class or using a weighted loss function (giving more weight to a smash class). However, this can improve the false positive rate. Therefore, we should include samples where the player moves fast or tries similar actions to a smash to make sure it is learning to detect them, not overfitting, and is robust to similar actions. In evaluation, accuracy is not the best metric: others such as precision, recall, or F1-score may be better. The evaluation should be as close to reality, so evaluation  should not be augmented with more smash events. 

4. Two training runs with the same configuration produce noticeably different results. One model is stable but less accurate; another is more accurate but produces jittery keypoints and inconsistent hit timing.
Provide a reasoning to decide which model is actually better for production. What additional tests, metrics, and scenario-based evaluations would you design to support the decision?

5. Which model would you propose for this application? Why?

I would choose the stable model, because:
- The perceived quality of the model is sometimes more important than a "perfect" accuracy. A stable model, even if slightly less accurate, feels solid and reliable.
- The jittery keypoints and inconsistent hit timing of the other model may lead to a poor performnace of other high-level, dependant applications (e.g., speed calculation, trajectory prediction, etc.).

The additional metrics and tests I would design are:
- A metric that counts how many times a keypoint moves beyond a biological possibility (e.g., a wrist moving 30cm in 0.03 seconds).
- A test where both models are compared on a video of a person standing perfectly still.
- A test where both models are compared on a fast-player moving video (e.g. a person performing a smash).
- A test with oclusion phenomena (the flicking model may fail here).