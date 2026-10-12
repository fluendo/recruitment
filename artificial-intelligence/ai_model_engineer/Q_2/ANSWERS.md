# Q2 - Pose Estimation for Padel Hits — Analytical Reasoning Test

### Question 1

**The model shows strong performance on validation data but fails on new matches recorded in different clubs with different camera heights and angles. Hit detection accuracy drops significantly.<br>Explain the most likely root causes of this failure and describe how you would systematically investigate and isolate the problem.**

I would investigate first whether there is any data leakage between the training and validation sets, for example frames from the same game, camera, or club. If the splits are correct, I would compare the distributions of the validation set and the new clubs to detect domain shifts in lighting, camera angles, resolution, or occlusions. If I find underrepresented conditions, I would incorporate representative data, retrain the model, and evaluate it on an independent test set with unseen clubs.

### Question 2

**After reviewing some samples, you discover that different annotators labeled keypoints slightly differently (especially wrists and elbows), and hit timestamps are sometimes shifted by a few frames.<br>Analyze how inconsistent annotations affect both pose estimation and hit classification, how you would detect label-quality problems at scale and how you would decide whether to relabel, filter, or reweight parts of the dataset. Justify your decision criteria.**

First, I would review the annotation guidelines and assess the disagreement among annotators, especially for wrists/elbows and hit timestamps. I would also check whether the noise is related to challenging cases such as motion blur or occlusions. Next, I would review or relabel the inconsistent samples, validate the quality of the new annotations, and only then retrain and re-evaluate the model.

### Question 3

**Smashes are rare in the dataset but very important for the product. The model often misses them while performing well on common hits.<br>Without changing the model architecture, explain which dataset, training, and evaluation strategy changes you would consider. Here, the idea is to discuss tradeoffs and risks of each approach and how you would verify that improvements are real and not overfitting artifacts.**

I would evaluate precision and recall specifically for smashes and examine their errors to confirm the impact of the issue. I would check whether the smashes are sufficiently represented, diverse, and correctly annotated. If there is a coverage issue, I would add targeted, representative data and evaluate techniques such as oversampling or fine-tuning.

### Question 4

**Two training runs with the same configuration produce noticeably different results. One model is stable but less accurate; another is more accurate but produces jittery keypoints and inconsistent hit timing.<br>Provide a reasoning to decide which model is actually better for production. What additional tests, metrics, and scenario-based evaluations would you design to support the decision?**

First of all, I would define the product requirements and hard constraints. I would not automatically choose the second model just because it has the highest accuracy: that accuracy is measured offline and does not necessarily translate into better product performance. I would compare both models using representative data and evaluate whether smoothing techniques can stabilize Model B without adding too much latency or losing fast movements. Then I would make my choice based on those trade-offs.

### Question 5

**Which model would you propose for this application? Why?**

First, I would define the product requirements and hard constraints: target hardware, latency/FPS, memory, licensing, and required accuracy. Next, I would create a shortlist of pose estimation models using model cards and public benchmarks, but I would make the final decision based on my own benchmarks using data representative of padel and the target hardware. I would compare accuracy, temporal stability, latency, VRAM, etc. YOLO Pose, RTMPose or MoveNet could be on the shortlist.
