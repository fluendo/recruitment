# Fluendo AI Model Engineer Technical Test - Q2

**Roger Esteve Sanchez**

---

### 1. The model shows strong performance on validation data but fails on new matches recorded in different clubs with different camera heights and angles. Hit detection accuracy drops significantly. Explain the most likely root causes of this failure and describe how you would systematically investigate and isolate the problem.

That sounds like a case where the training/validation dataset does not match the real world scenarios correctly. Probably the dataset is too small or was recorded in a very controlled scenario. 

What I would first do is implement data augmentation or increase in case it already was implemented. Although probably the best approach to solve this problem would be to increase the size of the dataset with samples in different scenarios that better match real cases, I would try first with techniques like cross validation, better data augmentation, checking class imbalance, etc. Techniques that are easier to implement and that do not require increasing the size of the dataset that, from experience I can tell it is not always as easy as it sounds.

### 2. After reviewing some samples, you discover that different annotators labeled keypoints slightly differently (especially wrists and elbows), and hit timestamps are sometimes shifted by a few frames. Analyze how inconsistent annotations affect both pose estimation and hit classification, how you would detect label-quality problems at scale and how you would decide whether to relabel, filter, or reweight parts of the dataset. Justify your decision criteria.

This is such an important problem, not only in pose estimation and hit classification but in all deep learning datasets. Inconsistent annotations would make the model also provide inconsistent predictions. At the end, the model is trying to minimize the loss function, so, the more inconsistent and contradictory a dataset is the harder it is for the model to converge to the mathematically optimal solution, because some samples are teaching the model the wrong solution.

The first thing I would find necessary is to establish a set of strict guidelines. A document that must explicitly define the pixel location for every keypoint or the exact frame where a hit must be tagged. It would also have to come with a set of explanations for some edge cases and it could be updated as some other edge cases are found.

To detect label quality I would implement cross-labelling in the labelling team. A system that makes the same image or dataset be tagged by multiple people. That way we could compare the different decisions and find outliers that might have to be relabeled or annotators that require retraining. 

Also, in case we got a robust enough model, I would send high-loss model predictions to senior annotators to see if it is a real model error or it is exposing a human error in the labelling process. Both of these last solutions require an extra verification, the key would also be to optimize this procedures, per example, how many people are required to tag the same sample.

### 3. Smashes are rare in the dataset but very important for the product. The model often misses them while performing well on common hits. Without changing the model architecture, explain which dataset, training, and evaluation strategy changes you would consider. Here, the idea is to discuss tradeoffs and risks of each approach and how you would verify that improvements are real and not overfitting artifacts.

**Dataset changes:**
* **Record more smash samples:** This would make the model have more examples of smashes to learn from but, would take more work and money. You also have a risk of overfitting if you only record smashes from one person.
* **Targeted Data Augmentation:** Apply data augmentation more aggressively to the smash samples. The tradeoffs here is increasing the training time and possibly creating mechanically impossible samples if you do not tune correctly the parameters (eg. extreme rotation).

**Training changes:**
* **Weighted Loss:** Increase the penalty of smash classification errors so the training focuses more on the smashes. By doing this we will increase smash recall but likely decrease precision (more false positives as other hits get misclassified as smashes).

**Evaluation changes:**
* **Smash specific metrics:** Instead of just looking at the overall accuracy, I would also start using the specific f1, precision, recall and accuracy for the smash label. If smashes are 10% of the data and the model has an overall 90% of accuracy, that is useless data if the specific smash accuracy is 0%, per example.
* **Non-random data splitting:** If the samples of a specific label are low it is very possible that this label becomes underrepresented if the data split is done randomly. The solution is to ensure the data splits are consistent and that the labels are represented with the same ratio in train, validation and test splits.

### 4. Two training runs with the same configuration produce noticeably different results. One model is stable but less accurate; another is more accurate but produces jittery keypoints and inconsistent hit timing. Provide a reasoning to decide which model is actually better for production. What additional tests, metrics, and scenario-based evaluations would you design to support the decision?

What I understand as a stable and less accurate model is a model that misses the keypoints by a few pixels but has a consistent error. And the other model has a lower average error but the individual errors have a higher variance. 

With this information I would choose the stable model to be run in production because it will be easier to work with. If it always fails the same way it could be corrected by postprocessing while a jittery result might be harder to apply postprocessing to because of the randomness. Moreover if the position of the keypoints is used for some sort of physics calculation like speed, a jittery result would break the math calculations while a stable result would maintain calculations correct even if all points are off by a few pixels.

In order to test this behaviour I would use a preannotated video to be able to track the error difference between frames. With this, we would be able to calculate average error difference (jitter), standard deviation, max/min value. All these metrics evaluate how stable a model is apart from being accurate or not.

I would also present a side by side comparison of the same video tagged by both models and have a panel of human vote for their preferred option. The most probable winner would be the stable model since the unstable version would appear glitched.

It is also important to note that if the event classifier model is based on the keypoints predicted by the previous model, the more stable results will be easier to predict rather than the noisy data generated by the unstable model.

### 5. Which model would you propose for this application? Why?

Currently the state of the art for accuracy/speed tradeoff is [RTMPose-m](https://arxiv.org/abs/2303.07399) reaching a 75.8% accuracy on COCO MAP with a latency of 2.3ms (430fps) in a 1660Ti GPU. It is robust to resolution changes and as of 2026 it is still cited as a robust performer in real world scenarios.

When comparing models, if we search for the extreme we have HRNet with very good accuracy but also very computationally expensive, while models like Google's MoveNet is very fast but loses on accuracy.

RTMPose-m is on the sweet spot for the accuracy/speed tradeoff while also being robust to resolution changes which I think would probably the best option.