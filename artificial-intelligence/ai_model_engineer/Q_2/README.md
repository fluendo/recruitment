
# Pose Estimation for Padel Hits — Analytical Reasoning Test

## Context

You are evaluating a Deep Learning system designed to detect padel hits using **pose estimation** from video. The model predicts player keypoints and derives hit events (forehand, backhand, smash, volley, serve) from body and arm motion. The system is already trained and partially deployed, but several performance and reliability issues have appeared in real scenarios.

<div style="text-align: center;">
  <img src="../imagery/padel.jpg" alt="Padel" width="30%" style="display: inline-block; margin-right: 2%;">
</div>

### Assumptions:
- We want to reduce computational cost so the lighter the models, the better 

### Questions:
1. The model shows strong performance on validation data but fails on new matches recorded in different clubs with different camera heights and angles. Hit detection accuracy drops significantly.
<br>Explain the most likely root causes of this failure and describe how you would systematically investigate and isolate the problem.

This is a typical sign of overfitting. The model is not learning generalized biomechanics, but rather memorizing specific patterns from the training data. Then, when new data (for instance, from a new padel club) is introduced, the model doesn't recognize the variations in environment, lighting, or camera heights, and fails.

Although model architecture choices could play a role, I would start by investigating the data and feature space:
- First, I'd check if applying a geometric normalization step (such as centering the skeleton on the player's hips and scaling by torso length) improves performance. This removes simple camera-scale differences.
- Then, I would segment the failing test matches by camera angle and height to evaluate model accuracy per subset. If accuracy drops only on some specific angle cameras, it confirms a viewpoint generalization issue rather than a general model failure.
- Also, I would analyze which joints or motion features contribute most to the hit classification. This reveals whether the model is relying on unstable spatial cues instead of core arm-swing trajectories. This could be done by using saliency maps.
- Finally, to fix the issue, I would apply geometric data augmentations during training to artificially expose the model to various viewing angles. Alternatively, fine-tuning the model on a small batch of data from the new clubs would probably adapt it to the new perspectives.

2. After reviewing some samples, you discover that different annotators labeled keypoints slightly differently (especially wrists and elbows), and hit timestamps are sometimes shifted by a few frames.
<br>Analyze how inconsistent annotations affect both pose estimation and hit classification, how you would detect label-quality problems at scale and how you would decide whether to relabel, filter, or reweight parts of the dataset. Justify your decision criteria.

Inconsistent annotations severely impact both pose estimation and hit classification by introducing spatial and temporal noise. For pose estimation, mislabeled joints like wrists and elbows confuse the model with mixed signals for identical physical postures and degrade performance for joint localization. For hit classification, even a small temporal shift misaligns the sequence, feeding the network frames of people standing while labeling them as the exact moment of impact (and vice versa). This breaks temporal models like LSTMs or Transformers, leading to false positives and mistimed detections.

To detect label errors without manual inspection, one method is to look at which training samples give the model the highest prediction loss. This naturally sorts most of the mislabeled instances to the top of our review list. Another powerful idea is to use biomechanical sanity checks which enforce human anatomy. Since a person's forearm length doesn't suddenly change from frame to frame, or elbows bend backward past natural limits, any of these unphysical scenarios can be automatically flagged as errors.

Finally, tracking annotator metadata helps catch systematic human habits. If you know which person or labeling team handled a specific batch of videos, you can audit them separately. For instance, if one particular annotator consistently marks hit timestamps a few frames too early across all their files, you’ve found a systematic bias rather than random mistakes, allowing you to quickly target and fix just that specific batch. But what if we don't know who made the mistake upfront? To catch those hidden errors, we can use a pre-trained model as a baseline and compare its predictions against our dataset's labels. By running distance metrics like Euclidean distance between the two, we can automatically flag suspicious data where the annotations differs from what the baseline expects.

When deciding how to handle the compromised data, the choice between filtering, relabeling, or reweighting depends on the scale and nature of the error. Filtering is best if the noise is low and randomly distributed. Relabeling is necessary if critical or systematic errors are found in the data (such as an entire batch from a specific annotator being skewed). Finally, reweighting via loss functions works well when dealing with moderate, uniform noise, allowing us to shift attention away from noisy samples so the model focuses on clean data.

3. Smashes are rare in the dataset but very important for the product. The model often misses them while performing well on common hits.
<br>Without changing the model architecture, explain which dataset, training, and evaluation strategy changes you would consider. Here, the idea is to discuss tradeoffs and risks of each approach and how you would verify that improvements are real and not overfitting artifacts.

Before jumping right into the possible solutions, it is important to mentions that choosing the right metrics is also crucial. When evaluating rare classes like smashes, standard accuracy metrics are completely unreliable. You really have to look at precision, recall, and the F1-score backed up by a detailed confusion matrix. 

To handle rare but important classes like smashes without changing the model architecture, several dataset and training strategies can be applied:
- First, we could oversample the minority class by implementing targeted data augmentation. We could apply geometric transforms (like rotation, scaling, and mirroring)and temporal speed variations.
- Second, implementing cost-sensitive learning by using a weighted loss function such as focal loss. This will heavily penalize the model when it misses a smash, forcing it to pay attention to those specific cases.
- Finally, balancing the overall dataset isn't enough if individual training batches end up completely skewed. Because of that, it's also crucial to use stratified batch sampling to guarantee that every training iteration includes a representative proportion of smash sequences, rather than letting random shuffling create unbalanced batches.

Playing with these fixes comes with its own set of risks and tradeoffs. If we boost the loss weights too much, the model could become too sensitive and start seeing smashes everywhere, turning normal serves or hard volleys into false positives. On the other side, if we apply too much data augmentation, we would end up teaching the model weird, unnatural body movements that could ruin the pose estimation.

Lastly, to verify that any improvements are genuine and not just overfitting artifacts, rigorous validation protocols are essential. we should use a strict out-of-sample strategy, ensuring that our validation and test sets contain smash clips recorded at completely new clubs with different players and camera angles. If performance remains robust on these unseen environments and the false-positive rate on standard hits hasn't blown up, you can confirm that the model has truly learned meaningful features rather than just memorizing training data.

4. Two training runs with the same configuration produce noticeably different results. One model is stable but less accurate; another is more accurate but produces jittery keypoints and inconsistent hit timing.
<br>Provide a reasoning to decide which model is actually better for production. What additional tests, metrics, and scenario-based evaluations would you design to support the decision?

From a product standpoint, consistency usually beats raw statistical peaks. Of course, we have to weigh the actual performance difference among models against what the user needs; if the accuracy gap is massive and the core priority is strictly hit classification, the second model might theoretically be tempting. However, we have to remember that joint misclassification will likely interfere in hit classification anyway, since the classifiers might rely directly on those keypoints.

More importantly, production credibility takes a massive hit if a model displays surreal errors; like putting ankles where wrists belong or placing elbows on knees. In padel, where users actually see the skeleton overlays and rely on precise hit-detection timestamps, erratic visuals instantly destroy the user experience. 

Therefore, to rigorously defend our choice for production, I’d design a few specific evaluations to help me make the final decision:
-  Compute Mean Absolute Acceleration (MAA): it measures how drastically a joint's speed changes from one frame to the next. Human limbs have mass and inertia; they cannot instantly teleport or change direction in a single 30ms video frame. If a model is jittery, a wrist joint might be predicted at pixel coordinate (100, 200) in frame one, (130, 210) in frame two, and (95, 205) in frame three. By calculating the MAA across the video, a stable model will show smooth, continuous changes. A jittery model will display massive acceleration spikes because the skeleton is shaking back and forth. Therefore, a high MAA score is a clear red flag that the pose estimation is unstable.
- Regarding the inconsistent hit timing, I’d look closely at the actual distribution of errors rather than just a single pass/fail score. To evaluate this, I would plot a histogram of the frame offsets (Predicted timestamp - Gold standard timestamp). You want to see a tight, bell-shaped distribution centered near zero with a narrow spread, rather than a flat, unpredictable spray of errors. By visualizing the histogram, we'll be able to see whether the errors are tightly clustered with a predictable offset - which we can easily fix programmatically - or if they are scattered wildly across a wide window, exposing a volatile system that will completely break user experience.
- Finally, i would also ensure that the dataset contain specific challenges. Standard validation datasets are usually filmed in perfect conditions with professional cameras. It is interesting to know, as well, how models behave with real-world chaos such as:
  - Outdoor cours lightning: Outdoor padel courts often suffer from uneven sunlight. If half the court is hit by bright sun and the other half is hit by shadow from the roof structure, the high contrast can trick the model into completely missing limbs on the dark side of the body.
  - Glass wall reflections: Padel courts include glass walls. When overhead lights hit the glass at weird angles, it can create reflections and ghost images that can easily misslead a pose model into tracking a reflection instead of the actual player.
  - Doubles overlap: Padel is a doubles game, meaning partners and opponents are constantly crossing paths, stepping in front of each other. When bodies overlap, pose models frequently get confused about whose arm belongs to whose skeleton, causing limbs to "swap" identities.

5. Which model would you propose for this application? Why?

For this application I'd definitely start with [YOLO-pose](https://docs.ultralytics.com/tasks/pose) and if time allows it, compare its performance with some other state-of-the-art model, such as [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).

The main reason is that it’s a single-stage model, meaning it handles both bounding box detection and keypoint estimation in one single loop. You don't have to run a separate detector first and then crop out every individual player, which saves a massive amount of compute time and makes it ideal for real-time deployment.

Another benefit is that it comes in a range of different sizes. Althoug I would start testing with the Nano or Small variants to keep things as lightweight as possible, I would also like to try with a Medium model too, just to see if a bit extra capacity helps stabilize the keypoints without totally crashing our latency.

Regarding hit classification, in order to handle the temporal side of the pipeline once the spatial tracking is already efficiently managed by the YOLO backbone, I’d pair it with a lightweight 1D Temporal Convolutional Network (1D-TCN) acting as our action classification head.

Instead of evaluating isolated frames, the 1D-TCN would get as input a rolling window of keypoint coordinates and velocities (YOLO output) and spot the kinematic distictive features of a padel hit.