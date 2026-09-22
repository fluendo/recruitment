
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
- Second, implementing cost-sensitive learning by using a weighted loss function such as Focal Loss. This will heavily penalize the model when it misses a smash, forcing it to pay attention to those specific motion profiles.
- Finally, balancing the overall dataset isn't enough if individual training batches end up completely skewed. Because of that, it's also crucial to use stratified batch sampling to guarantee that every training iteration includes a representative proportion of smash sequences, rather than letting random shuffling create unbalanced batches.

Playing with these fixes comes with its own set of risks and tradeoffs. If we boost the loss weights too much, the model could become too sensitive and start seeing smashes everywhere, turning normal serves or hard volleys into false positives. On the other side, if we go too crazy with data augmentation, we would end up teaching the model weird, unnatural body movements that could ruin the pose estimation.

Lastly, to verify that any improvements are genuine and not just overfitting artifacts, rigorous validation protocols are essential. we should use a strict out-of-sample strategy, ensuring that our validation and test sets contain smash clips recorded at completely new clubs with different players and camera angles. If performance remains robust on these unseen environments and the false-positive rate on standard hits hasn't blown up, you can confirm the model has truly learned meaningful features rather than just memorizing training data.

4. Two training runs with the same configuration produce noticeably different results. One model is stable but less accurate; another is more accurate but produces jittery keypoints and inconsistent hit timing.
<br>Provide a reasoning to decide which model is actually better for production. What additional tests, metrics, and scenario-based evaluations would you design to support the decision?

From a product standpoint, consistency beats raw statistical peaks. In padel, if our hit-detection timestamps or skeleton overlays are erratic, the user experience completely falls apart. To defend this choice for production, I’d design a few specific evaluations focused on real-world behavior:
-  

5. Which model would you propose for this application? Why?
