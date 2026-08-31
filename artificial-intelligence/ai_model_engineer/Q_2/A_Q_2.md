
### Questions:
# 1. The model shows strong performance on validation data but fails on new matches recorded in different clubs with different camera heights and angles. Hit detection accuracy drops significantly.
<br>Explain the most likely root causes of this failure and describe how you would systematically investigate and isolate the problem.

One possible problem could be overfitting. The model may have learned very well from specific camera angles and heights, so when these conditions change, the model has problems detecting the hits correctly.

Another possible problem is that the training and validation datasets are too similar. If most of the videos have been recorded with similar camera positions, the model has not learned to generalize to other situations.

I would first compare the videos where the model works well with the new videos where it fails. I would check camera height, camera angle. Also with videos and images I will check the lighting and image quality.

If the problem is about data, I will need to add new data with different angles, camera heigth and retrain.

# 2. After reviewing some samples, you discover that different annotators labeled keypoints slightly differently (especially wrists and elbows), and hit timestamps are sometimes shifted by a few frames.
<br>Analyze how inconsistent annotations affect both pose estimation and hit classification, how you would detect label-quality problems at scale and how you would decide whether to relabel, filter, or reweight parts of the dataset. Justify your decision criteria.

For the keypoints, I would allow a small tolerance around the annotated point instead of requiring exactly the same pixel. For example, we could consider a small area around the wrist or elbow as acceptable.

The same idea could be applied to hit timestamps. Instead of requiring exactly one frame, we could allow a small range, for example ±2 frames, depending on the video FPS and the type of hit.

To detect problems at scale, I would compare the annotations from different annotators and look for cases where the difference is too large. I would then manually review a sample of these cases to check if the difference is acceptable, for example because of the tolerance area, or if it is a real annotation error, such as marking the knee instead of the elbow.

If an annotation is clearly wrong, I would relabel it. If the sample is too ambiguous or the keypoints cannot be seen correctly, I would remove it from the training dataset. If the difference is small but still useful, I could give less importance to these samples during training.


# 3. Smashes are rare in the dataset but very important for the product. The model often misses them while performing well on common hits.
<br>Without changing the model architecture, explain which dataset, training, and evaluation strategy changes you would consider. Here, the idea is to discuss tradeoffs and risks of each approach and how you would verify that improvements are real and not overfitting artifacts.

Add more smash examples to the dataset, ideally with different players, clubs and situations. To avoid first question error.

Give more importance to smashes during training, increasing the weight of smash errors.

# 4. Two training runs with the same configuration produce noticeably different results. One model is stable but less accurate; another is more accurate but produces jittery keypoints and inconsistent hit timing.
<br>Provide a reasoning to decide which model is actually better for production. What additional tests, metrics, and scenario-based evaluations would you design to support the decision?

First, I would test both models in different matches and situations to see if the differences are something specific to some matches or if they happen consistently.

If the results are consistently different, I would consider what the final product is going to be used for.

For example, if the product is a training application where we want to analyze and correct the player's movements, I would probably choose the first model because stable keypoints are very important, even if it has lower accuracy.

On the other hand, if the product is mainly focused on statistics, such as counting smashes, volleys or forehands, I would probably prefer the second model because accuracy is more important in this case, as long as the jitter does not affect the final statistics.

I would therefore test both models with different matches and compare accuracy, keypoint stability, hit timing and computational cost before making the final decision.

# 5. Which model would you propose for this application? Why?

For this application, I would choose the model with the higher accuracy, because the main goal of the system is to correctly identify the different types of hits, such as smashes, volleys, forehands and backhands.




Note

I used AI assistance to improve the grammar, wording and structure of my answers. However, the ideas, reasoning and proposed solutions are my own, and I am fully responsible for the content and decisions presented in this document.