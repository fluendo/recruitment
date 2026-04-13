# Fluendo AI Model Engineer Technical Test

Roger Esteve Sanchez

---

## Part 1: Face Anonymization (Q1)

### 1. How can we improve face detection by adding reidentification to anonymize only specific individuals while keeping computational cost low?

To achieve reidentification while keeping the computational cost as low as possible, my strategy would be to use a tracker (like ByteTrack) to avoid reidentifying subjects every frame.

First, relying on a lightweight tracker to maintain identities across frames, triggering only the reidentification every 30 frames (for example) or when a new face enters the scene.

To optimize this further, I would try to implement an adaptive reID rate. For example, if a face is isolated in space, we can probably trust the tracker to know that person is the same without the need for reID. But in cases where faces intersect or are extremely close, we could try to increase the reID frequency to anticipate tracker errors. I think that could be a way to find a good balance.

In terms of scalability, if targets are few, standard NumPy/PyTorch would probably handle the job, but for scenes with lots of targets, a vector database would probably be required to maintain performance.

Another trick we could use to squeeze out performance would be to run the face detection model on a downscaled version of the frame. 

Finally, all these tricks come with their own tradeoffs, mostly trading accuracy for speed, so cautious experimentation would be required to find the optimal solution. I cannot choose a precise solution without practical experimentation.

### 2. How can we extend an anonymization pipeline with depth-based filtering to blur only background objects efficiently?

I find this question a bit ambiguous. I see it could mean two different scenarios: one would be to blur the entire background, and the other is to blur faces based on whether they are in the foreground or the background.

First, supposing that we have the data from a 2D camera and not a kind of camera that would give you the depth map natively (like Intel RealSense, LiDAR, Time-of-Flight sensor, etc.), we would have to use a Monocular Depth Estimation model. In particular, small models like FastDepth or MiDaS-v2-tiny. In order to be able to run it in real-time, we would probably have to experiment with downscaling of the frame in order to optimize the depth estimation further.

* **Scenario A (Blurring entire background):** It would be as easy as creating a mask based on a threshold where a pixel value of `0` would mean background and `1` would be foreground. Then we apply a Gaussian blur only to the background pixels.
* **Scenario B (Selective face blurring):** I would average the pixels corresponding to the bounding box of found faces and use that as the depth value of the face. Then use a threshold to decide whether the face has to be anonymized or not. 

Actually, this method is simple but would have its problems because the face bounding box will probably include parts of the background in the corners. A solution to this would be to use just the pixels based on an ellipse inside the bounding box that would remove the background pixels, or use the **median** instead of the average, or the mode (most frequent value) as the depth value of each face.

### 3. Imagine we have this case: our face detection system does not detect properly (bad accuracy) dark-skins nor bald-people; how would you improve system accuracy?

The first thing I would check is the class imbalance of the dataset. If I saw a clear class imbalance, and assuming we cannot ask for more samples, I would implement a weighted loss function—which is very easy to implement in the top frameworks—which would balance the loss value and require a good prediction in all classes to actually decrease consistently.

Actually, researching now I found **Focal Loss**, developed by Facebook, which seems to be the gold standard, because it dynamically adjusts to focus the learning on the hard examples. I would love to try that.

What we can also do is implement Data Augmentation, or in case it is already implemented, also using a weighted DA to account for the class imbalance.

If none of this works, it would probably either be due to the model being too small (causing underfitting) or too big (causing overfitting). This would be a case where we would have to do more in-depth experiments via hyperparameter tuning and enhancing the architecture to try and increase the accuracy.

### 4. Imagine this other case: let's assume we have added several bald people imagery to our dataset and we still are not increasing the accuracy. What may be happening? How would you fix this?

This could be a case of the dataset being too small. Even when being balanced, the dataset may not contain enough samples to help the model learn correctly. So I would implement (or increase) data augmentation.

And again, it could be that the model is not the correct size, so I would also do experiments with hyperparameter and architecture tuning.

### 5. Which MLOPs infrastructure would you design to manage this system? Just provide a high-level design and which tools would you use.

1. **Raw data storage:** AWS S3 or Google Cloud Storage.
2. **Dataset preparation (labelling/cleaning):** In this step, for my Degree’s Thesis I developed my own labelling and cleaning pipeline, but I’m also familiar with Roboflow. There are also many alternatives like CVAT or Label Studio.
3. **Model Development:** PyTorch
   * *Preprocessing:* Data Augmentation.
   * *Model Training:* I always use W&B (Weights & Biases) for experiment tracking.
   * *Model Evaluation:* W&B also helps handle the evaluation metrics, allowing you to compare models visually in real-time and generate reports.
4. **Model database (registry):** MLflow Model Registry. It is the industry standard and is compatible with W&B.
5. **SDK/API:**
   * *FastAPI (Python):* For a fast and easy starting point.
   * *NVIDIA Triton Inference Server or TorchServe:* For high-performance serving. 
   * *Deployment:* This step must come with CI/CD. The workflow I’m most used to is automatically containerizing the model with Docker or deploying to the API when a model hits the “production” branch on GitHub. For an anonymization pipeline governed by the EU AI Act, sending raw video to the cloud is legally risky. Therefore, deployment might need to happen on **Edge devices** (like an NVIDIA Jetson on-site).
6. **Performance analysis:** Monitor the model in production to send edge cases back to "Step 2".
   * *Evidently or Arize:* For model performance metrics.
   * *Prometheus + Grafana:* For hardware metrics.

### 6. From end user POV, which techniques would you use to improve final solution user experience?

UX for me is one of the hardest things to brainstorm at first because it is common that you think about lots of things to improve it, but once the product reaches the user things tend to not work as you expected. So for me, it highly depends on how the end product is going to look.

But a feature I would add is a way for the user to be able to see which faces the system has stored, change their name for easier identification, and a fast error correction system where the user can correct a misclassified face by telling the system the correct person that face should be tagged as.

Also, the system could tell you when it is less confident about its prediction. So users can see and actuate in the moment. This confidence check could also serve as a way to trigger the increase in reID until the model is more confident that the face is correctly classified.

### 7. (EXTRA) Provide a code snippet (practical example) on how would you implement any of these applications: reidentification or depth estimation, for face anonymization.

```python
import numpy as np

def anonymize_background_faces(rgb_frame, depth_threshold, face_detector, depth_estimator, median=True):
    # Get depth map and face bounding boxes
    depth_map = depth_estimator.predict(rgb_frame)
    bounding_boxes = face_detector.detect(rgb_frame)
    
    for box in bounding_boxes:
        x1, y1, x2, y2 = box.get_coordinates()
        
        # Crop depth map to face region
        face_depth_crop = depth_map[y1:y2, x1:x2]
        face_depth_value = -1
        
        # Calculate depth using median or mode
        if median: 
            face_depth_value = np.median(face_depth_crop)
        else:
            values, counts = np.unique(face_depth_crop, return_counts=True)
            face_depth_value = values[np.argmax(counts)]
            
        # Anonymize if the face is in the background
        if face_depth_value > depth_threshold:
            face_roi = rgb_frame[y1:y2, x1:x2]
            blurred_face = apply_gaussian_blur(face_roi, kernel_size=(51, 51))
            rgb_frame[y1:y2, x1:x2] = blurred_face
            
    return rgb_frame