## Question 1.1: 
How can we improve face detection by adding reidentification to anonymize only specific individuals while keeping computational cost low?

### Answer:

Whilst face detection and then blurring is a simple task that can achieved by a good face detection + extractor + blurring, adding re-identification, which means to first identify the detected faces and then blur only selective ones, has more moving parts to it than simple face detection. 

I have divided the task into 8 main steps: 

Step 0: Start with all detected faces blurred. A face remains anonymized until there is sufficient evidence that it belongs to an approved identity.

1. Face detection - using the opencv SSD detector with a resnet-10 backbone. It has a good accuracy + fast. 
    1. Blur the detected faces. (so we start with blurred faces)
    2. Output are bounding boxes for each detection.
2. Assign a tracking ID to each face. 
    1. I have kept tracking first (as opposed to second in fluendo’s pipeline) is because we can use a tracker to atleast know that this is the same person. (so we can aggregate embeddings later). 
3. Face Quality filtering - filter out low quality face observations in frames where the detected face is not good enough to generate an embedding.
    1. using the detection confidence and blur/sharpness of face.
    2. if the confidence is higher than a threshold, move to next step, otherwise we wait for next frames. 
4. Face Embedding - for each track, after the quality is good, we do a face embedding.
    1. generate face embedding (can be  512 dimensional in size). 
5. Embedding Aggregation - do step 4 for several frames and then aggregate to capture better embeddings for each tracked face. 
    1. We do a simple aggregation here over frames - but other techniques like attention pooling can be used as well. 
    2. this is being done for each tracked faces.
6. Embedding comparison - we compare the aggregated embeddings to the saved faces. 
    1. If the # of faces are high, we can use a hierarchical index over the embedding vectors to search and find matches faster. 
    2. Select the top match if it passes a predefined threshold.
7. Apply privacy decision - to keep the blur or to remove it.
    1. if a saved face is matched with high confidence, un-blur that particular tracked face. 
8. Periodic Re-identification - rerun re-identification to verify that the track's identity is still correct and detect tracking drift or identity switches.. 
    1. If a face is being tracked and is un-blured, re-identify it after a number of frames (e.g., 100), or sooner if tracking confidence drops.
    2. Also, between re-identification operations, the confirmed identity is cached against the track-ID, so we do not need to run the embedding model on every frame.
	
	
	

## Question 1.2: 
How can we extend an anonymization pipeline with depth-based filtering to blur only background objects efficiently?

### Answer

Depth based estimation can help improve the face anonymize pipeline by identifying faces that are in the background and then blurring them directly. 
This adds both robustness and speed improvements since we can optionally skip the re-identification step for faces that are in the background (farther away) 
and at the same time have low quality frames for face. 

Using a lightweight model like ZipDepth (6.1M parameters and open source), it will barely add any overhead to the current face anonymization pipeline discussed 
in Q1, whilst at the same time running in parallel to our face detection and tracking steps. 

The way that it works is the follows: 

- **Tracking:** runs the same as before.
- **Face detection:** runs the same as Q1 and gives us the bounding boxes for detected faces.
- **ZipDepth estimation:** runs side by side with face detection to output a depth map, i.e., a depth value attached to each pixel in the frame.
- **Depth for extracted faces:** Given the bounding boxes for detected faces, we extract the depth values for each face.
    - For each face, we calculate a median depth value (median of all depths for pixels in the face).
- **Depth Comparison**: We then create an Area of Interest around the face in order to compare the face depth against background depth.
    - Create a ring around the face
    - remove any other detected objects from that ring (so to only have background)
    - Calculate the median depth of this ring
    - Compare the median depth of this ring against the median depth of the face.
    - If median face depth < median ring depth:
        - the face is in the foreground - closer.
        - Otherwise the face is the background.
    - Comparing with local surroundings makes more sense rather than a fixed threshold like 10 meters.
- **Resume Privacy Pipeline:** If the face is in the foreground, we resume our privacy pipeline of re-identification, if its background, we keep the blur.



## Question 1.3: 
Imagine we have this case: our face detection system does not detect properly (bad accuracy) dark-skins nor bald-people; how would you improve system accuracy?

### Answer

I would divide this problem into two parts: 

1. Identify if there is a systematic problem regarding dark skin individuals and bald people and identify if there is a pattern to the errors. 
2. Apply fixes. 

**Part 1**: The first part is to identify if the model is actually making mistakes for these two categories. 

1. We create a dataset containing samples from both light and dark skins as well as both with-hair and bald people. A dataset that is representative of our real world use. 
2. Evaluate the model on this dataset to eliminate any biases during testing the model. 
3. Once the evaluation is done, verify the top errors and **recall/miss rate** for these two categories. 
    1. Manually checked the images/frames with the highest errors for these categories to see if there is a pattern or to understand what type of pictures for dark skinned individuals or bald people is the model getting wrong. 
        1. Maybe the model is consistently making mistakes where dark skinned individuals are in the background or there is low lighting. 

**Part 2:** Fixes

- The first fix is to improve model by adding failure cases to model training.
    - Identify errors and add these errors to the training set and fine tune. Enabling the model to improve.
    - For errors on dark skinned and bald people, collect missed faces, or faces detected with low confidence (but low depth values - meaning closer) and add them to the fine tuning dataset.
- For errors on dark skinned people, explore lightning conditions for these images/frames.
    - I have researched and read on mutliple instances that YOLO and similar models struggle significantly with low light situations.
    - It would be better to evaluate the model on low light conditions + dark skin.
    - If this is the case, we can experiment with the following:
        - brightness/contrast image augmentations
        - add more low light examples to the training dataset.
- For errors on bald people, the best way is to add more bald people to our training set.
    - But adding more bald people is not enough.
    - We first identify or do an evaluation on what is the model confusing bald people for.
        - For example, a reddit post stated that their YOLO model is confusing bald heads with BASKETBALL.
        - So instead of only adding more bald heads to the training sets, it would help to add more basketballs to the set as well.
        - Since it will allow the model to differentiate better. It will learn more about bald heads, and about basketballs, so as to not confuse these two anymore.
- Hard negative mining helps:
    - Instead of just adding more bald people to our training set, we add things that can be confused as a bald person so that the model learns to distinguishes.
    - For example:
    
    ```
    Bald face missed
          ↓
    Identify confusing visual patterns
          ↓
    Collect hard negatives
          ↓
    Add positives + hard negatives
          ↓
    Fine-tune detector
          ↓
    Evaluate subgroup recall again
    ```
    
    The hard negatives should be based on the actual observed errors rather than chosen randomly. 
    
- Overall we diagnose → collect failure cases → target the actual failure mode → fine-tune → evaluate subgroup recall again.


## Question 1.4:  
Imagine this other case: let's assume we have added several bald people imagery to our dataset and we still are not increasing the accuracy. What may be happening? How would you fix this?

### Answer

This question is similar to the previous one, and the exact fix that was applied to improve bald people accuracy is adding more bald images to the training dataset. 

A few problems can arise by this fix of adding more images: 

- The failure cases for bald people were not properly explored - so the bald images that were added, were not fully representative of failure cases for bald people.
    - The way to confirm and fix this is to first identify high error cases or cases where bald head faces were not detected.
    - This is a case of, “did we add more data or did we add more useful data”.
    - So rather than simply increasing the count of images, we add data that is representative of our failure case.
- Another problem that I have previously encountered when fine-tuning an OCR model was **Bad Labels.**
    - Since this type of detection data (OCR, face detection) is labelled using bounding boxes, there is a chance that the annotator made a mistake and this mistake was learned by the model.
    - So adding an extra layer of quality checks for data annotation will help.
- The problem of bald heads being mostly in background - this is another problem I faced when fine tuning for an OCR for ID cards.
    - I found that the expiry date was mostly at the bottom of the image and sometimes would be cut.
    - So making sure the bald heads are not only found in the backgrounds but should be as main subjects can help as well.
- The problem of overfitting bald heads.
    - Suppose the new images that were added are primarily in the same pose or of the same person, or basically very similar.
    - The model can overfit on this specific type of bald head and fail to generalize to new cases.
    - This can be improved in mainly two ways:
        - more variation in bald pictures. (can add augmentation as well)
        - a training strategy that reduces overfitting
            - regularization like dropout
            - batch norm
            - early stopping
			
			

## Question 1.5: 
Which MLOPs infrastructure would you design to manage this system? Just provide a high-level design and which tools would you use.

### Answer

Video Ingestion: 

- AWS EC2 running a dockerized service to ingest video frames using opencv.
    - the docker container runs the video processing pipeline using opencv and processes the incoming frames.
    - we can store the incoming video on S3 (adds to latency) but this adds storage and potentially processing latency. Another thing to consider is that since this is a privacy use case, it would be better to store only the anonymized frames (or when the re-identification has been done).

Face Detection

- The incoming frames are passed to the OpenCV DNN face detector, using the SSD detector with a ResNet-10 backbone.
    - The detector identifies whether faces are present in the frame and returns a bounding box and confidence score for each detected face.
    - The detector determines where the faces are.
    - Since SSD + ResNet-10 is relatively lightweight, we can initially run it on the GPU using the g4dn.xlarge family of aws machine (T4-16gb GPU), it has both multi-gpu and multi cpu options.
    - To reduce computational cost, we do not necessarily need to run detection on every frame. Detection can be performed periodically while the tracker handles the intermediate frames.
    - the output for this stage is: Face 1 → Bounding box + confidence 0.94
    - a lightweight tracker like SORT will also run on this machine.
    

Re-Identification: 

- **Hosted on:** Separate AWS EC2 instance within the **same VPC**
    - because workloads can become high due to the re-identification after a certain number of frames.
    - We can use a vector DB like aws aurora (pg-vector) to do efficient indexing and search over gallery.

Zip Depth:

- since its fairly lightweight, it can run on the same AWS machine as tracking and detection.

Monitoring: 

- Cloudwatch family
    - cloudwatch logs
    - cloudwatch dashboard
    - cloudwatch alerts and alarms
- We can also use sagemaker monitoring - this wll include drift monitoring and a whole bunch of model related metrics.
- Since we are on aws, its good practice to use everything aws.



## Question 1.6: 
From end user POV, which techniques would you use to improve final solution user experience?

### Answer

I think the biggest perspective needs to be that when the end-user is interacting with the live video feed, the anonymization + blurring should not affect the video quality, so the user should be able to see the video scene clearly.

So in order to minimize the blur damage to the video:

- Blur only the region of the face and use a mask that follows the face rather than a fixed rectangular box.
- Not using a very high blur strength can help as well - the person still feels like human, just with a blurred face.

Another challenge would be blur consistency. The face should be blurred properly across frames (or through time for the end user) and should not face issues like its blurred for 10 seconds and as the person moved away swiftly, the blur is removed and the person can be identified. 

- This would not look good as the end user can now identify that anonymous person and would hurt the goal of the end user.
- This is where periodic re-identification for each track ID mentioned in Q1 comes handy.

As an end user, I would be interested in the option of controlling privacy. 

- Being able to control blur strength would be good.
- Area of interests can be defined in the video feed, where if a person enters those areas, they would be unblurred (for security purposes).

The ability to do some form of correction for the video feed would be really helpful. 

- The end user can notice if the blur/no-blur was incorrect, the ability to flag it would add to the user experience - since now I know that this will be fixed.
- The ability of fix it themselves would be even better.


