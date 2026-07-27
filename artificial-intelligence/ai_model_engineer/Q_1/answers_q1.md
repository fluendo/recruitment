# Face Anonymization Test

---

#### Q1. How can we improve face detection by adding reidentification to anonymize only specific individuals while keeping computational cost low?

The objective here is not only to detect faces but to anonymize selected individuals by adding a re-identification step. This means the detector must be paired with a re-identification model that decides whether a face corresponds to a certain target identity.

As stated within the assumptions, the lighter the models, the better, and face detection is run on images and video. It is not defined whether the system corresponds to image or video scenario. Anyway, the first straightforward step, as the detector is already built, would be to build the re-identification module around our target identities.

The ReID module will learn to extract features, so that after a face is detected, its features can be extracted and compared to the ones we have in our gallery so the correct ID is assigned to it. Data from the scenario we are working in, taking into account the physical and spatial conditions is assumed to be accessible. With that, we could build a suitable dataset for a ReID model training, splitting it into training identities, and query and gallery ones. With that set up, the selected backbone could be trained. Following, the selection is discussed.

In order to determine which face-recognition backbone could fit into our task for the ReID stage, taking into account the cost-accuracy trade-off, I would use MobileFaceNet because of its efficience and design for real-time in our scenario ([https://doi.org/10.48550/arXiv.1804.07573](https://doi.org/10.48550/arXiv.1804.07573)). After some research, it could be a suitable model given its efficiency and low-cost setup, as it is a variant of the MobileNet, which is widely known for these efficiency characteristics. 

While one of the most expanded options for the loss selection is the triplet loss (and cross-entropy for classification loss), in this case this architecture uses another kind of loss, which is called ArcFace Loss. In another ReID problems I have worked on, I used the first one, so I was not aware of this specific face-recognition one. As it is our-domain explicit and proven to work, it would be selected for this concrete task.

At this point we would have the detector plus the ReID model. The full implementation depends on which scenario are we facing. I will assume a video-based one, as costs are stated to be crucial. In this case, I would add a lightweight tracker that maintains face tracks across frames. A good practical option for our real-world case would be ByteTrack. 

We expect our cost to slightly increase even though we selected lightweight options for this new ReID step. However, if we take into account that ByteTrack could manage the IDs across frames, maybe the ReID model would just have to be run once a new tracklet is initialized (or for its 5 first frames to be sure the ID is correct), or if we want to ensure the correctness every N frames to re-confirm the ID, not on each frame after detection. This could reduce the cost at a low theoretical risk.

With that new step introduced, target individuals could be identified and so that anonymized correctly without a large increase in the computational cost.

---



#### Q2. How can we extend an anonymization pipeline with depth-based filtering to blur only background objects efficiently?

In aim of extending the pipeline with depth-based filtering so foreground and background content are differentiated and then that last one blurred, we must evaluate which kind of scenario are we in. There are depth estimation pre-trained models which produce a per-pixel depth map frame by frame. With that information we could decide establishing a threshold if a pixel corresponds to the foreground or to the background. 

However, in this case we focus on faces, and so that on people. That is the reason why a person segmentation model could be run to differentiate them from background objects. With this approach, individuals would be kept sharp while background would be blurred. Theoretically, these kind of models are computationally cheaper than the depth-estimation ones, so our latency would not be compromised at all with them. Besides that, we could make it efficient by merging the mask of the segmentation model with the face detector.

It is the simpler and cheaper approach. However, it is true that crowded scenes or scenarios where segmentation models struggle could compromise a person detection/segmentation and so that compromising its anonymization. Depending on which scenario are we in, this approach could work or we would have to go for a depth-estimation model fine-tuning together with efficiency measures so the cost is not compromised.

Regarding this last one approach, the cheapest while effective architecture would be a FastDepth, which is a mobile encoder-decoder. Also architectures heavier (but als sharper) as MiDaS could be considered. However, if we take into account all the modules involved in the system, a FastDepth with a bit of post-processing could solve the task stated. Among these operations, if the result is not the expected but not far from it, I would suggest to add temporal EMA (Exponential Moving Average) to correct possible flicker in video sequence, or morphological operations to correct individuals boundaries.

To end up with this question, if a depth-estimation module is added, it could be used to define ROIs, and so that aiming to detect/re-identify just in these stated regions.

---



###### Q3. Imagine we have this case: our face detection system does not detect properly (bad accuracy) dark-skins nor bald-people; how would you improve system accuracy?

This case could be derived from many possibilities related to our data. Firstly, there could be a lack of representative training samples for these collectives. If not, then features could rely too havily in specific visual aspects that indirectly harm the predictions for bald and dark-skin people. We assume here the root cause to be unrepresented training data for these classes. Given that, our goal would be to rebalance the dataset distribution withou discarding useful data.

In order to achieve our goal, I would rather oversampling the minority classes so we don't risk the model's performance across the rest of classes nor throw away useful samples, as mentioned above. To do so effectively, I would propose using a generative model so synthetic samples are added to our dataset. Instead of generating them from scratch, we could use an image to image pipeline so random current samples of our dataset would be passed through a set of modifications (as removing hair or darkening their skin) so real-world face structures and environment conditions as lightening are kept.

The selected approach has associated costs. However, it does not increase the cost for the real-time system as belongs to the detector training pipelina. That is the reason why it has been selected although introducing additional computation. If there were an alternative constraint in this way, other non-generative options could be studied in order to oversample these minority classes.

---



#### Q4. Imagine this other case: let's assume we have added several bald people imagery to our dataset and we still are not increasing the accuracy. What may be happening? How would you fix this?

If several bald people images have been added and the model still predicts it wrongly, we must firstly inspect and analyze the classification errors for that specific class in order to extract some pattern among the imagery that could help us detecting the root cause. 

Apart from the quality of the images, and assuming they have been correctly labelled, if they are very similar between them, or do not include variety regarding gender, ethnics, or even age, the model could be memorizing them and not generalizing as expected. Domain-specific characteristics as lightning must be similar to other classes, so we ensure the model is not learning from the environment but from the individual.

In case data looks correct and we carried out the analysis without extracting any significant conclusion, then we could adjust the classification loss. We could try to use the focal loss, which is proven to work with difficult classes or hard-to-classificate samples. If not, then adding a higher weight for the bald-people class in our current loss could also work, as we would be forcing it to penalize in a heavier way errors in this class.

---



#### Q5. Which MLOPs infrastructure would you design to manage this system? Just provide a high-level design and which tools would you use.

I would design a MLOPs infrastructure divided in the following layers to ensure traceability and monitoring:

- Data Layer: To support raw video/images ingestion from cameras or user uploads. Also providing with dataset versioning for train/val/test sets.

Tools: AWS S3 to store raw images/video frames, CVAT to manage annotations and DVC for data versioning.

- Training Layer and Model Registry: In this layer is where reproducible pipelines for data preprocessing, training and evaluations live in. Besides, experiment tracking for metrics, hyperparameters and artifacts are also included in this layer. Models with a minimum performance would be stored trained and compiled (in .pt for instance), and automated tests to check latency should be run.

Tools: Docker to run training jobs in containers, MLFlow for experiment tracking and model registry, and CI/CD with Git for automatic deployment.

- Inference Layer: Here is where real-time service for video streams and batch pipeline for offline processing is managed. Every model (ReID, detector, tracking) should have a separated service. GPU inference must be supported here.

Tools: An inference engine, for instance Triton Inference Server and an API layer built with FastAPI.

- Monitoring: Information for latency, GPU usage or dropped frames would appear here, together with ML monitoring metrics as average confidence scores, anonymization miss rate, false anonymization rate... Also distribution shifts in input conditions must be monitored to detect data drift and prevent our model/s from performance degradation.

Tools: Grafana for monitoring dashboards and an appropiate data drift and quality tracking tool.

---



#### Q6. From end user POV, which techniques would you use to improve final solution user experience?

A final user would like to configure easily and understand what is happening in our system. That is why I would apply the following three improvements for a final product. I assume a user-friendly interface and the improvements do not regard any question of design, which I understand would also be crucial for the user to interact with the final solution:

- Let the user configure which kind of anonymization prefers among the stated: blur, pixelation or masking. Also I would add an option to configure if he/she wants to anonymize all the faces or just selected identities.
- The user needs the video to be smooth, so that I would apply temporal filters so the blurring (or masking or pixelling) does not flicker frame to frame.
- I would show clear feedback on what was anonymized and why. I would add some kind of preview mode where detection shows what would be anonymized in a selected video/image. Like a before/after preview. Besides that, I would not silent missing faces (both preview or not) but marking them as uncertainty zone.

