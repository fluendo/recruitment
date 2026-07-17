# Question #1

## Analytical thinking

### Problem description:

With the enforcement of the new EU AI Act, AI systems that process biometric data including face detection and recognition are now subject to stricter transparency, risk management, and privacy-preserving requirements. In many real-world video analytics scenarios (smart cities, retail, sports analytics, workplace safety, and media processing), camera streams may capture identifiable faces even when identity is not required for the system’s core purpose. Under the new regulatory framework, this creates legal and operational risk unless appropriate safeguards are applied.

To address these constraints, we have designed a **face anonymization** solution based on a CNN-based face detector (YOLO-style architecture) to locate faces in images and video streams and automatically applies anonymization filters (blur, pixelation, or masking). The objective is to preserve scene utility while preventing personal identification, supporting privacy-by-design and privacy-by-default principles required by the EU AI Act.

However, deploying such a system in production raises several technical challenges: selective anonymization (only specific identities), fairness and bias in detection accuracy across demographics, robustness under difficult visual conditions, computational efficiency for real-time processing, and proper MLOps governance. This test explores how to analyze, diagnose, and improve such a Deep Learning–based anonymization system from both model and infrastructure perspectives.

<div style="text-align: center;">artificial-intelligence/ai_model_engineer/Q_1/README.md
  <img src="../imagery/anonym_1.png" alt="Plate 1" width="60%" style="display: inline-block; margin-right: 2%;">
</div>

In this test, we will analyze real problems and dilemas related with this product in order to evaluate how would you act and reason in front of them.

### Assumptions:
- Face detection is CNN-based and runs on images and video
- The system must comply with EU AI Act privacy and risk-control expectations
- There is no constraint in which model architecture can be used for both reidentification & depth estimation
- We want to reduce computational cost so the lighter the models, the better

### Questions:

Please provide a document (and extra data if any) answering each question below; please note that no practical exercises are required here (just theoretical analysis) but feel free to add any.

#### 1. How can we improve face detection by adding **reidentification** to anonymize only specific individuals while keeping computational cost low?

- Reidentification is useful for situations in which certain faces need to be blurred, while others need to be kept unchanged. The computational cost increase of introducing ReID processing for each single frame in the video stream has the potential of making the solution not suited for real-time usage.

- One viable approach to this issue is looking to reduce the frequency at which the ReID model runs inference on the video stream, while running face detection on all frames. This approach implies the need for a tracking algorithm, as the anonymized face must be able to move through the frame without triggering another expensive ReID inference. In the case of H.264 or H.265 video streams, we can take advantage of the Group of Pictures (GOP) structure to track the movement of each face through the frame, information which the decoder modules produces already at no extra computational cost through the Motion Vectors. In the case of a real-time stream where face detection is running on every frame:

- In the first frame of the stream, which would be an I (Index) the face detection model detects all faces present in the frame. The ReID model identifies them, compares them to the face embeddings database and informs the anonymizer to blur only the desired face(s).

- The next frame can either be a P (Predictive) or an I frame:
  - If it is a P, we retrieve the motion vectors around the areas that we know had a face in the previous frame and find out where all the faces have moved in the frame, successfully tracking the new position of all faces, anonymized and non-anonymized, and allowing for the anonymizer to keep blurring the correct face without having to trigger another ReID. If a face is detected in a place where we don't expect the current faces to be, a selective ReID would be triggered only on the area where the new face is detected, decreasing the cost compared to processing the entire frame. Additionally, for complicated scenarios, like very shaky or noisy images, a "traditional" fast tracking algorithm like Intersection over Union (IoU) may be added in tandem to improve the system's tracking robustness.

  - If the frame is I, this can be for two reasons: a radical change in the image or a regular periodic I frame, which can be known by a very light Bounding Box Overlap check. If the faces in this I frame are closely positioned enough to the ones in the last P frame, we can safely assume that the scene has not changed much.
    1. The I frame was triggered by a radical change in the stream -> the ReID model must be triggered again to safely identify all possible new faces.
    2. The I frame was triggered by the regular GOP structure but there has not been a major change -> The frame can be treated like the P frame case, where present faces are easily tracked and new detected ones trigger a selective ReID run.

- In a real-time scenario, a typical GOP contains an I frame every 15 to 60 frames. The proposed approach reduces the workload of the ReID algorithm to 6.7% to 1.7% compared to a naive frame-by-frame ReID approach, even lower if the I frames are determined to not come from a major scene change, like it is often the case in workplace safety situations or smart cities.

- Face alignment would be an issue when detecting new faces, so the pipeline should incorporate an alignment algorithm before the reidintification to improve the accuracy of the ReID model.

- This implementation is feasible through a GStreamer pipeline by modifying `gst-libav` to enable the AV_CODEC_FLAG2_EXPORT_MVS flag on the decoder context. This allows us to receive the motion vectors calculated during decoding at no extra cost.

#### 2. How can we extend an anonymization pipeline with depth-based filtering to blur only background objects efficiently?

- The challenge of discriminating foreground vs background with only one camera can be solved with a Monocular Depth Estimation (MDE) CNN model, which is able to generate a 3D depth map from a static image. For this system, a GStreamer pipeline is proposed to handle the depth map obtained by the MDE model and apply selective gaussian blur past a certain depth threshold.

- Much like Q1.1, computational cost is non-negligible. For this reason, the MDE model should not run on every single frame at full resolution, but around every 5 to 10 frames instead, depending on the specific application, and stabilized with a temporal filter that averages the current depth map with 20% of the previous one. Also, a heavy downsampling of the video stream may be necessary for feeding the MDE model, as large resolution streams would most likely not allow real-time processing. After obtaining a low-res depth map, it must be upscaled with a Guided Filter, as to preserve boundary accuracy. With this, a solution supported in the extraction of the Motion Vectors during the decoding process would allow us to track and update the depth map every frame with enough accuracy in between inferences.

- The blurring can be performed by the GStreamer element `gleffects_blur` in the processing pipeline. A completely blurred image would be generated each frame. Then, the depth map obtained by the MDE would act as a normalized alpha mask, being used as a blending coefficient between the completely blurred frame and the original frame. In this manner, every part of the image would be blurred progressively as the object is more distant.

  - For each object (or face) detected, the blurring applied to each pixel of it can be calculated with:

$$I_{\text{final}} = \alpha(d) \cdot I_{\text{og}} + (1 - \alpha(d)) \cdot I_{\text{blur}}$$

  - Where:

    - I_final is the resulting image,
    - I_og is the original, unblurred image,
    - I_blur is the fully blurred image,
    - α(d) is a function of the depth extracted from the depth map, normalized between 1 for the areas not to blur (foreground) and 0 for the areas of maximum blur (absolute background).

- This α function would be biased to be <1 only when the depth detected of a certain pixel exceeds the threshold of the barrier between foreground and background. Potentially, this α function could be manipulated to emulate the behavior of a real camera's focus, behaving, for example, like a surge function: 

$$α(d) = \left( \frac{e \cdot d}{d_0} \right) e^{-\frac{d}{d_0}}$$

Where:
- α(d) is the blending coefficient, ranging from 0 to 1,
- d is the distance of the pixel,
- d_0 is the "focus" distance, of non-blurring.

#### 3. Imagine we have this case: our face detection system does not detect properly (bad accuracy) dark-skins nor bald-people; how would you improve system accuracy?

- This is a known issue on face detection systems, and directly relates to the Algorithmic Fairness point of the EU AI Act. First, we must understand the likely causes for this issue:

  - Light skin people represent the majority of the data available to train these systems. For this reason, when face detection CNNs are trained, a well performing model overall can hide bad performance on detecting faces of specific, less prevalent ethnicities in the data.
  - Considering that generally only men experience baldness at a rough 45% rate of adults, a representative dataset of the world's population would have a presence of 0.5*0.45= 0.225 (22,5%) of bald people. This is a fairly high percentage, yet, added to the possible error of good general performance but poor specific class performance described earlier to the fact that the absence of hair removes one of the most common visual "anchors" a CNN may depend on, detection may struggle unless compensated.
  - Due to dynamic range limitations on digital camera sensors and standard processing, CNNs often struggle picking up the facial features and nuances of dark-skinned individuals, especially in low light or high contrast situations. This makes it harder for the detection model to accurately pinpoint the facial features of the person, hurting performance.

- The proposed approach to improve the system's accuracy attacks the problem in two ways: The question of data quality and the question of majority class bias for loss functions:

  - More data needs to be generated for those people who are not as represented in the population, but who all equally deserve to have systems adapted to them. For this, data augmentation techniques must be performed, modifying the images' contrast, histogram equalization, low-lights correction, etc. The idea is that all demographics shall be equally represented in the dataset, and in a manner that trains the model to detect the faces of all demographics equally. Specifically to improve the performance on bald faces, data augmentation procedures should generate oclusions in the top part of the face, to help the model not learn the idea of "dark on the top of the head <=> face" as a hard correlation.

  - The training approach must be that of a multi-class detection task; it is necessary to divide the detections between races, gender and possibly hair/bald. Even if the final application does not require the model to distinguish these classes, it is necessary from a validation standpoint. The reason is that the standard Cross-Entropy loss function treats all errors as equally important, while in real life, these two demographics present specific challenges that must be compensated for. The Focal Loss function is a very good solution for this because it allows us to down-weight the importance of easy examples for the loss calculation, effectively forcing the model to give more relative importance to those classes with which it may struggle more.

- It is worth noting that even after data augmentation techniques have helped us produce a fully equally representative training dataset, we must acknowledge that bald people's features may still make it harder for an algorithm to detect them in the real world, as well as digital camera's limitations that may cause dark-skinned people's faces to be harder to detect, which is why the second solution for training the CNN is needed.

- To further improve accuracy, a light video processing previous to the detection, aimed towards rising the darker tones and reducing the contrast in the shades would likely improve the system's capacity to detect dark skin people's faces.

#### 4. Imagine this other case: let's assume we have added several bald people imagery to our dataset and we still are not increasing the accuracy. What may be happening? How would you fix this?

- If this issue is not a purely data quantity/quality issue or a training bias issue for the case of bald people, we must consider what other things might the CNN be learning during training rounds that make it hard for it to later detect bald faces.

- CNNs are said to act "lazily" in certain situations. This is, they may learn to bias their decision based on the path of less resistance, which is not necessarily the most precise ([DropBlock: A regularization method for convolutional networks](https://arxiv.org/abs/1810.12890)) This would mean, in this case, that the face detection CNN has made the "dark on the top of the head <=> face" association described in Q1.3 way too profoundly, making impossible for it to detect a face when that is not the case. For this reason, it would be the time to modify the training dataset to prove this specific hypothesis. Generating a dataset in which we intentionally drop (erase) contiguous regions of the face, especially the top part, will teach the model to not make any lazy assumptions that it may be making during runtime.

- Another area to cover during any data science process would be the image characteristics of those bald people's faces that we are introducing; they must be akin to the real world scenarios that the model is going to find in production. This means, that if the CNN is going to be deployed in a low resolution CCTV system, the images must be from, or else at least emulate, those CCTV images. This is an aspect that is both fairly easy to tackle and of critical importance for the success of the model.

<maybe talk about cnn bias towards texture>

#### 5. Which MLOPs infrastructure would you design to manage this system? Just provide a high-level design and which tools would you use.

- For a complex pipeline of sequential ML models like this one, the correct approach would be to follow CI/CD/CT practices. CI and CD correspond the known DevOps practices of Continuous Integration and Continuous Deployment. Added to these two, CT stands for Continuous Training: In this practice, an autonomous retraining pipeline is set up, and can be triggered automatically by the addition of new data to the training set or by detecting that the model has experienced accuracy degradation through automatized performance indicators.

  - In a face detection pipeline, indicators of performance degradation could mean tracking the progress of the confidence score of the face detections over time. The ReID model's match rate statistics could provide relevant information, for example if the unmatched detections experiences an abnormally high (possible false positives), or low (possible false negatives) number. Lastly, it is worth considering detecting changes to the captured video stream, for example, through comparing periodically the color histogram distribution over the image during certain hours. The logic behind this analysis is that the detection conditions vary throughout the year, as in summer there is more lihgt hours and more contrasted images, but in winter we may deal with much more face occlusions, reduced light hours and lower contrast during cloudy conditions.
  - To detect false positives/negatices and correct them autonomously, a Student-Teacher infrastructure would be a useful tool. In this, a second, heavier and extremely precise model would be deployed to run on certain samples of the production video stream or samples flagged as low confidence detections by the lighter deployed model. This "teacher" model would run in non real time and correct the deployed model's possible mistakes; if the the "student" model flags a frame as no face, but the "teacher" one detects a face with 95%, we have succesfully detected a false negative and would add the frame to the CT pipeline.

- In order to correctly track and manage this automatic training pipeline, we would build a DVC pipeline with all the necesary steps for training without human intervention; data preparation, training and objective evaluation metrics must be perfectly set up so they are trustable without continuous supervision. Then, the training triggering logic can be set up by CI tools like GitHub Actions. After retraining, the pipeline can be designed to launch a Pull Request if the resulting model presents better objective metrics than the currently deployed one. Although it is possible for the new model to be deployed autonomously, utilizing the PR as a safegurad point makes most sense if the system is to be used in critical applications.

#### 6. From end user POV, which techniques would you use to improve final solution user experience? 

- The application on which the system runs should expose certain parameters that allow for the operator to adapt the system's behavior in real-time. One reason would be that, while our system performs well overall, certain use cases or even specific days might require different parameters in, for example, ReID inference frequency, bounding box overlap criteria for tracking during specially slow or fast moving camera/objects, or blur control capabilities, making the strength of the blurring filter and the threshold depth a tweakable parameter. It is important to provide a tailored solution to the final client; some may require us to reduce the amount of training that the operator requires to utilize the system if they are non-technical, like security guards, while other clients might desire as much control as possible for their staff, like TV control rooms. The application must be adapted, whithin certain limits, to the final client.

- Some cameras provide the information of focus distance embedded within the video stream metadata, as is the case for professional cinema and modern TV camera units. This information can potentially be extracted to dinamically adjust the blurring threshold (The distance where the detected objects must start to appear blurred). Effectively, this would mean that this system would work seamlessly on dynamic, moving subject/camera situations without being limited to a fixed distance threshold. Some protocols that expose this capacity are the ARRI Lens Data System or the SMPTE RDD 18, which encodes these metadata within the video SDI signal.

- In applications where the aesthetic integrity of the video stream is a requisite, ReID can be coupled with a synthetic face overlay model, making the anonimyzed identity display a synthetically generated face instead of a blurred area. This makes improves the viewer's user experience and improves how seamless the solution is to the final user.

#### 7. (EXTRA) Provide a code snippet (practical example) on how would you implement any of these applications: reidentification or depth estimation, for face anonymization.

[Code Snippet for ReID pipeline with MV extraction for object tracking (Question 1.1)](q_1_7.py)