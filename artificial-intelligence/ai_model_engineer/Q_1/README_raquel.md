# QUESTION 1
**How can we improve face detection by adding **reidentification** to anonymize only specific individuals while keeping computational cost low?**

To do selective anonymization, we need to solve 2 challenges: where a face is and if that person is on the list of inidividuals that we can show unblurred. We cannot run an identity model on every detected face in every frame because it is computationally expensive and we would break the EU AI Act, as we would process biometric data of all the people. 

The approach we should consider is: 
1. Detect the faces with the YOLO architecture, already used, on every frame. 
2. Multi-object tracking to assign a unique ID to each face detected across frames, it will use IoU, no need of a neural network. One ID across frames. 
3. Embedding network (such as MobileFaceNet or InceptionResnetV1) that takes each cropped face with its ID and outputs a fixed embedding vector
4. Apply a cosine similarity against the list of embedding people who authorized to appear unblurred. If the similarity matches above a threshold, the face is clear, if not we blur it. If we have an unknowed face we blur it. 

As we have used a tracking object per each face detected we don't need to pass to the embedding network each face detected per frame just one per track, so we save energy here. Also the list of embedding people who authorized to appear unblurred is relatively small compared to all the people we will compare, is a matrix-vector multiplication. Lastly, feeding to the network cropped images is cheap than passing the whole frame. 


# QUESTION 2
**How can we extend an anonymization pipeline with depth-based filtering to blur only background objects efficiently?** 

In order to blur based on depth, we need to add a depth estimation model. This models looks the image and predicts for each pixel how far it is from the camera. The output is a depth map in 2D, a new image, where the close objects are with one color and the far objects with another one, to know which is foreground and which is background.  

As the depth of an object does not change super quickly from frame to frame we don't need to apply the depth map in each frame, we can do it every 3-5 frames and reusing the one computed in the last map. Making the cost cheaper. Also only apply it inside the bounding box detected in YOLO not in all the frame image. 

The pipeline will be:
1. First face detection as before using YOLO.
2. Depth estimation: using a small monocular depth model. Choosing small versions so that it is not very expensive to apply it in real time video
3. Sample the depth for each face: for each face box, look at the depth map inside that box and take the median depth value. We should use the median beacuse is robust against weird pixels. We will have one number per face telling us how far that face is from the camera. 
4. Apply a depth threshold. If the face is beyond a threshold we blur, if is closer we keep the image clear. 

# QUESTION 3
**Imagine we have this case: our face detection system does not detect properly (bad accuracy) dark-skins nor bald-people; how would you improve system accuracy?**

This is a bias problem of the data we have in the training set, where these kind of people represent a low minority compared to all the data we have of white skins and people with hair. In order to solve this, we need to have more representative data of all ethnicities, hairs....

We need to make sure that the data that we have now, in the training, validation and test dataset is stratisfied, that we have different people in all the datasets. Moreover, we have to obtain or produce more images of the minor groups, using more datasets to have an equal amount of people from different ethnicities or with GANs creating more images of the minor groups, before training the actual network. Also with the images that we have, add a variations of ligth or color. 

When training, we can also use focal loss so that the model learns which types of faces are more difficult to learn and put more weight to that class and less weight to the easy ones. 

# QUESTION 4
**Imagine this other case: let's assume we have added several bald people imagery to our dataset and we still are not increasing the accuracy. What may be happening? How would you fix this?**

We need to take into account several things when adding new data. We need to take a look at how this new data is labeled. It needs to be with the same bounding boxes as we had for the rest of the dataset, including the whole picture of the face, not cutting for example the forehead. Maybe we also need to rerun anchor k-means on a balanced subset. This would be an option because at the begining the model is mostly trained with hair faces and it perhaps the bounding boxes were typically tall rectangle and with bald people need to be shorter vertically. Re-run k-means will make that the anchors have information of all the different types of faces. We can also fine-tune freezing the backbone and lower the learning rate so that the backbone updates slowly to the new faces.

Also look for the amount of data we have for the bald people compare to the whole dataset. If the amount is low, we will need to rebalance the data to make sure the model sees all the different faces. As I mentioned before, also make sure that all sets have a representation of all the different kinds of faces. 


# QUESTION #5
**Which MLOPs infrastructure would you design to manage this system? Just provide a high-level design and which tools would you use.**

For a system like this, the most important aspects are that everything is versioned, traceable, gradual and reversible. 

I would use GitHub to host all the code and that all team members can have an access the code easily, saving versions and different forks.

For orchestration I would use Apache Airflow. The pipline of this problem follows clearly a DAG structre. Airflow handles this kind of structure allowing to retry processes if they fail and has an understandable user interface.

For tracking experiments I would use Weights and Biases as it provides a very good interface and allows a vast majority of graphics inside it. Also is easy to share the project across the team members. The second option for tracking would be MLflow. 

Therefore, the workflow would be: code lives in Github and triggers CI which builds container images and pushed them to a registry. Airflow schedules and runs training pipelines using those images. During training, every run logs metrics, hyperparameters, interesting metrics plots and artifacts to Weights & Biases. Each model is saved with a version tag. 


# QUESTION #6
**From end user POV, which techniques would you use to improve final solution user experience?**

We can make the blur look good, I would add padding around the face to soften the edges so that the final result is not a sharp rectangle. We could also provide different types of anonmyzation not only blur, also pixelation for example. Also make sure that if there is an error that the whole frame appears blured not showing the face just in case. Lastly, we could also deploy like a real time small image of the result of the final blured image so that the user can see how the final result will be an rely on the product.


# QUESTION #7
**(EXTRA) Provide a code snippet (practical example) on how would you implement any of these applications: reidentification or depth estimation, for face anonymization.**

The question_7.py file implements the steps I described at question question1. 
