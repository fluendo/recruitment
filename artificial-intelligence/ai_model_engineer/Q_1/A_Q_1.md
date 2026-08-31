
### Questions:

Please provide a document (and extra data if any) answering each question below; please note that no practical exercises are required here (just theoretical analysis) but feel free to add any.

# 1. How can we improve face detection by adding **reidentification** to anonymize only specific individuals while keeping computational cost low?
I would add two different systems.

The first one would be able to detect specific features, for example freckles or moles. The first time we identify a person, we could save these specific features and use them the next time we need to re-identify the person. These features could also be stored in a vector database to make the search faster.

The second system would be for short-term re-identification. In this case, we could re-identify a person using a combination of general visual features, such as skin tone, hairstyle and clothes.

I think combining both systems could help us reduce the computational cost, because we would not need to perform the most expensive re-identification process on every frame.
 
# 2. How can we extend an anonymization pipeline with depth-based filtering to blur only background objects efficiently?
I would add a depth estimation model to the pipeline.

The model would tell us which parts of the image are closer or further from the camera. Then, after detecting the faces, we could use this information to know if they are in the foreground or background.

For example, if a face is in the background, we apply the blur. If it is in the foreground, we leave it unchanged.

To reduce the computational cost, we could use a lightweight depth model and run it only every few frames in a video.

# 3. Imagine we have this case: our face detection system does not detect properly (bad accuracy) dark-skins nor bald-people; how would you improve system accuracy? 
First, I would check the training dataset to see if there are enough examples of bald people and people with darker skin tones.

If there are not enough examples, I would add more representative images and retrain the model.

If, for some reason, I cannot add more data, I would give more importance to these examples during training/(Oversampling for example), while being careful not to overdo it and cause overfitting.

I would also check if the problem could be related to other factors, such as lighting or image quality, because these could also affect the detection accuracy.
# 4. Imagine this other case: let's assume we have added several bald people imagery to our dataset and we still are not increasing the accuracy. What may be happening? How would you fix this? 
As I said before could be related to other factors such as ligthing or image quality. 

It could also be that the new images are not representative of the situations where the model is failing. For example, the model could work well with bald people in good lighting but fail in low-light situations.

I would first analyze the images where the model is failing to find the real cause of the problem. Then, depending on the results, I would improve the image quality, add more representative examples, or adjust the training process.

Overfitting could also be a problem, so I would compare the training and validation results to check if the model is learning the training images too specifically.

# 5. Which MLOPs infrastructure would you design to manage this system? Just provide a high-level design and which tools would you use.
Github/Gitlab for version controling
CI/CD for testing
Docker for conteniration
Qdrant as vector DB, for reidentification
Postress as relational DB for general system information 
MLflow could be used to save training parameters and results. I have not used MLflow but I think it could be a good option for tracking.
Object storage for storing images and videos.



# 6. From end user POV, which techniques would you use to improve final solution user experience? 
Before create the final video, would be great to see a preview. Not to be creating once and again the final video.
As user I would prefer not to need technical Know about the system to know how to use it. So the app must be very simple and easy to use.


# 7. (EXTRA) Provide a code snippet (practical example) on how would you implement any of these applications: reidentification or depth estimation, for face anonymization.



## 7. Re-identification - Practical Example

```python
import face_recognition

# Face already identified
known_image = face_recognition.load_image_file("known_person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# New detected face
new_image = face_recognition.load_image_file("new_face.jpg")
new_encoding = face_recognition.face_encodings(new_image)[0]

# Compare both faces
match = face_recognition.compare_faces(
    [known_encoding],
    new_encoding,
    tolerance=0.5
)

if match[0]:
    print("Same person -> anonymize")
else:
    print("Unknown person -> do not anonymize")

```
The first time a person is identified, I would save their face encoding. When a new face is detected, I would generate a new encoding and compare it with the stored one. If they are similar enough, the system identifies the person and applies the anonymization.




Note

I used AI assistance to improve the grammar, wording and structure of my answers. However, the ideas, reasoning and proposed solutions are my own, and I am fully responsible for the content and decisions presented in this document.