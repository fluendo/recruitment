1. How can we improve face detection by adding **reidentification** to anonymize only specific individuals while keeping computational cost low?

> One concern with reidentification is that it is strictly regulated by the EU AI Act. Since EU AI Act classification depends on actual use, I would first make sure of the legal implications in each case acconting for the fact that the final purpose of the application is anonimization. We already have a YOLO-like model doing the face detection. Now we need a model that gives a label to each detected face. We can use a CNN with an architecture that does not end with a classifier, but with an embedding layer that outputs a vector. Then this embedding is compared against a library of previously computed embeddings, and decide if can be attributed to the same subject based on Euclidean distance or some other similarity metric. But with the custom-made CNN and embedding we will need to train it first. Alternatively we can use readily available CNN embeddings with pretrained weights for face recognition. Running the face identifier network at every frame in addition to YOLO would significantly increase cost. We can instead run the identifier network only every n frames, n being defined according to some rule like pixels moved per frame, or losing track of detection that we are tracking with IoU.
  
2. How can we extend an anonymization pipeline with depth-based filtering to blur only background objects efficiently?

> Adding monocular depth estimation (MDE) to the previous pipeline. Only when Re-ID model is called (criterion defined before) we apply MDE. If MDE determines distance below threshold (close), no identification needed. Otherwise, run re-ID embedding and follow pipeline defined before. This allows the economy of deploying MDE only every n frames and the economy of not running the Re-ID embedding when the detected box is not far. If the MDE is relatively expensive compared to our pipeline we can also consider running it only once every x times the Re-ID is deployed.
 
3. Imagine we have this case: our face detection system does not detect properly (bad accuracy) dark-skins nor bald-people; how would you improve system accuracy? 

> I would try adding samples of those groups to the database to address a possible lack of representability. This has the inconvenient of having to train our model with a newly enriched dataset instead of just using available pretrained weights.

4. Imagine this other case: let's assume we have added several bald people imagery to our dataset and we still are not increasing the accuracy. What may be happening? How would you fix this? 

> Now we can affirm that bald people being underrepresented in the dataset is not the cause for the bad accuracy. The problem may be that this group of individuals is more difficult to be detected. Possible explanation: the shape and contrast of hair in non-bald people makes it easier for the model to encode their features in a reduced space representation. Another possibility is label bias because some labelers systematically mistake bald people with some other object. I would explore first all possibilities from the training, annotation and data side, and then go to more costly and less likely causes like model architecture. Example for model architecture: Hyperparameter tuning using Optuna with this group accuracy as objective function (then check if other groups accuracy degraded too much).

5. Which MLOPs infrastructure would you design to manage this system? Just provide a high-level design and which tools would you use.

> Git and Github for code versioning. DVC for database versioning. CI/CD consisting in small k-tests every time a developer performs a commit. Docker for a containerized environment. Performance monitoring: automated measurement of KPIs: accuracy, recall per group (fairness), framerate.

6. From end user POV, which techniques would you use to improve final solution user experience? 

> Add user adjustable parameters to control detection sensitivity, depth threshold, maximum number of individuals to be blurred. Add warnings when reidentification confidence is low. Export a summary of statistics: number of reidentifications, reidentifications per minute, average in-frame time, average frames between reidentifications, etc. This can improve model transparency in case it is asked by authorities to final users or developers. 

7. (EXTRA) Provide a code snippet (practical example) on how would you implement any of these applications: reidentification or depth estimation, for face anonymization.