# Q1 - Analytical thinking

### Question 1

**How can we improve face detection by adding reidentification to anonymize only specific individuals while keeping computational cost low?**

To improve face detection while keeping computational costs low, I would combine face detection, temporal tracking, and event-driven Re-ID.

### Question 2

**How can we extend an anonymization pipeline with depth-based filtering to blur only background objects efficiently?**

I would combine person/object detection with relative depth estimation to estimate the foreground depth, preserving foreground objects while blurring pixels identified as background. To keep computational cost low, I would avoid running depth estimation on every frame and reuse depth information between frames.

### Question 3

**Imagine we have this case: our face detection system does not detect properly (bad accuracy) dark-skins nor bald-people; how would you improve system accuracy?**

First, I would check precision and recall by slices and intersections, inspect the false negatives, and review the quality and representativeness of the data. Based on the evidence, I would test the appropriate intervention: adjusting the threshold or preprocessing, or if there is a real issue with data coverage, adding representative examples and performing fine-tuning. Finally, I would evaluate both the affected groups and any potential regressions, and validate the results in production.

### Question 4

**Imagine this other case: let's assume we have added several bald people imagery to our dataset and we still are not increasing the accuracy. What may be happening? How would you fix this?**

Adding more images of bald people may not improve the model if baldness isn't actually the cause of the failure or if the new data doesn't represent the problematic production conditions. I would analyze interactions such as baldness + lighting, camera angle, resolution, and face size, as well as review labels and false negatives, to identify the actual cause. Based on that evidence, I would correct the dataset, preprocessing, or model as appropriate and re-evaluate.

### Question 5

**Which MLOPs infrastructure would you design to manage this system? Just provide a high-level design and which tools would you use.**

I would design a pipeline like this: **Data → Train → Evaluate → Register → Deploy → Monitor**. I would use DVC for versioning datasets and annotations, MLflow for experiment tracking and model registry, and CI/CD to automatically run tests and quality gates before deploying a new model.

I would first deploy to staging/canary and monitor system and model metrics in production using tools such as Prometheus and Grafana. Problematic production cases would be reviewed and, once validated and annotated, could feed into future iterations of the dataset and retraining.

### Question 6

**From end user POV, which techniques would you use to improve final solution user experience?**

The user could click on a person or select a face from a list to anonymize it, and a temporary bounding box or highlight might appear. If there is uncertainty about which face to anonymize, a temporary blur could be applied until the person's ID is detected.
