# Question #1

## Analytical thinking

### Problem description:

With the enforcement of the new EU AI Act, AI systems that process biometric data including face detection and recognition are now subject to stricter transparency, risk management, and privacy-preserving requirements. In many real-world video analytics scenarios (smart cities, retail, sports analytics, workplace safety, and media processing), camera streams may capture identifiable faces even when identity is not required for the system’s core purpose. Under the new regulatory framework, this creates legal and operational risk unless appropriate safeguards are applied.

To address these constraints, we have designed a **face anonymization** solution based on a CNN-based face detector (YOLO-style architecture) to locate faces in images and video streams and automatically applies anonymization filters (blur, pixelation, or masking). The objective is to preserve scene utility while preventing personal identification, supporting privacy-by-design and privacy-by-default principles required by the EU AI Act.

However, deploying such a system in production raises several technical challenges: selective anonymization (only specific identities), fairness and bias in detection accuracy across demographics, robustness under difficult visual conditions, computational efficiency for real-time processing, and proper MLOps governance. This test explores how to analyze, diagnose, and improve such a Deep Learning–based anonymization system from both model and infrastructure perspectives.

<div style="text-align: center;">
  <img src="../imagery/anonym_1.png" alt="Plate 1" width="60%" style="display: inline-block; margin-right: 2%;">
</div>

In this test, we will analyze real problems and dilemmas related with this product in order to evaluate how would you act and reason in front of them.

### Assumptions:
- Face detection is CNN-based and runs on images and video
- The system must comply with EU AI Act privacy and risk-control expectations
- There is no constraint in which model architecture can be used for both reidentification & depth estimation
- We want to reduce computational cost so the lighter the models, the better

### Questions:

Please provide a document (and extra data if any) answering each question below; please note that no practical exercises are required here (just theoretical analysis) but feel free to add any.

1. How can we improve face detection by adding **reidentification** to anonymize only specific individuals while keeping computational cost low?

Processing video data efficiently poses a significant challenge because computational complexity escalates rapidly due to the huge volume of frames processed every second. Running a heavy deep learning pipeline on every single frame can easily create a performance bottleneck. That said, video data contains also massive temporal redundancy; consecutive frames look pretty similar because human subjects do not drastically change their position or appearance in a matter of milliseconds.

Therefore, to keep computational costs low while incorporating face re-identification for selective anonymization, we could split the process into two distinct stages. 
- Stage 1: Continuous Detection and Tracking (Every frame)
A lightweight face detector (such as YOLO) will run continuously on every frame to locate faces and output bounding box coordinates. This is paired with a robust multi-object tracker that maintains identity continuity across consecutive frames. The tracker relies on motion and appearance models to assign and carry persistent IDs smoothly, preventing the need for continuous inference.
- Stage 2: Selective Re-Identification (On demand)
Face Re-ID feature extraction is decoupled from the continuous loop and will only be triggered selectively. For example, when working with compressed video streams, the system could use I-frames and P-frames to decide when to run Re-ID:
  - On P-Frames (Predictive Frames): Because P-frames use motion data to track how objects move from the previous frame, the system can smoothly follow existing faces without running expensive Re-ID every time.
  - On I-Frames (Index Frames): I-frames represent a "new" picture or a scene change. Therefore, when an I-frame appears, the system will trigger the Re-ID model to safely verify or register possible new faces.
As an extra precaution against ID drift, the system can also run Re-ID every "x" number of frames or during random checks.

Finally, to make identity verification fast and scalable, targeted faces are pre-processed and stored in a local face database as compact numerical embedding vectors. When a verification check is triggered during the Re-ID stage, the system extracts the current face crop's embedding and compares it against the database using some efficient similarity metrics, such as the cosine distance. This mathematical comparison enables instant face identification while keeping GPU load low.

2. How can we extend an anonymization pipeline with depth-based filtering to blur only background objects efficiently?

In contrast to stereo camera setups, monocular estimation works with any standard single RGB camera, which significantly reduces hardware costs and setup complexity. Moreover, it avoids the complex triangulation calculations required by stereo systems, making the pipeline simpler and faster.

Once the depth map is computed, the system could apply a specific depth threshold: any face detected beyond that distance boundary (the background) will be smoothly blurred out, while foreground subjects and faces will remain crisp and clear.

In orther to imporve the pipeline and make it more efficient, we could apply several optimization strategies such as:
- Input Resolution Downscaling: Depth and detection models do not require ultra high resolutions to accurately figure out spatial layers or locate faces. By downscaling input video frames before feeding them into the depth estimation network, we could imporve our inference speed without compromising background/foreground split quality.
- Tracking Algorithms: Just like with face Re-ID, running depth estimation on every single frame would increase the complexity of our model. To keep performance high, we could tie depth calculations directly into our object tracking workflow. Instead of generating a new depth map constantly, the system could compute them selectively; such as on video stream I-frames, when a new subject enters the scene, or during periodic random checks.

3. Imagine we have this case: our face detection system does not detect properly (bad accuracy) dark-skins nor bald-people; how would you improve system accuracy? 

To address poor accuracy for specific demographics like dark skinned or bald individuals, my approach would focus on three main areas:

- Dataset Rebalancing: My first step would be to inspect the training dataset. Minority groups, such as dark skin or bald individuals, are often underrepresented in standard public datasets. Therefore, it is not surprising that models develop a natural bias if not treated carefully. If this imbalance is the root cause, I would rebalance the dataset to ensure these groups have fair representation during training. For instance, we could use data augmentation:
  - For dark skinned individuals, this would include adjustments to lighting, contrast, and exposure to ensure the detector handles various skin tones and different lighting environments.
  - For bald individuals, we could use synthetic data generation or targeted augmentations. This will force the model to focus on core facial geometry rather than relying mostly on hair or hairline cues.
- Balanced Metrics and Loss Function: finally, I would adjust how the model is optimized and evaluated. Relying on standard accuracy can be misleading because a model can achieve a high score while completely failing the minority class. I would switch to balanced accuracy metrics and implement weighted loss functions (like focal loss) to heavily penalize misclassifications on underrepresented groups, forcing the model to pay equal attention to all demographics.

4. Imagine this other case: let's assume we have added several bald people imagery to our dataset and we still are not increasing the accuracy. What may be happening? How would you fix this? 

If adding more bald imagery to the dataset doesn't improve accuracy, it indicates that the bottleneck is not just data quantity; it is due to the model architecture or how it learns. Here is how I would diagnose and fix the issue:
- Check Model Capacity: The model might be structurally unsuited for the task. It could be too simple to capture the necessary invariant features, or too complex, leading to poor generalization. I would experiment with alternative backbone architectures to see if capacity is the limiting factor.
- Inspect Learned Features: To figure out what features the model is actually relying on, I would use saliency maps. These visualization tools highlight the exact pixels that drive the model's predictions. By combining them with targeted testing on specific data subsets, we can easily spot if the model is taking shortcuts (such as relying on hairlines or forehead shapes rather than true facial geometry).
  - If we identify that the model is leaning on a shortcut, we can actively modify the training inputs. For example, we could temporarily mask, blur, or augment the hairline region across all training images. This forces the model to ignore hair entirely and instead learn other facial geometry like bone structure and eye-nose-mouth spacing.
- Specialized Demographic Models: If a single global model struggles to handle diverse physical traits simultaneously, we could train specific models for different types od data. To merge them into a unified system, we could use:
  - A voting system which combines predictions from multiple models.
  - A gating network: a router classifier that first identifies general facial traits and dynamically routes the frame to the most appropriate specialized expert model. This will increase the complexity of the whole pipeline.

5. Which MLOPs infrastructure would you design to manage this system? Just provide a high-level design and which tools would you use.

- Version Control and automation - Git CI/CD: will be used to manage tracking code, pipeline scripts, and configuration files. Every push or pull request will automatically trigger a CI/CD pipeline to run code linting and validation checks.
- DVC and Amazon S3: datasets, annotations, and model checkpoints versions could be controlled via .dvc metadata files committed directly to Git. The actual heavy data files (such as training datasets and video snippets) could be pushed and pulled from an S3 object storage bucket, ensuring full data reproducibility across team members and training runs.
- Scalable model training - Cloud GPU instances such as EC2: model training, fine-tuning, and handling data augmentation scripts are delegated to scalable cloud GPU nodes. Training jobs are triggered automatically or manually from Git pipelines.
- Packaging and deployment - Docker: everything will be packaged into a self-contained Docker container. This guarantees that the software environment on local development machines matches the production edge hardware identically, preventing runtime discrepancies.

6. From end user POV, which techniques would you use to improve final solution user experience? 

Since the same system might be deployed for users with slightly different privacy goals, it is important to expose a few tunable parameters so each user can customize the behavior. Some of those parameters could include:
- Adjustable anonymization intensity: sers can easily control the strength of the privacy effect, as well as choose between different styles of anonymization, such as traditional pixelation, heavy blurring, or gaussian noise.
- Flexible operational modes: provide used a simple way to lets them switch between different policies; such as anonymizing unknown individuals while keeping whitelisted subjects visible, or applying anonymization to everyone.
- Simple face whitelisting: giving users a straightforward dashboard or interface to quickly add a new authorized face to the system without needing data engineering skills.

7. (EXTRA) Provide a code snippet (practical example) on how would you implement any of these applications: reidentification or depth estimation, for face anonymization.