Initial draft of the answers were done using Claude and can be found here:
https://docs.google.com/document/d/1y5jacFhWmig5gfnAydiMScff4FqqVgjbYCTDHHLS3hw/edit?usp=sharing

# Q&A

## 1. 
How can we improve face detection by adding **reidentification** to anonymize only specific individuals while keeping computational cost low?

### Pipeline
We can add a re-identification stage after YOLO-style Face Detection. The pipeline looks like this:
1) Face detection (YOLO): runs every frame, create bounding boxes for faces and provide a confidence score for each crop. If the confidence of the crop containing a face is over a threshold (for example 0.5, but needs to be determined empirically), it goes to the next step.
2) Embedding extraction: produces a compact identity vector per each face crop.
3) Identity matching: compares vector against a gallery. Only in the case in which the most similar vector has a value lower than the threshold (to be determined), the face is passed to the anonymisation stage.
4) Anonymisation: Blurring of faces to anonymise.

Using gstreamer seems is possible to leave to the final user the face choice of detection threshold, blurring intensity and (possibly) also reidentification threshold [https://fluendo.com/blog/face-anonymization-with-gstreamer/]

Optimizations:
- Try half-precision (FP16) inference for the embedding network. Computational performance on memory should improve and to be taken into consideration if paired with negligible accuracy drop.
- It is possible to cache embedding for tracked faces across frames using IoU based tracking (for example using ByteTrack[https://arxiv.org/abs/2110.06864 ByteTrack: Multi-Object Tracking by Associating Every Detection Box])
- Stack all crops into a single batch and run them through the network in one shot, for example the input for a 112x112 MobileFaceNet has NCHW (Number of samples, Channels, Height, Width) shape of [N, 3, 112, 112]. For example, if N=4 it will looks like this:
    [crop_1, crop_2, crop_3, crop_4] → one forward pass → [emb_1, emb_2, emb_3, emb_4]


Models to consider for Embedding Extraction considering on-device scenarios, including github implementations:
1) MobileFaceNet (2018, 1k+ citations paper, https://arxiv.org/pdf/1804.07573)
1.1) https://github.com/foamliu/MobileFaceNet (apache license, 6 issues in total, python 3.6.8, PyTorch 1.3.0) - worth trying but would prefer more recent code and possibly to work with a framework that is more active
2) GhostFaceNet (2023, 182 citations, https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10098610)
2.1) https://github.com/HamadYA/GhostFaceNets (MIT license, repo provided with the original paper, 50+ closed issues, Python 3.9.12, TensorFlow 2.8.0) - more rencent repo, would prefer torch as I have more experience with it
3) Arcface (need to check if is small enough to run on device)
3.1) https://github.com/chenggongliang/arcface

Though not explicitely asked I tried to look for model implementations (including weigths and commercially usable) in Github/Hugging Face. None of the ones I found was really what I hoped to find. The best scenario here would have been to find a framework that allows to try different methods to generate embeddings for both accuracy and efficiency. Would be for example interesting to check whether bigger models are more accurate and if there is a way to distill this information to improve smaller model accuracy.

Later found: MobileFaceNet (via the insightface library)

The models taken into consideration were the ones proposed by Claude, MobileFaceNet for example seems a good candidate as it really seems made for edge devices. Would be also interesting to look into the citations of the paper for newer model or find surveys that compares model. 

### Some literature:
- https://fluendo.com/blog/face-anonymization-with-gstreamer/
    I was able to install gtreamer on my mac, but was not able to install Fluendo plugins as seems only available for Win.
- https://www.reddit.com/r/computervision/comments/1l4dvfm/strategies_for_object_reidentification/
- https://www.mdpi.com/1424-8220/25/5/1410 A Review of Machine Learning and Deep Learning Methods for Person Detection, Tracking and Identification, and Face Recognition with Applications
- https://arxiv.org/abs/2110.06864 ByteTrack: Multi-Object Tracking by Associating Every Detection Box
- https://arxiv.org/pdf/2303.15334 ByteTrackV2: 2D and 3D Multi-Object Tracking by Associating Every Detection Box
- https://link.springer.com/content/pdf/10.1007/s10462-025-11212-y.pdf Multi-object tracking review: retrospective and emerging
trend
- https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/cvi2.70010 A Review of Multi‐Object Tracking in Recent Times

## 2. 
How can we extend an anonymization pipeline with depth-based filtering to blur only background objects efficiently?

### Pipeline
1) Run face detector → get face bounding boxes.
2) Run depth estimator on the full frame (or a downsampled version) → depth map D.
3) For each pixel p not inside any face bounding box: if D[p] > threshold_depth → apply blur kernel; else keep sharp.
4) Composite: blend blurred background with the original foreground region.

### Models
1) Depth Anything V2
2) MiDaS MobileNetV2 (lighter alternative at the cost of some detail in cluttered scenes)
3) https://arxiv.org/html/2504.07093v1 FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution

### Optimizations
- Depth at half resolution: depth maps are spatially smooth; estimating at 320×240 then upsampling with guided/bilateral filter preserves edge sharpness at half the compute.
- Temporal smoothing: (exponential) moving average across frames prevents flickering without lagging.
- ROI (Region of Interest) masking: reduce to zero depth inference inside known face bounding boxes; the network does not need to be precise where a hard mask already exists.
- GPU sharing: depth net and face detector share the same CUDA stream; interleave kernel execution to maximise GPU utilisation.

### Depth Threshold Calibration
- In uncalibrated (relative) depth mode, a fixed percentile (e.g. 'blur the 40% of pixels with the largest depth value') is more robust than a fixed absolute threshold. In metric mode (Depth Anything V2 Metric), a concrete distance (e.g. > 2 m) can be used and exposed as a user-configurable parameter.


## 3. 
Imagine we have this case: our face detection system does not detect properly (bad accuracy) dark-skins nor bald-people; how would you improve system accuracy? 

### Root Cause Diagnosis
#### Dark skin tones
- CNN face detectors trained predominantly on lighter-skinned datasets (e.g. WIDER FACE, IJB-C) learn texture and highlight patterns that generalise poorly to darker complexions — low reflectance reduces the contrast cues the detector relies on.
#### Bald individuals
- Hair is a powerful low-level cue many detectors implicitly use for head localisation; its absence degrades bounding-box recall, especially in tilted or partial views.

### Evaluation Protocol
- Always report per-demographic metrics (precision, recall, F1, false-negative rate) segmented by Fitzpatrick skin-tone scale and hair presence. 
- A single aggregated mAP (Mean Average Precision) can hide significant disparity. 
- [literature] https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

### Tools - one of these should be enough
- Custom confusion matrix stratification script
- Amazon SageMaker Clarify (if using AWS Sagemaker as integrates natively into SageMaker Pipelines for automated fairness gating on each training run)
- Fairness Indicators (TFX - Tensorflow)
- Aequitas.


### Data-Side Interventions
#### Targeted dataset augmentation
- Target class-balance: aim for rough parity per demographic in training batches to prevent majority-class dominance.
#### Harvest balanced supplementary data from public corpora
- Look for dataset that improve diversity towards the least represented classes, for example: DivFace, Diversity in Faces (DiF), MegaFace subsets.
- For bald subjects: crawl public domain imagery of bald celebrities, medical datasets, sports footage (swimmers, martial arts) [not sure whether this option is legally feasible]
#### Photometric augmentation
- Random gamma correction (γ ∈ [0.4, 1.5]) simulates lighting variation across skin tones.
- Colour jitter with wider saturation/hue ranges improves dark-tone generalisation.
- Hair removal simulation via image inpainting (e.g. LaMa model) applied stochastically during training generates synthetic bald variants from existing data.
#### Hard negative mining
Log all false negatives per demographic during validation to inform the next data collection round.

---
Aggregation might look like this:

- Dark skin:  42 false negatives out of 150 samples → 28% miss rate
- Light skin:  8 false negatives out of 400 samples →  2% miss rate
- Bald:       35 false negatives out of 90 samples  → 39% miss rate

---

---
The flow might look like this:

1. Train on training set
2. Evaluate on validation set → log false negatives per demographic
3. Use that log to decide: "we need more dark-skin + bald samples"
4. Collect/augment → add to training set
5. Retrain → re-evaluate on validation set
6. Repeat until satisfied
7. Final evaluation on test set (once, at the end)

---

### Model-Side Interventions
- Uses loss function resistant to unbalanced datasets: Train with focal loss or class-balanced cross-entropy.
- Multi-scale detection heads: bald faces tend to fail at intermediate scales; face detector uses hair as a boundary cue helping the network localise where the face ends. The network struggles to decide where the "face region" is, especially at intermediate scales (32–96 px). Too small and the face is just a blob anyway. Large enough and there are enough other features to compensate. The intermediate range is the sweet spot where hair absence hurts the most. The fix is to verify that the anchor sizes or training image crops explicitly include faces in the 32–96 px range in sufficient quantity, so the related detection head gets enough gradient signal to learn bald-face detection at that scale.

### Literature
- https://www.mozillafoundation.org/en/blog/facial-recognition-bias/
- https://arxiv.org/html/2509.10980v2 TrueSkin: Towards Fair and Accurate Skin Tone Recognition and Generation
- https://arxiv.org/pdf/2009.07838 FairFace Challenge at ECCV 2020: Analyzing Bias in Face Recognition

## 4. 
Imagine this other case: let's assume we have added several bald people imagery to our dataset and we still are not increasing the accuracy. What may be happening? How would you fix this?

Adding data is necessary but not sufficient. Persistent poor accuracy despite augmentation points to several failure modes:

### Most Likely Root Causes
#### Label noise / annotation inconsistency
- If bald-head crops from the new imagery were partially mislabelled, or annotators drew bounding boxes inconsistently (too tight, cutting off the crown), the detector learns a corrupted signal. The network capacity is wasted correcting conflicting gradients.
- Fix: audit the new subset with a second annotator or automated label-quality tool (e.g. CleanLab, Cleanlab Studio). Re-annotate any samples with IoU < 0.7 between annotators.
#### Distribution mismatch — wrong kind of data
- If the added bald imagery comes from a different domain (e.g. studio portraits at high resolution when the target deployment is CCTV at 480p), the model overfits to the new domain without generalising to production conditions.
- Fix: collect or synthesise data that matches the deployment distribution — same camera type, resolution, compression artefacts, lighting conditions, viewpoint angles.
#### Class imbalance still too large
- If the overall dataset is 90 % non-bald after augmentation, batch sampling will still present predominantly non-bald examples. The gradient contribution from the new minority samples is too diluted.
- Fix: use stratified / oversampling (e.g. WeightedRandomSampler in PyTorch) to ensure the bald subset appears at least 15–20 % of each batch, or apply class-weighted loss.
#### Learning rate or fine-tuning strategy mismatch
- If the existing model was fine-tuned with a learning rate too high, catastrophic forgetting wipes out previously learned features. Too low, and the new data barely shifts the weights.
- Fix: use a discriminative learning rate — freeze the backbone for 1–2 epochs, fine-tune only the head; then unfreeze with a 10×–100× lower LR on the backbone.
#### Evaluation data leakage or wrong test set
- If the test set was accidentally contaminated with training samples, or if it does not actually contain difficult bald/dark-skin cases, metrics will look flat or misleadingly good.
- Fix: re-split data with strict subject-level (not frame-level) separation. Ensure the validation and test sets contain a demographically representative sample of hard cases.

### Recommended Debugging Workflow
- Run inference on the new bald-subject subset alone — log per-image recall to isolate which bald types still fail.
- Visualise attention maps (GradCAM) on failing samples — identify which features the model is using (or ignoring).
Check training loss curves split by demographic subset — if the bald-subset loss is not decreasing, the problem is optimisation, not data volume.
- Re-run CleanLab confidence-weighted label quality analysis on the new annotations.


## 5. 
Which MLOPs infrastructure would you design to manage this system? Just provide a high-level design and which tools would you use.

Below is a high-level infrastructure design covering the full lifecycle: data → training → evaluation → deployment → monitoring. The design favours open-source tooling aligned with an on-premise / edge-first production context like Fluendo's.

### Data Management

- Storage: AWS S3: raw frames, augmented crops, model artefacts
- Versioning Code: Git
- Versioning Data: DVC (Data Version Control) to tracks dataset lineage
- [optional] Labelling: Label Studio self-hosted, supports bounding box + demographic tagging
- [optional] Quality: CleanLab for automated label noise detection, integrated as a pre-training step


### Experiment Tracking & Training

- Tracking: MLflow to logs params, metrics, artefacts per run; model registry built in
- Training jobs: PyTorch [optional - Pytorch Lightning]; containerised in Docker, orchestrated with Kubernetes Jobs
- [optional] PyTorch DDP for multi-GPU; Horovod for multi-node if needed
Config mgmt
- [optional] Hydra: hierarchical config composition for sweep experiments


### Evaluation & Fairness
- Automated per-demographic evaluation: on every PR merge (CI gate) — fails if recall drops > 2 % on any Fitzpatrick (skin type) scale bucket.
- Fairness dashboard: MLflow using Fairness Indicators integration.
- Regression test suite: fixed hard-negative clips (dark-skin, bald, occluded, low-light) run on every model version.

### Model Packaging & Deployment
- Registry: MLflow Model Registry with stage gates: Staging → Production
- Export: ONNX using TensorRT (for NVIDIA edge/server) or OpenVINO (for Intel edge)
- Serving: Triton Inference Server — multi-model, dynamic batching, built-in metrics
- GStreamer pluging: Custom GStreamer element wraps Triton client, integrating natively into Fluendo's Raven AI Engine pipeline


### Monitoring & Drift Detection
- Frame-level inference metrics (latency P50/P99, throughput, GPU util) exported via Prometheus + Grafana.
- Data drift: Evidently AI — monitors input distribution (face crop brightness histogram, aspect ratio) vs. training baseline.
- Model drift alert: if rolling false-negative rate > threshold over a 24 h window, trigger re-training pipeline automatically.

### CI/CD Pipeline
- Code push → GitHub Actions: lint, unit tests, integration tests.
- Dataset version pinned in DVC tag → training job submitted.
- MLflow compares new run vs. registered champion: auto-promotes if all fairness gates pass.
- Docker image built → pushed to registry → Triton deployment updated via Helm chart.
- Smoke test on staging stream → promote to production.


## 6. 
From end user POV, which techniques would you use to improve final solution user experience? 

Face anonymisation is a background system component. Users experience it indirectly through configurability, latency and quality. 

### Operator Configuration & Transparency
1) Per-stream configuration UI: users can select anonymisation mode (blur, pixelate, mask, inpaint) and set depth/ReID thresholds per camera zone without redeploying. These parameters can be exposed via the gstreamer interface.
2) Debug View: display detection confidence and depth scores overlaid on the video, toggled per channel, to help users tune thresholds without guessing. The parameter to enable/disable debug view can be exposed via the gstreamer interface.
3) Audit log & anonymisation map: for compliance, record timestamps and face counts per frame in a side-channel metadata stream.

### Performance & Reliability
1) Allow user to choose which detection model to use. Use YOLOv8m by default.
2) Graceful degradation: if GPU (or CPU) utilisation exceeds 90 %, fall back to a lighter detection model (e.g. YOLOv8n instead of YOLOv8m) rather than dropping frames; notify the user via dashboard.
3) Cold-start caching: warm up model weights on service start, not on first frame; eliminates the visible spike of unprocessed frames at stream initialisation.

### Privacy UX for End Subjects
1) Insertion API: in scenarios where subjects opt in (e.g. office analytics), provide a user-friendly system to provide consent and take photos/videos that will be used as enrollment material.
2) Deletion API: allow subjects to request removal of their gallery;

### Anonymisation Quality & Naturalness
1) Inpainting-based anonymisation: replace faces with a plausible synthetic texture (using a lightweight GAN or diffusion inpainting model) rather than a blur rectangle. Produces a more natural-looking scene for broadcast applications.
2) Smooth temporal transitions: blend anonymisation masks across frames using IoU-tracked IDs; eliminates the flickering that distracts viewers when detection confidence fluctuates.


## 7. (EXTRA) Provide a code snippet (practical example) on how would you implement any of these applications: reidentification or depth estimation, for face anonymization.