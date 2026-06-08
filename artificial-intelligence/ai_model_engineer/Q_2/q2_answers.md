https://docs.google.com/document/d/1HjMPjtX_qnp0ngmgXnLHpO5A5q3BM7nfuX7qK5z9rRk/edit?usp=sharing

# Q&A

## 1
The model shows strong performance on validation data but fails on new matches recorded in different clubs with different camera heights and angles. Hit detection accuracy drops significantly.
<br>Explain the most likely root causes of this failure and describe how you would systematically investigate and isolate the problem.

The most likely root causes of the failure is the difference between camera positions, clubs/players diversity to the data in training/validation. An investigation can be carried out to define what is the impact of each. Depending on the investigation, the priority of ad hoc mitigation strategies can be selected.

### Root Cause Analysis
#### Camera Viewpoint & Geometry Shift
Pose estimation models are sensitive to the projection geometry of a scene. A change in camera height, tilt, or horizontal angle alters:
- The apparent foreshortening of limbs (a raised arm from a high-angle camera looks shorter than from a low one).
- The visibility of joints — a top-down camera may occlude one shoulder behind the torso, never seen during training.
- The 2D skeleton aspect ratio — a player running toward the camera shows a very different keypoint distribution than one running laterally.

Models trained on clips from a single-club installation effectively learn a camera-specific prior. When that prior is violated, keypoint localisation degrades, and downstream hit classification — which relies on wrist/elbow trajectory — collapses.

#### Background & Illumination Distribution Shift
Different clubs have different court colours (green, blue, terracotta), wall textures, and glass transparency. If player segmentation or appearance-based keypoint detectors rely implicitly on contrast vs. the background, new backgrounds break them.
Indoor vs. outdoor lighting conditions (dynamic sunlight, shadows, lens flare) create appearance distributions never seen during training.

#### Player Appearance Diversity
Skin tone, clothing colour, and kit type affect any model that has not been trained on a sufficiently diverse corpus.
If training data came from one club, player height and body proportion distribution may be narrower than deployment.

#### Hit Classification Temporal Shift
Even if pose is localised correctly, the hit classifier was tuned on clips from specific cameras. Frame-rate, latency, and the temporal window over which wrist velocity is computed must match the training distribution. A camera mounted 2 m higher will produce earlier peak wrist acceleration in screen coordinates for the same physical swing.

### Systematic Investigation Protocol
#### Step 1 — Instrumented Evaluation
- Collect 50–100 labelled clips from each new venue (at minimum: one match per camera configuration).
- Run evaluation separately per venue, per camera angle bucket (e.g. high / medium / low) and per hit type.
- Report keypoint OKS (Object Keypoints Similarity) and hit-classification accuracy broken down by these strata — not as a single aggregate.
- [literature] https://learnopencv.com/object-keypoint-similarity/
- Strata in this case: 1) Per venue / club, 2) Per camera angle bucket (high / medium / low), 3) Per hit type (forehand, backhand, smash, volley)

#### Step 2 — Failure Mode Isolation
- Visualise predicted keypoints on failing clips. Does the pose look reasonable but the hit label is wrong (classifier issue), or are keypoints already wrong (detector issue)?
- Run the pose detector with ground-truth bounding boxes — removes any bbox detection errors from the analysis.
- Check per-joint failure: wrist and elbow errors on new-angle footage confirm viewpoint sensitivity, not general model failure.

#### Step 3 — Feature Distribution Analysis
- Project training and new-venue keypoint sequences into a 2D embedding (PCA or UMAP) — visual confirmation of distribution shift.
- Compute per-joint coordinate statistics (mean, std, skewness) for training vs. failing venues — statistically significant deviation pinpoints which joints are affected.

### Mitigation Strategy
Recommended fixes by root cause
- __Viewpoint shift:__
Camera-normalisation pre-processing (homography alignment to canonical view) + viewpoint augmentation during fine-tuning
- __Background shift:__ Domain-adaptive training with style-transfer augmentation (AdaIN or RandAugment) on court texture/colour
- __Temporal hit shift:__ Re-calibrate hit-event timing model per camera with 15–20 labelled hits from each new venue (lightweight fine-tuning)
- __General OOD data:__ Collect ≥500 clips from diverse venues and add them to the training set with camera metadata as a conditioning feature
- __Priority order:__ keypoint localisation must be fixed before hit classification — errors compound through the pipeline.



## 2
After reviewing some samples, you discover that different annotators labeled keypoints slightly differently (especially wrists and elbows), and hit timestamps are sometimes shifted by a few frames.
<br>Analyze how inconsistent annotations affect both pose estimation and hit classification, how you would detect label-quality problems at scale and how you would decide whether to relabel, filter, or reweight parts of the dataset. Justify your decision criteria.

Label noise in pose estimation and temporal event detection is particularly damaging because errors propagate through multiple stages: keypoint localisation → skeleton-based feature extraction → hit classification.

### How Inconsistent Annotations Affect the System
#### Wrist & Elbow Keypoint Jitter
- Wrists and elbows are the highest-velocity joints during a hit and carry the most diagnostic signal. A few-pixel labelling inconsistency between annotators introduces artificial noise into the joint trajectory. The model learns a blurred, uncertain representation of these joints — exactly where precision matters most. During inference, wrist keypoints become jittery, which directly degrades velocity and acceleration features fed to the hit classifier.

#### Timestamp Shift on Hit Events
- A hit event spans roughly 3–8 frames at 25–30 fps. If annotators disagree on which frame constitutes the "contact" frame by ±2–3 frames, the model receives contradictory supervision: the same wrist-velocity profile is sometimes labelled as a hit and sometimes not. This creates a noisy decision boundary for the classifier, manifesting as inconsistent hit timing at inference — exactly the behaviour observed.

### Detecting Label-Quality Problems at Scale
#### Keypoint consistency checks
- Inter-annotator agreement (IAA) on a shared held-out subset — compute per-joint PCK (Percentage of Correct Keypoints) between annotator pairs. Flag joints with IAA < 90 % as unreliable.
- Temporal smoothness filter: fit a low-pass filter to per-annotator joint trajectories — annotated positions with high-frequency noise above a biomechanical threshold (wrists can't accelerate faster than ~50 px/frame at 30 fps) are likely annotation errors.
- Cross-annotator consistency score per clip: average per-joint distance between two independent annotations. Clips with high variance are candidates for relabelling.

#### Hit timestamp consistency checks
- If multiple annotators labelled the same clip, compute the temporal spread of hit annotations. Clips with spread > 2 frames on a 30 fps stream are ambiguous.
- Confidence-weighted label smoothing: use the temporal spread as a proxy for label uncertainty; apply Gaussian-smoothed one-hot targets around the annotated frame rather than hard labels.
- CleanLab (for the classification head): run the trained model on training data and flag samples where the model's predicted probability disagrees strongly with the label — a reliable signal of label noise even without a second annotator.

### Decision Framework: 
#### Relabel 
- __When to use__ High-value, rare, or hard-to-collect samples (smashes, unusual angles)
- __Cost__ High — requires expert annotators and coordination time
- __Risk__ Low if done with annotation guidelines; medium if rushed
- __Tool / method__ Label Studio with consensus protocol and annotator calibration session
#### Filter/Remove
- __When to use__ Clips with fundamental ambiguity or irrecoverable annotation errors
- __Cost__ Zero — immediate but reduces dataset size
- __Risk__ Data loss — especially dangerous for rare classes
- __Tool / method__ IAA threshold < 80 % OR temporal spread > 3 frames → drop
#### Reweight
- __When to use__ Large dataset; label noise is spread uniformly, not systematic
- __Cost__ Low — algorithmic, no human time
- __Risk__ If noise is systematic, reweighting can amplify it
- __Tool / method__ CleanLab confidence weighting or simple per-sample loss weighting

### My Recommended Decision Criteria
1) Compute per-clip IAA score for both keypoints and hit timestamps.
2) Tier 1 (IAA < 70 % OR timestamp spread > 3 frames): relabel if it belongs to a rare class (smash, volley), otherwise filter.
3) Tier 2 (70–85 % IAA): keep but apply label smoothing / confidence reweighting.
4) Tier 3 (> 85 % IAA): keep as-is.

A brief calibration session with all annotators before the relabelling pass — showing examples of correct wrist vs. elbow placement and defining the contact frame with a video reference — is the cheapest way to prevent the problem recurring.



## 3
Smashes are rare in the dataset but very important for the product. The model often misses them while performing well on common hits.
<br>Without changing the model architecture, explain which dataset, training, and evaluation strategy changes you would consider. Here, the idea is to discuss tradeoffs and risks of each approach and how you would verify that improvements are real and not overfitting artifacts.

Smashes are safety-critical for the product (missed smash = missed key event) but rare in natural match footage. The following changes span data strategy, training strategy, and evaluation design — each with explicit tradeoffs.

### Dataset Strategies
#### Targeted Data Collection
The most robust fix. Curate clips specifically containing smash events: broadcast footage, club archives, existing video libraries tagged with smash metadata. Even 200–300 additional smash clips can meaningfully shift the class boundary.
Risk: collection cost and annotation time. Mitigate by prioritising short clips around known smash timestamps.

#### Temporal Augmentation of Existing Smash Clips
Time-stretch augmentation: vary playback speed ±20 % to produce wrist-trajectory variants without losing structural validity.
Horizontal flip: valid for padel symmetry — produces a mirrored-side smash at zero cost.
Frame-jitter augmentation: randomly shift the labelled hit timestamp ±1 frame during training (within the biomechanically plausible range) to reduce timestamp-specific overfitting.
Risk: augmented samples are not independent — they share the underlying motion. Use a conservative multiplier (2–4×) and validate on unaugmented held-out smash clips only.

#### Synthetic Data via Pose Interpolation / Motion Mixing
Interpolate between existing smash keypoint sequences (linear or SLERP on joint angles) to generate new synthetic skeleton sequences. No video required — only the pose representation.
Risk: synthetic skeletons may not correspond to valid biomechanics if interpolation is too aggressive. Constrain by biomechanical joint-angle limits and verify with a sports science reviewer.

### Training Strategies
#### Class-Weighted Loss
Assign a per-class weight inversely proportional to class frequency. For smash: weight = N_total / (N_classes × N_smash). This costs nothing in compute and is the first thing to try.
Risk: if smash is extremely rare (< 0.5 % of frames), very high weights can destabilise training. Cap at 10× and monitor validation loss carefully.

#### Focal Loss
Focal loss down-weights well-classified easy examples and focuses gradient on hard ones. Particularly effective when the majority class (no-hit / forehand) is both frequent and easy to classify.
Hyperparameter γ ∈ [1.5, 3.0] — tune on a held-out smash validation set, not the main validation set.

#### Oversampling (SMOTE-style for sequences or simple repetition)
Repeat smash training samples in the batch-sampler to achieve a target ratio (e.g. 1:5 smash-to-forehand). Simpler than loss reweighting and often comparably effective.
Risk: if the same 50 smash clips appear 10× per epoch, the model memorises them. Combine with augmentation to avoid this.

### Evaluation Strategy — Verifying Real Improvement vs. Overfitting
#### Stratified held-out test set
Maintain a completely separate test set with a minimum of 50 smash clips from different venues and players, never touched during training or validation. This is the ground truth for any improvement claim.

#### Smash-specific evaluation metrics
- __Smash recall:__ Primary metric — the product cannot miss smashes. Target ≥ 90 %
- __Smash precision:__ Secondary — tolerate some false positives (annoying but non-critical). Target ≥ 70 %
- __F2 score:__ F-beta with β=2 explicitly weights recall over precision — use as the optimisation target
- __Temporal IoU:__ For detected smash events: fraction of predicted window overlapping ground-truth window. Captures timing quality, not just detection
- __Confusion matrix:__ Full per-class CM — verify improvement is not at the expense of forehand/backhand accuracy

#### Overfitting detection checks
- Compare metrics on augmented validation (same clips as training, augmented differently) vs. held-out test set — large gap signals memorisation.
- Learning curve: plot smash recall vs. epoch. If it plateaus quickly and the gap between train and val recall is > 15 %, training is overfitting.
- Zero-shot venue test: evaluate on one held-out venue with zero smash samples in training — tests generalisation, not memorisation.


## 4
Two training runs with the same configuration produce noticeably different results. One model is stable but less accurate; another is more accurate but produces jittery keypoints and inconsistent hit timing.
<br>Provide a reasoning to decide which model is actually better for production. What additional tests, metrics, and scenario-based evaluations would you design to support the decision?

This is a classic stability-accuracy tradeoff in temporal models. The right choice depends on the product's tolerance for each failure mode. Here is a structured reasoning framework and a test battery to support the decision.

### Characterising the Two Failure Modes
#### Model A (Stable)
- __Keypoint behaviour:__ Smooth trajectories, lower per-frame noise
- __Hit timing:__ Consistent temporal events
- __Downstream impact:__ Reliable feature extraction for classifier
- __User experience risk:__ Slightly under-detects subtle hits
- __Cause hypothesis:__ Higher regularisation, lower LR, or more temporal smoothing
#### Model B (Accurate)
- __Keypoint behaviour:__ Higher per-frame accuracy but jittery — temporal inconsistency
- __Hit timing:__ Hit timestamps shift between runs / frames
- __Downstream impact:__ Velocity/acceleration features are noisy → classifier degraded
- __User experience risk:__ Erratic events — feels unreliable to end users
- __Cause hypothesis:__ Lower regularisation, or no temporal consistency loss

### Additional Tests & Metrics to Design
#### Temporal Consistency Metrics
- __Joint velocity smoothness:__ compute the mean absolute second derivative of each joint trajectory. Lower = smoother. This is independent of accuracy and directly measures jitter.
- __OKS temporal consistency:__ for frames t and t+1, compute the difference in OKS score. High variance = jittery localisation even if average OKS is good.
- __Hit timestamp variance:__ label the same 30 clips with each model 5 times (or across multiple inference passes if stochasticity exists). Report standard deviation of predicted hit frame.

#### Downstream Pipeline Stress Tests
- Feed each model's keypoint output into the hit classifier and measure classifier accuracy. Model B may win on raw keypoint OKS but lose on downstream hit F1 — which is what actually matters.
- Compute wrist and elbow velocity features derived from each model's output. A model with jittery wrists will produce velocity spikes that the classifier interprets as phantom hits (false positives).

#### Scenario-Based Evaluation Suite
Evaluation scenarios
- __Fast smash:__ Wrist moves 80–120 px/frame. Jitter is amplified — does Model B produce false split-second hits?
- __Slow net play:__ Minimal wrist motion. Stable model should hold; jittery model may hallucinate movement.
- __Occluded wrist:__ Player partially behind post. Model B may flicker; Model A may hold a smoother interpolated position.
- __Long rally:__ 5+ minutes of continuous play. Measure hit count drift (Model B may accumulate more false positives over time).
- __Multi-player frame:__ Both players visible. Jitter risk is doubled — evaluate per-player keypoint consistency.

#### Post-Processing as a Diagnostic
Apply a lightweight Kalman filter or exponential moving average (α = 0.5) to Model B's keypoints and re-evaluate. If Model B + smoothing matches or exceeds Model A + smoothing, the instability is in the output layer only and can be post-processed away. If accuracy also degrades, the jitter reflects deeper representational noise.

### Decision Framework
Production deployment should choose Model A unless all three conditions hold:
- Model B's keypoint OKS advantage is > 3 points on the hardest scenario (smash, fast play).
- Model B + Kalman smoothing reduces hit timestamp variance to within 1 frame of Model A.
- Model B's downstream hit classifier F2 score is at least equal to Model A's.

If any condition fails, Model A is safer: a slightly less accurate but temporally stable model produces more reliable user-facing events, is easier to debug in production, and will generate fewer support tickets. Accuracy can be recovered through data; user trust lost to erratic behaviour is harder to restore.

In practice, the ideal outcome is to diagnose why Model B is jittery and apply a temporal consistency loss (e.g. OKS loss on adjacent frames, motion smoothness regularisation) to a third run — capturing the accuracy of B with the stability of A.


## 5
Which model would you propose for this application? Why?

Given the system constraints — edge / real-time deployment, GStreamer pipeline integration, compute efficiency, and domain-specific requirements (fast wrist/elbow motion, occlusion, multi-person) — here is the recommended stack.

### Primary Recommendation: YOLOv8-Pose (nano or small)
YOLOv8-Pose at a glance
- __Architecture:__ Single-stage anchor-free detector + keypoint regression head (17 COCO keypoints, extensible)
- __Parameters:__ YOLOv8n-pose: ~3 M params; YOLOv8s-pose: ~11 M params
- __Speed:__ ~180 FPS on RTX 3060 (nano); ~90 FPS (small) at 640×640 — comfortably real-time at 25–30 fps
- __Export:__ Native ONNX and TensorRT export — direct integration into Triton / Raven AI Engine
- __Multi-person:__ Full multi-person support in a single pass — handles 2 players + spectators natively
- __Training:__ Well-documented fine-tuning on custom keypoint sets; easy to add domain-specific joints (racket hand, centre of mass)

### Why Not the Alternatives
#### ViTPose (Base/Large)
- __Pros:__ SOTA accuracy on COCO; strong on occluded joints
- __Cons for this use case:__ 180 M+ params; >10× slower than YOLOv8-pose; violates efficiency constraint
#### MediaPipe Pose (Base/Large)
- __Pros:__ Ultra-light (< 5 M); works on mobile CPU
- __Cons for this use case:__ Single-person only; poor accuracy on fast sports motion; no easy fine-tuning
#### HRNet-W32 (Base/Large)
- __Pros:__ High OKS on benchmark datasets
- __Cons for this use case:__ Heavy (29 M), slow (needs top-down detection step), hard to export to TensorRT cleanly
#### OpenPose (Base/Large)
- __Pros:__ Mature, well-documented
- __Cons for this use case:__ Slow inference (bottom-up PAF approach); not suitable for real-time edge deployment
#### RTMPose (small) (Base/Large)
- __Pros:__ Purpose-built for real-time, ~4 M params, excellent speed-accuracy
- __Cons for this use case:__ Less ecosystem tooling than YOLOv8; newer, less battle-tested in production
#### RTMPose-s
- RTMPose (from MMPose / OpenMMLab) is purpose-designed for real-time multi-person pose estimation and outperforms YOLOv8-pose on OKS benchmarks at similar parameter counts. If the team has PyTorch / MMDeploy familiarity, it is a strong alternative. For a team already using Ultralytics YOLO infrastructure (likely, given the face detector in Q1), YOLOv8-pose reduces integration friction significantly. 


### Domain Adaptation: What to Fine-tune
- Add padel-specific keypoints: racket-hand wrist (dominant), grip point, shoulder rotation axis — these are the primary hit-discriminating joints and are not reliably captured by standard COCO 17-joint skeleton.
- Fine-tune on labelled padel clips (mixed clubs, camera heights) with camera viewpoint as a metadata label. Even 2 000–5 000 labelled frames produce meaningful domain adaptation.
- Use SimCC (Simulated Coordinate Classifier) head instead of heatmap regression for the keypoints most sensitive to fast motion (wrist, elbow) — better sub-pixel accuracy and faster inference than Gaussian heatmaps.

### Production Architecture
Recommended deployment stack to integrate with Fluendo's existing infrastructure:
- YOLOv8-pose-s fine-tuned on padel → exported to TensorRT FP16.
- Served via NVIDIA Triton Inference Server (same instance as face anonymiser if applicable) with dynamic batching.
- GStreamer appsink → inference → appsrc pattern, wrapped in a Raven AI Engine plugin, outputting skeleton metadata as a structured GStreamer meta buffer downstream.
- Lightweight post-processing stage: Kalman filter on wrist/elbow trajectories + hit-event classifier (a 3-layer MLP or 1D-CNN on windowed joint features) running on CPU alongside GPU inference.
- Prometheus metrics: inference latency P50/P99, keypoint confidence distribution, hit-rate per session — enables drift monitoring without a separate MLOps layer.
