import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cosine similarity threshold for deciding if a face matches someone in the gallery.
SIM_THRESHOLD = 0.6

# Re-check a track's identity every N frames
REFRESH_EVERY = 35

# Folder containing one reference image per authorized person 
GALLERY_DIR = "reference_images"

# 1. Face detector: small YOLO variant trained specifically for faces
detector = YOLO("yolov8n-face.pt")

# 3. Face embedding network
embedder = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

def embed_face(bgr_crop: np.ndarray) -> np.ndarray:
    """
    Convert a BGR face crop into a unit-norm embedding for cosine similarity matching.
    """
    face = cv2.resize(bgr_crop, (160, 160))

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = embedder(tensor).cpu().numpy()[0]

    return emb / (np.linalg.norm(emb) + 1e-8)

def build_gallery(folder: str) -> dict:
    """Load every image in `folder` and compute its embedding."""
    gallery = {}
    if not os.path.isdir(folder):
        print(f"[!] Gallery folder '{folder}' not found. Everyone will be blurred.")
        return gallery

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        image = cv2.imread(path)
        if image is None:
            continue

        # Run the detector on the reference image to crop the face cleanly
        result = detector(image, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            print(f"[!] No face found in {filename}, skipping.")
            continue

        # Use the first detected face (reference images should contain one person)
        x1, y1, x2, y2 = map(int, result.boxes[0].xyxy[0].tolist())
        crop = image[max(0, y1):y2, max(0, x1):x2]

        person_id = os.path.splitext(filename)[0]  # filename without extension
        gallery[person_id] = embed_face(crop)
        print(f"[+] Added {person_id} to the gallery.")

    return gallery

gallery = build_gallery(GALLERY_DIR)

# 4. Gallery matching via cosine similarity

def is_authorized(emb: np.ndarray) -> bool:
    """
    Return True if the embedding matches someone in the gallery above threshold.
    """
    if not gallery:
        return False

    gallery_matrix = np.stack(list(gallery.values()))         # shape (N, 512)
    similarities = gallery_matrix @ emb                       # shape (N,)
    best_similarity = float(similarities.max())

    return best_similarity >= SIM_THRESHOLD

def apply_blur(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """
    Anonymization filter: Blur the face region in-place with a kernel that scales with face size
    """
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    
    kernel_size = max(15, ((x2 - x1) // 3) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)

def main():
    track_decisions = {}

    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("[!] Could not open video source.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Detector + tracker (each box gets an ID across frames)
        results = detector.track(frame, persist=True, verbose=False)[0]

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(box.id.item()) if box.id is not None else -1

                # Look up the cached decision for this track (if any)
                decision, age = track_decisions.get(track_id, (None, 0))
                needs_identity_check = decision is None or age >= REFRESH_EVERY

                if needs_identity_check:
                    # Run the embedding model 
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        emb = embed_face(crop)
                        decision = "allow" if is_authorized(emb) else "anonymize"
                        track_decisions[track_id] = (decision, 0)
                else:
                    # Reuse the cached decision, just increment the age counter
                    track_decisions[track_id] = (decision, age + 1)

                # Privacy-by-default: if unsure or not on the list, blur
                if decision != "allow":
                    apply_blur(frame, x1, y1, x2, y2)

        cv2.imshow("Anonymized Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()