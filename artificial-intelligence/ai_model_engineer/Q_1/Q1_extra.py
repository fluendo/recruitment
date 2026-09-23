import cv2
import numpy as np
from ultralytics import YOLO

class OpenCVDepthAnonymizer:
    def __init__(self, source=0):
        # Initialize video source
        self.cap = cv2.VideoCapture(source)
        
        self.frame_counter = 0
        self.skip_interval = 15          # Compute depth every N frames, use cache info in between
        self.cached_depth_map = None
        
        # Pipeline tunnable configurations
        self.downscale_resolution = (384, 384) # Low resolution to speed up inference
        self.distance_threshold = 0.6          # Threshold separating foreground from background
        self.blur_kernel = (51, 51)            # Strength of background blur
        
        # Load Depth Estimation Model
        self.depth_net = cv2.dnn.readNetFromONNX("monocular_depth_model.onnx")

        # Load Face Detection Model (for instance, OpenCV Haar Cascade)
        self.face_net = YOLO("yolo26n-face.pt")

    def downscale_for_inference(self, frame):
        """
        Reduces resolution of images to minimize computational load for the depth network
        """
        return cv2.resize(frame, self.downscale_resolution)

    def run_depth_inference(self, small_frame, original_shape):
        """
        Runs monocular depth estimation. Gets as input the downscaled frame (small_frame) and
        the original frame shape to resize the depth map back to original dimensions afterwards
        """
        h, w = original_shape[:2]
        
        # Preprocess frame into a blob format, normalizes pixel values to [0,1] and swaps channels from BGR to RGB
        blob = cv2.dnn.blobFromImage(small_frame, 1/255.0, self.downscale_resolution, swapRB=True, crop=False)
        
        # Run depth estimation model
        self.depth_net.setInput(blob)
        raw_depth = self.depth_net.forward()
    
        # Resize back to original frame dimensions and normalize (0 to 1)
        depth_map = cv2.resize(raw_depth[0], (w, h))
        normalized_depth = cv2.normalize(depth_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        return normalized_depth

    def detect_faces(self, frame):
        """
        Runs YOLO inference to detect faces and returns a list of bounding boxes: [(x1, y1, x2, y2), ...]
        """
        # Run YOLO prediction
        results = self.face_net(frame, verbose=False, conf=0.5)
        
        boxes = []
        for r in results:
            for box in r.boxes:
                # Extract coordinates [x1, y1, x2, y2] and convert to integers
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                boxes.append((x1, y1, x2, y2))
                
        return boxes

    def apply_selective_face_blur(self, frame):
        h, w = frame.shape[:2]
        self.frame_counter += 1
        
        # Instead of computing depth on every frame, we compute it selectively based on should_recompute
        should_recompute = (
            self.frame_counter % self.skip_interval == 0 or 
            self.cached_depth_map is None
        )
        if should_recompute:
            small_frame = self.downscale_for_inference(frame)
            self.cached_depth_map = self.run_depth_inference(small_frame, frame.shape)

        # Reuse the cached map for intermediate frames
        current_depth_map = self.cached_depth_map
        
        output_frame = frame.copy()

        # Detect Faces using YOLO
        detected_faces = self.detect_faces(frame)

        # Compute median depth per face and blur only background faces
        for (x1, y1, x2, y2) in detected_faces:
            # Ensure coordinates stay within valid frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Isolate the depth map values strictly inside this face's bounding box
            face_depth_region = current_depth_map[y1:y2, x1:x2]
            
            if face_depth_region.size == 0:
                continue
                
            avg_face_depth = np.median(face_depth_region)
            if avg_face_depth > self.distance_threshold:
                # BACKGROUND FACE -> Blur only specific face region of interest (ROI)
                face_roi = output_frame[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_roi, self.blur_kernel, 30)
                output_frame[y1:y2, x1:x2] = blurred_face
            else:
                # FOREGROUND FACE -> Do not blur
                pass
                
        return output_frame

    def start(self):
        print("Starting YOLO + Depth Anonymization Pipeline... Press 'q' to exit.")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("End of video stream or failed to grab frame.")
                break
            
            processed_frame = self.apply_selective_face_blur(frame)
            cv2.imshow("YOLO Selective Depth Anonymization", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = OpenCVDepthAnonymizer(source=0)
    pipeline.start()