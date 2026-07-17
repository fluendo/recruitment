import sys
import numpy as np
import cv2
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GstVideo

# Initialize GStreamer
Gst.init(None)

def calculate_iou(boxA, boxB):
    """Helper to calculate Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class GstMVTrackerPipeline:
    def __init__(self, video_path):
        self.video_path = video_path
        # Schema: [{"id": 1, "bbox": (x1, y1, x2, y2), "should_blur": True, "embedding": [...]}]
        self.tracked_faces = []  
        self.io_threshold = 0.4  # Threshold to verify if a face matches an existing track
        
        # Build GStreamer Pipeline String
        pipeline_str = f"""
            filesrc location={self.video_path} ! qtdemux ! h264parse ! 
            avdec_h264 name=decoder ! videoconvert ! appsink name=sink emit-signals=true max-buffers=1 drop=true
        """
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.setup_elements()

    def setup_elements(self):
        # 1. Export Motion Vectors natively via modified gst-libav underlying flags
        decoder = self.pipeline.get_by_name("decoder")
        if decoder:
            decoder.set_property("options", "flags2=+export_mvs")

        # 2. Setup Frame Sink
        sink = self.pipeline.get_by_name("sink")
        sink.connect("new-sample", self.on_frame_arrival)

    def convert_buffer_to_ndarray(self, buf, caps):
        """Converts GStreamer buffer memory into a processable OpenCV BGR Image."""
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return None
        
        # Assuming standard BGR/RGB output format from videoconvert
        frame_data = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
        buf.unmap(map_info)
        return frame_data.copy()

    def run_face_alignment(self, face_img):
        """Aligns facial landmarks to drastically increase ReID model consistency."""
        # TODO: Inject your facial landmark alignment logic here (e.g., affine transform)
        return face_img

    def run_reid_models(self, frame_data, selective_roi=None):
        """
        Runs heavy inference models. If selective_roi is provided, it isolates 
        and extracts features only for that newly discovered region.
        """
        if selective_roi:
            print(f"-> Selective ReID: Processing targeted region {selective_roi}")
            x1, y1, x2, y2 = selective_roi
            face_crop = frame_data[y1:y2, x1:x2]
            aligned_face = self.run_face_alignment(face_crop)
            # ReID Inference Loop:
            # embedding = reid_model(aligned_face)
            # if embedding matches target_db -> self.tracked_faces.append(...)
        else:
            print("-> Full Heavy Inference: Running YOLO Face Detection across frame")
            # detected_faces = yolo_model(frame_data)
            # self.tracked_faces = [] Rebuild active profile indices
            pass

    def track_with_motion_vectors(self, gst_buffer, current_frame_img):
        """
        Extracts native GStreamer motion vectors and evaluates their reliability.
        Falls back to IoU tracking if movement is chaotic, too fast, or missing.
        """
        meta = GstVideo.buffer_get_video_motion_vector_meta(gst_buffer) if hasattr(GstVideo, 'buffer_get_video_motion_vector_meta') else None
        
        # Hard thresholds for motion vector validation
        MAX_TRUSTED_STD = 12.0       # Max direction variance. Higher means chaotic/noisy motion.
        MAX_TRUSTED_MAGNITUDE = 45.0 # Max speed (pixels/frame). Higher means too fast to track linearly.
        MIN_VECTOR_DENSITY = 4       # Minimum number of vectors inside a box needed to trust calculation.
        
        trigger_iou_fallback = False
        
        # Check Condition 1: Do we have metadata and vectors at all?
        if meta and hasattr(meta, 'vectors') and len(meta.vectors) > 0:
            
            for face in self.tracked_faces:
                x1, y1, x2, y2 = face["bbox"]
                face_dx = []
                face_dy = []
                
                # Isolate vectors belonging solely to this specific face region
                for mv in meta.vectors:
                    if x1 <= mv.dst_x <= x2 and y1 <= mv.dst_y <= y2:
                        face_dx.append(mv.dx)
                        face_dy.append(mv.dy)
                
                # Check Condition 2: Do we have enough vectors to calculate metrics safely?
                if len(face_dx) >= MIN_VECTOR_DENSITY:
                    # Convert to numpy arrays for vector math
                    np_dx = np.array(face_dx)
                    np_dy = np.array(face_dy)
                    
                    # Calculate magnitude (speed) and standard deviation (chaos)
                    magnitudes = np.sqrt(np_dx**2 + np_dy**2)
                    avg_magnitude = np.mean(magnitudes)
                    std_dx = np.std(np_dx)
                    std_dy = np.std(np_dy)
                    
                    # Check Condition 3: Is the movement clean, smooth, and manageable?
                    if avg_magnitude < MAX_TRUSTED_MAGNITUDE and std_dx < MAX_TRUSTED_STD and std_dy < MAX_TRUSTED_STD:
                        # Smooth, reliable motion -> Apply cheap vector shift tracking
                        median_dx = int(np.median(np_dx))
                        median_dy = int(np.median(np_dy))
                        face["bbox"] = (x1 + median_dx, y1 + median_dy, x2 + median_dx, y2 + median_dy)
                        print(f"-> MV Tracked: Face moved by ({median_dx}, {median_dy}). Speed: {avg_magnitude:.1f}")
                    
                    else:
                        # ELSE CONDITION A: Movement is too fast or vectors point in chaotic directions
                        print(f"-> MV Alert: Too much/noisy movement (Mag: {avg_magnitude:.1f}, Std_X: {std_dx:.1f}).")
                        trigger_iou_fallback = True
                else:
                    # ELSE CONDITION B: Encoder switched to Intra-refresh blocks due to radical changes
                    print("-> MV Alert: Missing vectors in box region (Intra-refresh triggered by encoder).")
                    trigger_iou_fallback = True
        else:
            # ELSE CONDITION C: No metadata present in stream stream configuration
            print("-> MV Alert: No motion vector metadata found in this frame.")
            trigger_iou_fallback = True

        # Handle the unified fallback path
        if trigger_iou_fallback:
            print("-> Cheap Tracking Fallback: Activating high-speed IoU overlap engine.")
            self.execute_iou_fallback_tracking(current_frame_img)

    def execute_iou_fallback_tracking(self, current_frame_img):
        """
        Fallback loop executing ultra-lightweight traditional tracking (e.g., OpenCV template 
        matching or a light cascade) paired with an IoU matrix overlap check.
        """
        # fresh_detections = light_face_detector(current_frame_img)
        # Update self.tracked_faces based on closest spatial overlap matches
        pass
    def on_frame_arrival(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        
        # Fetch current raw pixel array for alignment or selective inference passes
        frame_img = self.convert_buffer_to_ndarray(buf, caps)
        
        # Determine Frame Type natively via GStreamer Buffer Flags
        is_keyframe = not (buf.flags & Gst.BufferFlags.DELTA_UNIT)

        if is_keyframe:
            print(f"\n--- [I-Frame Detected] Buffer PTS: {buf.pts} ---")
            
            if len(self.tracked_faces) == 0:
                # Cold start: Run full pipeline
                self.run_reid_models(frame_img)
            else:
                # 1. Run a fast, light bounding-box detection check
                # current_boxes = light_face_detector(frame_img)
                current_boxes = []
                
                # 2. Evaluate Bounding Box Overlap to distinguish periodic GOP resets from hard scene changes
                total_overlap = 0
                for tracked in self.tracked_faces:
                    max_overlap = max([calculate_iou(tracked["bbox"], cb) for cb in current_boxes] + [0])
                    total_overlap += max_overlap
                
                avg_overlap = total_overlap / len(self.tracked_faces) if self.tracked_faces else 0
                
                if avg_overlap < self.io_threshold:
                    print("-> Radical Scene Change confirmed on I-Frame. Resetting IDs.")
                    self.run_reid_models(frame_img)
                else:
                    print("-> Standard GOP Periodic I-Frame (Scene stable). Treating via tracking rules.")
                    self.track_with_motion_vectors(buf, frame_img)
        else:
            print(f"\n--- [P-Frame Detected] Buffer PTS: {buf.pts} ---")
            # Apply low-cost vector calculation updates 
            self.track_with_motion_vectors(buf, frame_img)
            
            # 3. Dynamic Scan for unexpected new faces in unmapped regions
            # new_detections = light_face_detector(frame_img)
            new_detections = []
            for det in new_detections:
                is_tracked = any(calculate_iou(det, f["bbox"]) > self.io_threshold for f in self.tracked_faces)
                if not is_tracked:
                    # Run targeted ReID solely on the newly discovered region
                    self.run_reid_models(frame_img, selective_roi=det)

        return Gst.FlowReturn.OK

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        bus = self.pipeline.get_bus()
        try:
            msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        except KeyboardInterrupt:
            pass
        self.pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    tracker = GstMVTrackerPipeline("cctv_stream.mp4")
    tracker.start()