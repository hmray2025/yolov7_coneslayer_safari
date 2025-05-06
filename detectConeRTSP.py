import asyncio
import websockets
import cv2
import json
import time
import numpy as np
from datetime import datetime
import base64
import threading
import queue
import os
import torch
import torch.backends.cudnn as cudnn
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, clip_coords, make_divisible

from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit  # This is needed for CUDA initialization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detection_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("detection_system")

# Configuration class
class Config:
    """Configuration settings for the detection system"""
    # RTSP stream settings
    RTSP_URL = "rtsp://rinao:unicorn@192.168.0.100:8554/streaming/live/1"
    FRAME_BUFFER_SIZE = 1
    
    # WebSocket settings
    WEBSOCKET_URI = "ws://192.168.0.101:8085"
    WS_PING_INTERVAL = 20
    WS_PING_TIMEOUT = 20
    WS_RECONNECT_DELAY = 2
    
    # Detection settings
    MODEL_PATH = "coneslayer-deluxe.pt"
    CONFIDENCE_THRESHOLD = 0.5
    DETECTION_COOLDOWN = 1.0  # seconds
    
    # Output settings
    DETECTION_DIR = Path("detections")
    JPEG_QUALITY = 85
    
    # Performance settings
    OPENCV_THREADS = 2
    TORCH_THREADS = 4
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.DETECTION_DIR.mkdir(exist_ok=True)


# Set CUDA device if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optimize_system_resources():
    """Configure system for optimal performance"""
    # # Set process priority
    # try:
    #     import psutil
    #     p = psutil.Process(os.getpid())
    #     p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
    #     logger.info("Process priority increased")
    # except (ImportError, PermissionError) as e:
    #     logger.warning(f"Could not set process priority: {e}")
    
    # Configure threading and CUDA settings
    torch.set_num_threads(Config.TORCH_THREADS)
    cv2.setNumThreads(Config.OPENCV_THREADS)
    
    # Enable CUDA optimizations if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info(f"CUDA enabled: {torch.cuda.get_device_name(0)}")
    
    # Check for OpenCV CUDA support
    opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if opencv_cuda:
        logger.info(f"OpenCV CUDA support enabled with {cv2.cuda.getCudaEnabledDeviceCount()} devices")
    
    return {
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "opencv_cuda": opencv_cuda
    }

def img_to_base64_string(image):
    _,image = cv2.imencode('.jpg',image)
    imgBase64 = base64.b64encode(image.tobytes())
    return f"data:image/jpeg;base64,{imgBase64.decode()}"

async def send_json(ws, data: Dict):
    """Send JSON data through WebSocket connection"""
    json_data = json.dumps(data)
    await ws.send(json_data)
    logger.debug(f"Sent message: {json_data[:100]}...")


def rtsp_reader(rtsp_url: str, frame_queue: queue.Queue, stop_event: threading.Event):
    """
    Read frames from RTSP stream in a dedicated thread
    
    Args:
        rtsp_url: URL of the RTSP stream
        frame_queue: Queue to store the most recent frame
        stop_event: Event to signal thread termination
    """
    logger.info(f"Starting RTSP reader for {rtsp_url}")
    
    # Try GStreamer pipeline first for lower latency
    gst_pipeline = (
        f'rtspsrc location={rtsp_url} latency=0 buffer-mode=auto ! '
        f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink max-buffers=1 drop=true'
    )
    
    try:
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            logger.info("GStreamer pipeline successfully initialized")
        else:
            logger.warning("GStreamer pipeline failed, falling back to standard capture")
            cap = cv2.VideoCapture(rtsp_url)
    except Exception as e:
        logger.warning(f"GStreamer error: {e}, falling back to standard capture")
        cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        logger.error(f"Failed to open RTSP stream: {rtsp_url}")
        return
    
    # Set capture properties for performance
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
    
    try:
        frame_count = 0
        start_time = time.time()
        last_log_time = start_time
        fps = 0
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            
            if ret:
                # cv2.imshow("Raw Image", frame)
                frame_count += 1
                current_time = time.time()
                
                # Calculate and log FPS every 5 seconds
                if current_time - last_log_time >= 5.0:
                    fps = frame_count / (current_time - last_log_time)
                    logger.info(f"RTSP Stream FPS: {fps:.2f}")
                    frame_count = 0
                    last_log_time = current_time
                
                # Clear queue and add new frame
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    # Get and discard one item
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(frame)
            else:
                logger.warning("Failed to read frame, reconnecting...")
                cap.release()
                time.sleep(1)
                # cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    logger.error("Reconnection failed, retrying...")
                    time.sleep(5)
                    cap = cv2.VideoCapture(rtsp_url)
                    continue
            
            # Small sleep to prevent CPU overuse
            time.sleep(0.001)
    except Exception as e:
        logger.error(f"Error in RTSP reader: {e}")
    finally:
        logger.info("Stopping RTSP reader")
        cap.release()


class DetectionModel:
    """GPU-accelerated object detection model with optimized inference"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize detection model with GPU acceleration
        
        Args:
            model_path: Path to the model file
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.imgsz = 1280  # Default image size
        self.iou_thres = 0.50
        self.reference_colors = np.array([
            (248, 113, 84),
            (241, 67, 60),
            (252, 64, 29),
            (185, 39, 42),
            (255, 134, 106)], dtype=np.float32)
        self.threshold = 100
        logger.info(f"Loading detection model from {model_path}")
        
        try:
            
            # Load model to GPU if available
            if torch.cuda.is_available():
                # Load model
                
                self.model = attempt_load(model_path, map_location=DEVICE)  # load FP32 model
                stride = int(self.model.stride.max())  # model stride
                imgsz = check_img_size(self.imgsz, s=stride)  # check img_size
                self.model.eval()
                
                # Half precision for faster inference
                self.model = self.model.half()
                logger.info("Model loaded in half precision on GPU")
            else:
                # CPU fallback
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                self.model.eval()
                logger.info("Model loaded on CPU")
            
            # Warm up the model to reduce first inference latency
            self._warmup()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _warmup(self):
        """Warm up the model to reduce first inference latency"""
        logger.info("Warming up model...")
        cudnn.benchmark = True
        dummy_input = torch.zeros((1, 3, self.imgsz, self.imgsz)).to(DEVICE)
        if torch.cuda.is_available():
            dummy_input = dummy_input.half()
        
        with torch.no_grad():
            for _ in range(2):
                _ = self.model(dummy_input)
        
        cuda_code = """
        __global__ void detect_colors(float *frame, float *ref_colors, float *result, 
                                     int width, int height, int num_colors, float threshold) {
            // Calculate pixel position
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= width * height) return;
            
            // Get pixel RGB values
            float r = frame[idx * 3];
            float g = frame[idx * 3 + 1];
            float b = frame[idx * 3 + 2];
            
            // Initialize result to 0 (no color match)
            result[idx] = 0;
            
            // Check each reference color
            for (int c = 0; c < num_colors; c++) {
                float ref_r = ref_colors[c * 3];
                float ref_g = ref_colors[c * 3 + 1];
                float ref_b = ref_colors[c * 3 + 2];
                
                // Calculate Euclidean distance
                float distance = sqrtf(powf(r - ref_r, 2) + powf(g - ref_g, 2) + powf(b - ref_b, 2));
                
                // If distance is less than threshold, mark as match and break
                if (distance <= threshold) {
                    result[idx] = c + 1;  // +1 so 0 can mean "no match"
                    break;
                }
            }
        }
        """
        
        self.module = SourceModule(cuda_code)
        self.kernel = self.module.get_function("detect_colors")
        logger.info("Model warmup complete")
    
    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection with CUDA acceleration and no_grad for speed
        
        Args:
            frame: Image frame as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class
        """
        try:
            # Measure inference time
            start_time = time.time()
            
            # Preprocess
            input_tensor = self.preprocess(frame)
            
            # Inference
            with torch.no_grad():
                results = self.model(input_tensor)[0]
            
            # Postprocess
            detections = self.postprocess(results, input_tensor.shape)
            
            inference_time = (time.time() - start_time) * 1000  # ms
            logger.info(f"Inference time: {inference_time:.2f}ms, Detections: {len(detections)}")
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def colorDetect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect reference colors in the input frame.
        
        Args:
            frame: RGB image as numpy array with shape (height, width, 3)
            
        Returns:
            Mask where each pixel is labeled with the index+1 of the matched color
            or 0 if no match found
        """
        height, width = frame.shape[:2]
        
        # Convert to float32 and copy to device
        frame_gpu = cuda.mem_alloc(frame.astype(np.float32).nbytes)
        cuda.memcpy_htod(frame_gpu, frame.astype(np.float32).reshape(-1))
        
        # Copy reference colors to device
        ref_colors_gpu = cuda.mem_alloc(self.reference_colors.nbytes)
        cuda.memcpy_htod(ref_colors_gpu, self.reference_colors)
        
        # Allocate output array
        result = np.zeros(width * height, dtype=np.float32)
        result_gpu = cuda.mem_alloc(result.nbytes)
        
        # Configure grid dimensions
        block_size = 256
        grid_size = (width * height + block_size - 1) // block_size
        
        # Run kernel
        self.kernel(
            frame_gpu, ref_colors_gpu, result_gpu, 
            np.int32(width), np.int32(height),
            np.int32(len(self.reference_colors)), np.float32(self.threshold),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        
        # Copy result back to host
        cuda.memcpy_dtoh(result, result_gpu)
        
        # Reshape to 2D mask
        return result.reshape(height, width)
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert frame to tensor and normalize for YoloV7 inference.

        Args:
            frame: Image frame as numpy array (BGR format)

        Returns:
            Preprocessed tensor ready for model inference
        """
        # Resize and pad the frame to the expected input size while maintaining aspect ratio
        stride = int(self.model.stride.max())  # Get model stride
        expected_size = make_divisible(self.imgsz, stride)  # Ensure size is divisible by stride
        frame, ratio, (dw, dh) = letterbox(frame, expected_size, stride=stride, auto=False)

        # Convert BGR to RGB
        frame = frame[:, :, ::-1]

        # Convert to CHW format (channels, height, width)
        frame = frame.transpose(2, 0, 1)

        # Convert to contiguous array
        frame = np.ascontiguousarray(frame, dtype=np.float32)

        # Normalize pixel values to [0, 1]
        frame /= 255.0

        # Convert to PyTorch tensor and add batch dimension
        img_tensor = torch.from_numpy(frame).to(DEVICE).unsqueeze(0)

        # Use half precision if CUDA is available
        if torch.cuda.is_available():
            img_tensor = img_tensor.half()

        # Debugging: Log the shape of the preprocessed tensor
        logger.debug(f"Preprocessed tensor shape: {img_tensor.shape}")

        return img_tensor
    
    def postprocess(self, results, tensor_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        Process model outputs into structured detections.

        Args:
            results: Raw model output.
            tensor_shape: Shape of the input image (H, W, C).

        Returns:
            List of detection dictionaries with bbox, confidence, and class.
        """
        detections = []
        try:
            # Apply NMS
            result_array = non_max_suppression(results, self.conf_threshold, self.iou_thres)

            for i, det in enumerate(result_array):
                if det is not None and len(det):
                    # Rescale boxes from model input size to original image size
                    det[:, :4] = scale_coords(tensor_shape[2:], det[:, :4], [720,1280]).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.model.names[int(cls)]} {conf:.2f}'
                        detections.append({
                            "bbox": xyxy,
                            "confidence": float(conf),
                            "class": label,
                            "input_shape": tensor_shape
                        })

        except Exception as e:
            logger.error(f"Error in postprocessing: {e}", exc_info=True)
            return []

        return detections

def draw_detection(frame: np.ndarray, detection: Dict, original_shape: Tuple[int, int]) -> np.ndarray:
    """
    Draw detection bounding box and label on frame.

    Args:
        frame: Image frame.
        detection: Detection info with bbox, confidence, and class.
        original_shape: Original shape of the frame (height, width).

    Returns:
        Frame with detection visualization.
    """
    # Unpack detection info
    bbox = detection["bbox"]
    x1, y1, x2, y2 = [int(coord) for coord in bbox]

    # Clip coordinates to ensure they are within the image bounds
    x1 = max(0, min(x1, original_shape[1]))
    y1 = max(0, min(y1, original_shape[0]))
    x2 = max(0, min(x2, original_shape[1]))
    y2 = max(0, min(y2, original_shape[0]))

    # Draw bounding box
    label = detection["class"]
    color = (0, 255, 0)  # Green
    c1, c2 = (x1, y1), (x2, y2)
    cv2.rectangle(frame, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)

    # Calculate text size for better positioning
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

    # Draw background rectangle for text
    cv2.rectangle(
        frame,
        (x1, y1 - text_size[1] - 10),
        (x1 + text_size[0], y1),
        color,
        -1
    )

    # Draw text
    cv2.putText(
        frame,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),  # Black text
        2
    )

    return frame

async def process_frames(detection_model: DetectionModel, 
                         frame_queue: queue.Queue, 
                         websocket_uri: str, 
                         stop_event: threading.Event):
    """
    Process frames for object detection and send alerts via websocket
    
    Args:
        detection_model: Initialized detection model
        frame_queue: Queue containing the latest frame
        websocket_uri: URI for WebSocket connection
        stop_event: Event to signal process termination
    """
    last_detection_time = 0
    detection_count = 0
    
    logger.info(f"Starting frame processing with WebSocket: {websocket_uri}")

    # Get names and colors
    names = detection_model.model.module.names if hasattr(detection_model.model, 'module') else detection_model.model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Create a window
    # cv2.namedWindow("RTSP Stream", cv2.WINDOW_NORMAL)
    
    while not stop_event.is_set():
        # try:
        # Connect with configured ping settings
        async with websockets.connect(
            websocket_uri,
            ping_interval=Config.WS_PING_INTERVAL,
            ping_timeout=Config.WS_PING_TIMEOUT
        ) as ws_client:
            logger.info("WebSocket connection established")
            
            while not stop_event.is_set():
                try:
                    # Get the latest frame from the queue
                    try:
                        current_frame = frame_queue.get_nowait()
                    except queue.Empty:
                        # No frame available, wait briefly
                        logger.info("No frame available, waiting...")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Current time for cooldown check
                    current_time = time.time()
                    
                    # Skip detection if we're in cooldown period
                    if current_time - last_detection_time < Config.DETECTION_COOLDOWN:
                        await asyncio.sleep(0.001)
                        continue
                    
                    # Run detections
                    detections = detection_model.detect(current_frame)
                     # Filter by confidence (already done in the model, but just in case)
                    valid_yolo_detections = [d for d in detections if d["confidence"] > Config.CONFIDENCE_THRESHOLD]

                    color_detections = detection_model.colorDetect(current_frame)
                    # logger.info(f"Color detection result: {color_detections}")
                    if color_detections.any():
                        # Convert color detections to a mask
                        # logger.info(f"Color detection result: {color_detections}")
                        logger.info(f"Color detection result: {np.argwhere(color_detections > 0)}")
                    valid_color_detections = [d for d in color_detections if d > 0]
                    logger.info(f"Frame Processed: {len(valid_yolo_detections)} YOLO detections, {len(valid_color_detections)} color detections")
                   
                    if valid_yolo_detections | valid_color_detections>5:
                        
                        # Take the highest confidence detection
                        detection = max(valid_yolo_detections, key=lambda x: x["confidence"])
                        # Save the image with detection
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        detection_count += 1
                        image_filename = f"detection_{timestamp}_{detection_count:04d}.jpg"
                        image_path = str(Config.DETECTION_DIR / image_filename)
                        # Draw bounding box on frame copy for saving
                        annotated_frame = draw_detection(current_frame.copy(), detection, [720,1280,3])
                        imageUrl = img_to_base64_string(annotated_frame)
                        # Send detection via websocket
                        # print("Sending Websocket")
                        await send_json(ws_client,
                                        {"action": "NewAlert", "args": 
                                            {"event": "alert", "image": imageUrl}})
                        # cv2.imshow("RTSP Stream", annotated_frame)
                        # Update last detection time
                        last_detection_time = current_time
                        logger.info(f"Detection sent: {detections[0]['class']} ({detections[0]['confidence']:.2f})")
                    else:
                        logger.info("No valid detections found")
                        # cv2.imshow("RTSP Stream", current_frame)
                        # pass
                        
                    # Minimal sleep for event loop to process other tasks
                    await asyncio.sleep(0.001)
                
                    
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.warning(f"WebSocket connection lost: {e}, reconnecting...")
                    break
                        
        # except Exception as e:
        #     logger.error(f"Connection error: {e}")
        
        # Wait before attempting reconnection if not stopping
        if not stop_event.is_set():
            logger.info(f"Attempting reconnection in {Config.WS_RECONNECT_DELAY} seconds...")
            await asyncio.sleep(Config.WS_RECONNECT_DELAY)


async def main():
    """Main application entry point"""
    try:
        logger.info("Starting detection system")
        
        # Create required directories
        Config.create_directories()
        
        # Optimize system resources
        system_info = optimize_system_resources()
        logger.info(f"System configuration: {system_info}")
        
        # Create shared queue for frames
        frame_queue = queue.Queue(maxsize=Config.FRAME_BUFFER_SIZE)
        
        # Create stop event for graceful shutdown
        stop_event = threading.Event()
        
        # Initialize detection model
        detection_model = DetectionModel(
            model_path=Config.MODEL_PATH, 
            conf_threshold=Config.CONFIDENCE_THRESHOLD
        )
        
        # Start RTSP reader in a separate thread
        rtsp_thread = threading.Thread(
            target=rtsp_reader, 
            args=(Config.RTSP_URL, frame_queue, stop_event),
            daemon=True,
            name="RTSPReaderThread"
        )
        rtsp_thread.start()
        
        # Give the thread a moment to start capturing frames
        await asyncio.sleep(2)
        logger.info("RTSP reader initialized, starting frame processing")
        
        # Start processing frames
        try:
            await process_frames(
                detection_model=detection_model, 
                frame_queue=frame_queue,
                websocket_uri=Config.WEBSOCKET_URI,
                stop_event=stop_event
            )
        finally:
            # Ensure clean shutdown
            logger.info("Shutting down...")
            stop_event.set()
            rtsp_thread.join(timeout=5.0)
            logger.info("Detection system stopped")
    
    except KeyboardInterrupt:
        logger.info("Detection system interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)


if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())