import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class ColorDetector:
    """A class to detect specified colors in video frames using GPU acceleration."""
    
    def __init__(self, reference_colors: List[Tuple[int, int, int]], threshold: float = 30.0):
        """
        Initialize the ColorDetector with reference colors.
        
        Args:
            reference_colors: List of RGB color tuples to detect
            threshold: Euclidean distance threshold for color similarity
        """
        self.reference_colors = np.array(reference_colors, dtype=np.float32)
        self.threshold = threshold
        self._setup_gpu()
        
    def _setup_gpu(self):
        """Set up the CUDA kernel for color detection."""
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
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
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
    
    def visualize_detected_colors(self, frame: np.ndarray, mask: np.ndarray, 
                                 highlight: bool = True) -> np.ndarray:
        """
        Visualize detected colors in the frame.
        
        Args:
            frame: Original RGB frame
            mask: Detection mask from detect() method
            highlight: If True, highlight detected pixels; if False, show only detected pixels
            
        Returns:
            Visualization image
        """
        vis_frame = frame.copy()
        
        if highlight:
            # Highlight detected pixels
            for i in range(len(self.reference_colors)):
                color_idx = i + 1
                color_mask = (mask == color_idx)
                if np.any(color_mask):
                    vis_frame[color_mask] = [255, 0, 0]  # Highlight in yellow
        else:
            # Show only detected pixels
            background = np.zeros_like(frame)
            for i in range(len(self.reference_colors)):
                color_idx = i + 1
                color_mask = (mask == color_idx)
                if np.any(color_mask):
                    background[color_mask] = self.reference_colors[i]
            vis_frame = background
            
        return vis_frame

def demo_with_webcam():
    """Demo function using webcam input."""
    # Reference colors to detect (RGB format)
    reference_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255)   # Cyan
    ]
    
    detector = ColorDetector(reference_colors, threshold=300.0)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert from BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect colors
            mask = detector.detect(frame_rgb)
            
            # Visualize detected colors
            highlighted = detector.visualize_detected_colors(frame_rgb, mask)
            
            # Convert back to BGR for display
            highlighted_bgr = cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR)
            
            # Display the result
            cv2.imshow('Color Detection', highlighted_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

def test_with_image(image_path: str, reference_colors: List[Tuple[int, int, int]]):
    """Test the color detector with a static image."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
        
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize detector
    detector = ColorDetector(reference_colors, threshold=50.0)
    
    # Detect colors
    mask = detector.detect(image_rgb)
    
    # Visualize results
    highlighted = detector.visualize_detected_colors(image_rgb, mask, highlight=True)
    detected_only = detector.visualize_detected_colors(image_rgb, mask, highlight=False)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Highlighted Detection")
    plt.imshow(highlighted)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Detected Colors Only")
    plt.imshow(detected_only)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage with webcam:
# demo_with_webcam()

# Example usage with an image:
# reference_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
# test_with_image("path/to/image.jpg", reference_colors)

def process_video_file(
    video_path: str, 
    reference_colors: List[Tuple[int, int, int]], 
    output_path: Optional[str] = None,
    threshold: float = 50.0
):
    """
    Process a video file to detect specified colors.
    
    Args:
        video_path: Path to input video file
        reference_colors: List of RGB color tuples to detect
        output_path: Path to save output video (if None, will display only)
        threshold: Color detection threshold
    """
    # Initialize detector
    detector = ColorDetector(reference_colors, threshold=threshold)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect colors
            mask = detector.detect(frame_rgb)
            
            # Visualize detected colors
            highlighted = detector.visualize_detected_colors(frame_rgb, mask)
            
            # Convert back to BGR for display/saving
            highlighted_bgr = cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR)
            
            # Write to output file if specified
            if writer:
                writer.write(highlighted_bgr)
            
            # Display progress
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Processing frame {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)")
            
            # Display the frame (if not processing too many frames)
            if frame_count < 1000:  # Only show UI for shorter videos
                cv2.imshow('Color Detection', highlighted_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Video processing complete.")

# Main function for command-line usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect colors in video using GPU acceleration")
    parser.add_argument("--input", type=str, help="Input video file path (or 'webcam' for live feed)")
    parser.add_argument("--output", type=str, help="Output video file path", default=None)
    parser.add_argument("--threshold", type=float, help="Color detection threshold", default=50.0)
    
    args = parser.parse_args()
    
    # Default reference colors
    reference_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),   # Cyan
        (139, 126, 125)    # Black/brown
    ]
    
    if args.input == "webcam":
        demo_with_webcam()
    elif args.input:
        process_video_file(args.input, reference_colors, args.output, args.threshold)
    else:
        print("Please specify an input video with --input or use --input webcam for live feed")