import cv2
import argparse
import time
import sys

def display_rtsp_stream(rtsp_url, window_name="RTSP Stream"):
    """
    Display an RTSP video stream using OpenCV
    
    Args:
        rtsp_url (str): The URL of the RTSP stream
        window_name (str): Title of the display window
    """
    # Create a VideoCapture object
    cap = cv2.VideoCapture(rtsp_url)
    
    # Check if the stream was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream at {rtsp_url}")
        return
    
    print(f"Successfully connected to RTSP stream at {rtsp_url}")
    print("Press 'q' to quit")
    
    # Create a window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Display frames in a loop
    try:
        while True:
            # Read a frame from the stream
            ret, frame = cap.read()
            
            # If frame reading was not successful, try to reconnect
            if not ret:
                print("Error reading frame, attempting to reconnect...")
                cap.release()
                time.sleep(2)  # Wait before reconnecting
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    print("Reconnection failed, exiting")
                    break
                continue
            
            # Display the frame
            cv2.imshow(window_name, frame)
            
            # Wait for 1ms and check if 'q' was pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stream display interrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Stream closed")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Display RTSP video stream")
    parser.add_argument("rtsp_url", help="URL of the RTSP stream (e.g., rtsp://username:password@camera_ip:port/stream)")
    parser.add_argument("--window-name", default="RTSP Stream", help="Name of the display window")
    
    args = parser.parse_args()
    
    # Display the stream
    display_rtsp_stream(args.rtsp_url, args.window_name)