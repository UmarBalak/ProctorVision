import cv2
import time
from functions import *

def process_frame(camera):
    # Call the function that processes the video feed and returns metrics
    result, eye_d, head_d, fps, obj_d, alert_msg = run(camera)
    return result, eye_d, head_d, fps, obj_d, alert_msg

def main():
    # Initialize the camera
    camera = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        return

    # Initialize variables
    violation_count = 0
    camera_active = False  # Flag to track camera status
    prev_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()

        if not ret:
            print("Failed to capture video. Check your camera connection.")
            break

        # Calculate frame rate
        current_time = time.time()
        elapsed_time = current_time - prev_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        prev_time = current_time

        # Convert the frame to RGB for displaying
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)

        # Resize the frame to reduce height (adjust width to maintain aspect ratio)
        frame_rgb = cv2.resize(frame_rgb, (int(frame_rgb.shape[1] * 1.0), int(frame_rgb.shape[0] * 0.75)))

        # Display the resulting frame
        cv2.imshow('Live Feed', frame_rgb)

        # Process frame metrics
        result, eye_d, head_d, fps, obj_d, alert_msg = process_frame(camera)

        if not result:
            violation_count += 1
            print(f"Warning: {violation_count} - {alert_msg}")
            speak(f"Warning number {violation_count}")

            if violation_count == 4:
                print("The exam has been terminated.")
                speak("The exam has been terminated.")
                break

        else:
            # Print real-time metrics to console
            print(f"FPS: {fps:.2f}")
            print(f"Eye Direction: {eye_d}")
            print(f"Head Direction: {head_d}")
            print(f"Background: {'Ok' if obj_d else 'Object detected'}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
