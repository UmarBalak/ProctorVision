import cv2
import streamlit as st
import time
from functions import *

st.set_page_config(
    page_title="Examiner.AI",
    page_icon=":camera:",
    initial_sidebar_state="expanded",
)

# Custom CSS to style buttons, text, layout, and metric cards
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: black;
            text-align: center;
            margin-bottom: 30px;
        }
        .subheader {
            font-size: 24px;
            font-weight: bold;
            color: black;
        }
        .camera-status {
            font-size: 18px;
            font-weight: bold;
            color: #32CD32;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .stButton>button:hover {
            background-color: white;
            color: #FF4B4B;
        }
        .streamlit-container {
            padding: 20px;
        }
        .metric-card {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            color: #FF4B4B;
            text-align: center;
            font-family: Arial, sans-serif;
        }
        .metric-card h2 {
            font-size: 18px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .metric-card p {
            font-size: 26px;
            margin: 0;
            font-weight: bold;
        }
        .metric-card span {
            font-size: 14px;
            margin-top: 5px;
            display: block;
            color: #f0f0f0;
        }
        .fps-card {
            background-color: white;  /* Green for FPS */
        }
        # .eye-card {
        #     background-color: #17a2b8;  /* Light blue for Eye Direction */
        # }
        # .head-card {
        #     background-color: #ffc107;  /* Yellow for Head Direction */
        # }
        # .obj-card {
        #     background-color: #dc3545;  /* Red for Background Status */
        # }
    </style>
""", unsafe_allow_html=True)

# Initialize the camera
camera = cv2.VideoCapture(0)

def process_frame(camera):
    # Call the function that processes the video feed and returns metrics
    result, eye_d, head_d, fps, obj_d, alert_msg = run(camera)
    return result, eye_d, head_d, fps, obj_d, alert_msg

def main():
    st.markdown("<h1 class='main-title'>Examiner.AI:<br> AI Proctored Exam System </h1>", unsafe_allow_html=True)

    # Sidebar for camera control buttons
    with st.sidebar:
        st.header("Controls")
        start_button = st.button("Start Monitoring")
        stop_button = st.button("Stop/Reset")

        # Placeholder for warnings below the buttons
        warning_count_placeholder = st.empty() 
        warning_placeholder1 = st.empty()
        warning_placeholder2 = st.empty()
        warning_placeholder3 = st.empty()
        warning_placeholder4 = st.empty()

    # Placeholder for the camera status message
    camera_status_placeholder = st.empty()

    # st.markdown("<h3 class='subheader'>Live Video Feed</h3>", unsafe_allow_html=True)
    subheading1_placeholder = st.empty()
    subheading1_placeholder.markdown("<p class='camera-status'>Press 'Start Monitoring' to begin.</p>", unsafe_allow_html=True)
    # Placeholder for video feed with reduced height
    FRAME_WINDOW = st.image([])  # Placeholder for the video feed

    # st.markdown("<h3 class='subheader'>Real-Time Metrics</h3>", unsafe_allow_html=True)
    subheading2_placeholder = st.empty()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    # Create placeholders for real-time metrics using the card-based UI
    with c1:
        fps_placeholder = st.empty()
    with c2:
        eye_placeholder = st.empty()
    with c3:
        head_placeholder = st.empty()
    with c4:
        obj_placeholder = st.empty()

    # Initialize warning and violation counters
    violation_count = 0
    camera_active = False  # Flag to track camera status

    # If the start button is pressed
    if start_button:
        camera_status_placeholder.markdown("<p class='camera-status'>Camera is active. Press 'Stop/Reset' to stop.</p>", unsafe_allow_html=True)
        prev_time = time.time()
        camera_active = True

    while camera_active:
        ret, frame = camera.read()

        if not ret:
            st.error("Failed to capture video. Check your camera connection.")
            break

        # Check if the frame is valid before processing
        if frame is not None:
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

            if frame_rgb is not None:
                subheading1_placeholder.markdown("<h3 class='subheader'>Live Video Feed</h3>", unsafe_allow_html=True)
                FRAME_WINDOW.image(frame_rgb, caption="Live Feed", use_column_width=False)
                subheading2_placeholder.markdown("<h3 class='subheader'>Real-Time Metrics</h3>", unsafe_allow_html=True)

                # Process frame metrics
                result, eye_d, head_d, fps, obj_d, alert_msg = process_frame(camera)

                if not result:
                    violation_count += 1
                    warning_count_placeholder.markdown(f"<p class='warning'>Warnings: {violation_count}</p>", unsafe_allow_html=True)

                    if violation_count == 1:
                        warning_placeholder1.warning(alert_msg)  # Display warning in sidebar
                        # speak(alert_msg)
                    elif violation_count == 2:
                        warning_placeholder2.warning(alert_msg)  # Display warning in sidebar
                        # speak(alert_msg)

                    if violation_count == 3:
                        warning_placeholder3.warning(alert_msg)
                        # speak(alert_msg)
                        # speak("This is the last warning, after this, your exam will be terminated.")

                    # If 4th warning is triggered, stop the camera automatically
                    if violation_count == 4:
                        camera_active = False  # Auto-stop camera after 4th warning
                        break

                else:
                    # Update real-time metrics with card-based UI
                    fps_placeholder.markdown(f"""
                        <div class="metric-card fps-card">
                            <h2>FPS</h2>
                            <p>{fps:.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    eye_placeholder.markdown(f"""
                        <div class="metric-card eye-card">
                            <h2>Eye Direction</h2>
                            <p>{eye_d}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    head_placeholder.markdown(f"""
                        <div class="metric-card head-card">
                            <h2>Head Direction</h2>
                            <p>{head_d}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    obj_placeholder.markdown(f"""
                        <div class="metric-card obj-card">
                            <h2>Background</h2>
                            <p>{"Ok" if obj_d else "Object detected"}</p>
                        </div>
                    """, unsafe_allow_html=True)

        # Stop if the stop button is pressed
        if stop_button:
            camera_active = False
            break

        # Add a short delay to reduce CPU usage
        time.sleep(0.01)

    # Clear all video frames, metrics, and status after camera is stopped
    FRAME_WINDOW.empty()
    fps_placeholder.empty()
    eye_placeholder.empty()
    head_placeholder.empty()
    obj_placeholder.empty()
    camera_status_placeholder.empty()  # Clear the camera status message
    warning_placeholder1.empty() 
    warning_placeholder2.empty()
    warning_placeholder3.empty() # Clear the warning messages
    warning_placeholder4.empty()
    if violation_count == 4:
        st.warning("The exam has been terminated. Please contact the administrator.")
        # speak("The exam has been terminated. Please contact the administrator.")

    # Cleanup resources
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
