import cv2
import streamlit as st
import tempfile
import time
import os
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

def process_frame(video_cap):
    # Call the function that processes the video feed and returns metrics
    result, eye_d, head_d, fps, obj_d, alert_msg = run(video_cap)
    return result, eye_d, head_d, fps, obj_d, alert_msg

def main():
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    if 'tmp_file_path' not in st.session_state:
        st.session_state.tmp_file_path = None

    st.markdown("<h1 class='main-title'>Examiner.AI:<br> AI Proctored Exam System </h1>", unsafe_allow_html=True)

    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        start_button = st.button("Start Processing")
        stop_button = st.button("Stop/Reset")

        # Placeholder for warnings below the buttons
        warning_count_placeholder = st.empty() 
        warning_placeholder1 = st.empty()
        warning_placeholder2 = st.empty()
        warning_placeholder3 = st.empty()
        warning_placeholder4 = st.empty()

    # Initialize a flag to manage the initial message and layout
    video_active = False
    violation_count = 0

    # Main container
    with st.container():
        # Placeholder for the camera status message
        video_status_placeholder = st.empty()
        subheading1_placeholder = st.empty()
        subheading2_placeholder = st.empty()

        # Only show the initial message if the start button has not been clicked
        if not video_active:
            subheading1_placeholder.markdown("<p class='camera-status'>Upload a video file and press 'Start Processing'.</p>", unsafe_allow_html=True)

        # Initialize placeholders for the video feed and metrics
        with st.expander("Video Feed", expanded=video_active):
            FRAME_WINDOW = st.image([])  # Placeholder for the video feed

        with st.expander("Real-Time Metrics", expanded=video_active):
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            with c1:
                fps_placeholder = st.empty()
            with c2:
                eye_placeholder = st.empty()
            with c3:
                head_placeholder = st.empty()
            with c4:
                obj_placeholder = st.empty()

        if start_button and uploaded_file:
            # Hide the initial message and update video status
            st.session_state.uploaded_file = uploaded_file
            subheading1_placeholder.empty()
            video_status_placeholder.markdown("<p class='camera-status'>Processing video file. Press 'Stop/Reset' to stop.</p>", unsafe_allow_html=True)
            video_active = True

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.tmp_file_path = tmp_file.name

            # Initialize VideoCapture with the temporary file path
            video_cap = cv2.VideoCapture(st.session_state.tmp_file_path)

            while video_active:
                try:
                    ret, frame = video_cap.read()

                    if not ret:
                        if violation_count < 4:
                            st.success("Exam completed successfully.")
                        video_active = False
                        break
                except Exception as e:
                    st.error(f"Error during video processing: {e}")

                # Process frame metrics
                result, eye_d, head_d, fps, obj_d, alert_msg = process_frame(video_cap)

                # Convert frame to RGB and display
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.flip(frame_rgb, 1)
                    frame_rgb = cv2.resize(frame_rgb, (int(frame_rgb.shape[1] * 1.0), int(frame_rgb.shape[0] * 1.0)))

                    FRAME_WINDOW.image(frame_rgb, caption="Processed Feed", use_column_width=True)

                    if not result:
                        violation_count += 1
                        warning_count_placeholder.markdown(f"<p class='warning'>Warnings: {violation_count}</p>", unsafe_allow_html=True)

                        if violation_count == 1:
                            warning_placeholder1.warning(alert_msg)  # Display warning
                        elif violation_count == 2:
                            warning_placeholder2.warning(alert_msg)  # Display warning
                        elif violation_count == 3:
                            warning_placeholder3.warning(alert_msg)
                        
                        # If 4th warning is triggered, stop processing automatically
                        if violation_count == 4:
                            video_active = False  # Auto-stop video processing after 4th warning
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
                    video_active = False
                    break

                # Add a short delay to reduce CPU usage
                time.sleep(0.01)

            # Clear all video frames, metrics, and status after processing is stopped
            FRAME_WINDOW.empty()
            fps_placeholder.empty()
            eye_placeholder.empty()
            head_placeholder.empty()
            obj_placeholder.empty()
            video_status_placeholder.empty()  # Clear the video status message
            warning_placeholder1.empty() 
            warning_placeholder2.empty()
            warning_placeholder3.empty()  # Clear the warning messages
            warning_placeholder4.empty()
            
            # Show message if terminated by warnings
            if violation_count == 4:
                st.warning("The exam has been terminated. Please contact the administrator.")

            # Cleanup resources
            video_cap.release()
            if st.session_state.tmp_file_path and os.path.exists(st.session_state.tmp_file_path):
                os.remove(st.session_state.tmp_file_path)
                # st.info("Temporary video file deleted.")
                st.session_state.tmp_file_path = None

            # Remove uploaded file from session state
            if st.session_state.uploaded_file:
                st.session_state.uploaded_file = None
                st.info("Uploaded video file removed.")
                # st.info("Temporary video file deleted.")


if __name__ == "__main__":
    main()