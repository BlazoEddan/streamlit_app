import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import os
from datetime import datetime


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


SAVE_FOLDER = "saved_frames"
os.makedirs(SAVE_FOLDER, exist_ok=True)


st.title("🎥 Live Object Detection & Tracking")
st.write("Real-time AI object detection using YOLOv8")

st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.4
)

save_images = st.sidebar.checkbox("Save Detected Frames")

# Alert object selection
alert_object = st.sidebar.selectbox(
    "Trigger Alert For:",
    ["person", "cell phone", "bottle", "laptop", "chair"]
)


def video_frame_callback(frame):

    img = frame.to_ndarray(format="bgr24")

    results = model.predict(
        source=img,
        conf=confidence,
        verbose=False
    )

    annotated_frame = img.copy()

    total_objects = 0
    person_count = 0
    alert_detected = False


    for result in results:

        boxes = result.boxes

        if boxes is not None:

            for box in boxes:

                # Get class ID
                cls_id = int(box.cls[0])

                # Label name
                label = model.names[cls_id]

                # Confidence
                conf = float(box.conf[0])

                total_objects += 1

                # Count persons
                if label == "person":
                    person_count += 1

                # Trigger alert
                if label == alert_object:
                    alert_detected = True

                # Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw label
                cv2.putText(
                    annotated_frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


    cv2.putText(
        annotated_frame,
        f"Total Objects: {total_objects}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )



    cv2.putText(
        annotated_frame,
        f"People Count: {person_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    if alert_detected:

        cv2.putText(
            annotated_frame,
            f"ALERT: {alert_object.upper()} DETECTED!",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )


        if save_images:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = os.path.join(
                SAVE_FOLDER,
                f"detected_{timestamp}.jpg"
            )

            cv2.imwrite(filename, annotated_frame)

    return av.VideoFrame.from_ndarray(
        annotated_frame,
        format="bgr24"
    )


webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)
