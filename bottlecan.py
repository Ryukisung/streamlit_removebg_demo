import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np

# Page Configuration
st.set_page_config(
    layout="wide",
    page_title="Bottle and Can Detection"
)

# Custom CSS
st.markdown("""
    <style>
    /* Background color */
    .stApp {
        background-color: #D1C7BD;
    }
    /* Primary color for buttons and titles */
    .css-1d391kg, .css-1vbd788, .st-b7, .st-df, .st-bq, .st-cq, .st-cp, .st-cl, .st-cm, .st-co, .st-cn, .st-bz, .st-bn {
        color: #AC9C8D !important;
    }
    .css-1d391kg, .css-1vbd788, .st-b7, .st-df, .st-bq, .st-cq, .st-cp, .st-cl, .st-cm, .st-co, .st-cn, .st-bz, .st-bn {
        border-color: #F0F8FF !important;
    }
    .css-1d391kg, .css-1vbd788, .st-b7, .st-df, .st-bq, .st-cq, .st-cp, .st-cl, .st-cm, .st-co, .st-cn, .st-bz, .st-bn {
        background-color: #E0FFFF !important;
    }
    /* Text color */
    .css-1d391kg, .css-1vbd788, .st-b7, .st-df, .st-bq, .st-cq, .st-cp, .st-cl, .st-cm, .st-co, .st-cn, .st-bz, .st-bn {
        color: #333333 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title("Bottle and Can Detection")
st.markdown("""
    **Detect Bottles and Cans in Your Image**

    :cat: Try uploading an image to watch the model detect bottles and cans.
    Full quality images can be downloaded from the sidebar.
    :grin:
""")

# Sidebar
st.sidebar.header("Upload and Download :gear:")
st.sidebar.write(
    "Upload an image to detect bottles and cans, and download the processed image.")

# Set maximum file size to 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Load the YOLO model
model = YOLO("yolov8m_custom.pt")  # Replace with your custom model path


def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def detect_objects(upload):
    try:
        # Ensure the file is read as bytes
        image_bytes = upload.read()

        # Open the image using PIL
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)

        # Ensure the image has 3 channels (RGB)
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        results = model(image_np)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Draw bounding box
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, f"{cls} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        col1.write("**Original Image :camera:**")
        col1.image(image, use_column_width=True)

        output_image = Image.fromarray(image_np)
        col2.write("**Output Image :wrench:**")
        col2.image(output_image, use_column_width=True)
        st.sidebar.markdown("\n")
        st.sidebar.download_button("Download detected image", convert_image(
            output_image), "detected.png", "image/png")

    except Exception as e:
        st.error(f"Error processing image: {e}")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(
            "The uploaded file is too large. Please upload an image smaller than 10MB.")
    else:
        detect_objects(upload=my_upload)
else:
    # Display a default example image
    try:
        with open("4.jpeg", "rb") as f:
            example_image_bytes = f.read()
            detect_objects(upload=BytesIO(example_image_bytes))
    except FileNotFoundError:
        st.error("Example image not found. Please upload an image.")
