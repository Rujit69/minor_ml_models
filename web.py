import streamlit as st
from predict import predict_image, load_model
import tempfile
import os
import uuid
from streamlit_cropper import st_cropper
from PIL import Image

# Load the model once when the app starts
model, device = load_model()

# Streamlit UI to upload an image
st.markdown("# AI Generated Face Detector")
st.markdown("### Upload an image of a face to see if it is real or fake!")

# Initialize session state to store uploaded files and predictions
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "show_crop" not in st.session_state:
    st.session_state.show_crop = False  # Initialize a flag for cropping

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Generate a unique ID for the uploaded file
    file_id = str(uuid.uuid4())

    # Create a NamedTemporaryFile that will be automatically deleted when closed
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    # Store the file path in session state
    st.session_state.uploaded_files[file_id] = temp_file_path

    # Load the image for processing
    img = Image.open(temp_file_path)

    # Convert RGBA images to RGB to ensure compatibility
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    # Display the uploaded image
    st.image(img, caption='Uploaded Image.', use_container_width=True)

    # Button to toggle cropping functionality
    if st.button("Crop Image"):
        st.session_state.show_crop = True

    # If user clicks the button, display cropping interface
    if st.session_state.show_crop:
        # Image cropping UI
        st.markdown("### Crop the image before making a prediction")
        cropped_img = st_cropper(img, realtime_update=True, box_color="blue", aspect_ratio=(1, 1))

        # Convert the cropped image to a temporary file
        cropped_temp_path = f"{tempfile.gettempdir()}/{file_id}_cropped.jpg"
        cropped_img.save(cropped_temp_path)

        # Use the cropped image for prediction
        prediction_img_path = cropped_temp_path

        # Display the cropped image
        st.image(cropped_img, caption='Cropped Image', use_container_width=True)
    else:
        # Use the original image for prediction if cropping is skipped
        prediction_img_path = temp_file_path

    # Add a predict button for this specific file
    if st.button(f"Predict for {uploaded_file.name}") and file_id in st.session_state.uploaded_files:
        try:
            # Perform prediction on the selected image and get confidence score
            prediction, confidence = predict_image(prediction_img_path)
            confidence_percentage = confidence * 100  # Convert to percentage

            if prediction == "real":
                st.success(f"✅ Face is: {prediction} with {confidence_percentage:.2f}% confidence.")
            else:
                st.error(f"❌ Face is AI generated: {prediction} with {confidence_percentage:.2f}% confidence.")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
        finally:
            # Ensure the file is properly deleted after prediction
            try:
                if os.path.exists(prediction_img_path):
                    os.remove(prediction_img_path)  # Delete the temporary file
            except Exception as e:
                st.error(f"❌ Error deleting file: {e}")
            if file_id in st.session_state.uploaded_files:
                del st.session_state.uploaded_files[file_id]
