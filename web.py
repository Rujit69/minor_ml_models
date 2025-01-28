import streamlit as st
from predict import predict_image, load_model
import tempfile
import os
import uuid

# Load the model once when the app starts
model, device = load_model()

# Streamlit UI to upload an image
st.markdown("# AI Generated Face Detector")
st.markdown("### Upload an image of a face to see if it is real or fake!")

# Initialize session state to store uploaded files and predictions
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

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

    # Display the uploaded image
    st.image(temp_file_path, caption='Uploaded Image.', use_container_width=True)

    # Add a predict button for this specific file
    if st.button(f"Predict for {uploaded_file.name}") and file_id in st.session_state.uploaded_files:
        temp_file_path = st.session_state.uploaded_files[file_id]  # Retrieve file path from session state

        try:
            # Perform prediction and get confidence score
            prediction, confidence = predict_image(temp_file_path)
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
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)  # Delete the temporary file
            except Exception as e:
                st.error(f"❌ Error deleting file: {e}")
            if file_id in st.session_state.uploaded_files:
                del st.session_state.uploaded_files[file_id]