import streamlit as st
from pathlib import Path
from PIL import Image
import os
import sys

sys.path.insert(1, os.getcwd())

from report.create_report import generate_report

# Define a function to ensure better separation of concerns
def generate_prediction_report(uploaded_file, model):
    dir_tmp = Path('.') / 'tmp'
    dir_tmp.mkdir(parents=True, exist_ok=True)

    if uploaded_file is not None:
        st.write('You selected:', model)

        # Save the uploaded image to a file
        with open(dir_tmp / 'image', 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Generate the report based on the selected model and uploaded image
        predicted_class = generate_report(model, uploaded_file)
        st.write(f"Predicted Class: {predicted_class[0]}")

        # Provide a link to download the generated report
        report_path = 'output/main.pdf'
        if os.path.exists(report_path):
            with open(report_path, "rb") as file:
                btn = st.download_button(
                    label="Download Report",
                    data=file,
                    file_name="report.pdf",
                    mime="application/pdf"
                )
        else:
            st.write("Report generation failed.")
    else:
        st.write('Please select an image first!')

# Streamlit web app begins here
st.title('Bird Prediction on Image')

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:
    st.image(Image.open(uploaded_file))

model = st.selectbox(
    'Which model would you like to use for prediction',
    ['Random Forest', 'SVM', 'CNN']
)

predict = st.button("Predict", type="primary")

if predict:
    generate_prediction_report(uploaded_file, model)
