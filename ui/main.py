import streamlit as st

import os
import sys
from pathlib import Path
from PIL import Image

sys.path.append(os.path.abspath('.'))

from report.create_report import generate_report

dir_tmp = Path('.') / 'tmp'

if not dir_tmp.exists():
    dir_tmp.mkdir(parents=True)

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
    if uploaded_file is not None:
        st.write('You selected: ', model)
        with open(dir_tmp / 'image', 'wb') as f:
            f.write(uploaded_file.getvalue())
        generate_report(model, uploaded_file)
        with open("output/main.pdf", "rb") as file:
            btn = st.download_button(
                label="Download Report",
                data=file,
                file_name="report.pdf",
                mime="application/pdf"
            )
    else:
        st.write('Please select the image first!')
