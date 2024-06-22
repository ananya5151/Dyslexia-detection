

import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO

# Load the YOLOv8n-cls model
model = YOLO("best10.pt")

# Function to perform object detection on the image
def perform_object_detection(image):
    try:
        # Preprocess the image
        image = image.resize((224, 224))  # Resize the image to the required size
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1).div(255.0).unsqueeze(0)

        # Perform object detection
        results = model(image)[0]

        return results
    except Exception as e:
        st.error(f"Error during object detection: {e}")
        return None

# Function to check if the image is dyslexic or not
def check_dyslexia(image):
    try:
        # Perform object detection on the image
        results = perform_object_detection(image)

        # Check if the object detection was successful
        if results.boxes is not None:
            if results is not None and hasattr(results, 'boxes') and len(results.boxes) > 0:
                # Assuming the first detection is the most relevant
                # You can modify this logic based on your specific requirements
                if results.boxes[0].cls[0] == 0:
                    return True
                else:
                    return False
            else:
                return False
            
        # if is_dyslexic:
        #     st.write("**The image is dyslexic.**")
        #     st.write(f"Dyslexic probability: {results.boxes[0].cls[0]:.2f}")
        # else:
        #     st.write("**The image is not dyslexic.**")
        #     st.write(f"Non-dyslexic probability: {1 - results.boxes[0].cls[0]:.2f}")


    except Exception as e:
        st.error(f"Error during dyslexia detection: {e}")
        return False

# Streamlit app
st.set_page_config(page_title="Dyslexia Webapp")

hide_menu_style = """
<style>
#MainMenu {visibility: hidden; }
footer {visibility: hidden; }
</style>
"""

st.markdown(hide_menu_style, unsafe_allow_html=True)
st.header("Dyslexia Web APP")

tab1, tab2 = st.tabs(["Home", "Writing"])

with tab1:
    st.header("Home Page")
    st.write("""
    Dyslexia is a learning disorder that involves difficulty reading due to problems identifying 
    speech sounds and learning how they relate to letters and words (decoding). Also called a 
    reading disability, dyslexia is a result of individual differences in areas of the brain that 
    process language.

Dyslexia is not due to problems with intelligence, hearing or vision. Most children with dyslexia 
can succeed in school with tutoring or a specialized education program. Emotional support also plays 
an important role.

Though there's no cure for dyslexia, early assessment and intervention result in the best outcome. 
Sometimes dyslexia goes undiagnosed for years and isn't recognized until adulthood, but it's never 
too late to seek help.""")

    st.subheader("Dyslexia- India")
    st.write("""
With regard to sociodemographic variables of primary school students, majority of the students 
56 (56%) belong to the age group of 6 years and 44 (44%) were 7 years. On gender, 57 (57%) were 
female and 43 (43%) were male. With regard to the religion, 88 (88%) were Hindu, 8 (8%) were 
Muslims, and 4 (4%) were Christians. With respect to occupational status of father, majority were 
private employee (47%), daily wages 39%, government employee 10%, and business 4%. Regarding the 
occupational status of mother, most of them were housewife (75%), daily worker 15%, private employee 9%, and government employee 1%.

Among the 100 samples, 50% were selected from I standard and another 50% were selected from II 
standard. With respect to the place of residence, 51 (51%) are from urban area and 49 (49%) are 
from rural area. In terms of language spoken by them Majority of the primary school students 95 
(95%) of them were speaking Kannada commonly at home and 05 (05%) of them were speaking Telugu at 
home. The entire primary school students, i.e., 100 (100%) of them, are speaking English at school. 
In connection with the data on their academic performance, 50 (50%) are having average academic performance, 44 (44%) are having good, 
and 6 (6%) are having excellent academic performance.""")

with tab2:
    st.title("Dyslexia Detection App")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Dyslexia"):
            is_dyslexic = check_dyslexia(image)
            # if is_dyslexic:
            #     st.write("**The image is dyslexic.**")
            # else:
            #     st.write("**The image is not dyslexic.**")
            if is_dyslexic:
                st.write("**The image is dyslexic.**")
                st.write(f"Dyslexic probability: {results.boxes[0].cls[0]:.2f}")
            else:
                st.write("**The image is not dyslexic.**")
                st.write(f"Non-dyslexic probability: {1 - results.boxes[0].cls[0]:.2f}")




