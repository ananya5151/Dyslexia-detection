import streamlit as st
from PIL import Image
import os
from textblob import TextBlob
import language_tool_python
import requests
import pandas as pd
import random
import speech_recognition as sr
import time
import eng_to_ipa as ipa


from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

from abydos.phonetic import Soundex, Metaphone, Caverphone, NYSIIS

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


# image to text API authentication
subscription_key_imagetotext = "3028fbe3d1f248a9a3e32b4d583f2eab"
endpoint_imagetotext = "https://aaaaaaaaaaa---aakriti.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(
    endpoint_imagetotext, CognitiveServicesCredentials(subscription_key_imagetotext))

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# text correction API authentication
api_key_textcorrection = "e64c5dbabf79474f8fc40e960c7fc7f0"
endpoint_textcorrection = "https://api.bing.microsoft.com/v7.0/SpellCheck"

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

my_tool = language_tool_python.LanguageTool('en-US')

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# method for extracting the text


def image_to_text(path):
    read_image = open(path, "rb")
    read_response = computervision_client.read_in_stream(read_image, raw=True)
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower() not in ['notstarted', 'running']:
            break
        time.sleep(5)

    text = []
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                text.append(line.text)

    return " ".join(text)

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# method for finding the spelling accuracy


def spelling_accuracy(extracted_text):
    spell_corrected = TextBlob(extracted_text).correct()
    return ((len(extracted_text) - (levenshtein(extracted_text, spell_corrected)))/(len(extracted_text)+1))*100

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# method for gramatical accuracy


def gramatical_accuracy(extracted_text):
    spell_corrected = TextBlob(extracted_text).correct()
    correct_text = my_tool.correct(spell_corrected)
    extracted_text_set = set(spell_corrected.split(" "))
    correct_text_set = set(correct_text.split(" "))
    n = max(len(extracted_text_set - correct_text_set),
            len(correct_text_set - extracted_text_set))
    return ((len(spell_corrected) - n)/(len(spell_corrected)+1))*100

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# percentage of corrections


def percentage_of_corrections(extracted_text):
    data = {'text': extracted_text}
    params = {
        'mkt': 'en-us',
        'mode': 'proof'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Ocp-Apim-Subscription-Key': api_key_textcorrection,
    }
    response = requests.post(endpoint_textcorrection,
                             headers=headers, params=params, data=data)
    json_response = response.json()
    return len(json_response['flaggedTokens'])/len(extracted_text.split(" "))*100

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


def get_feature_array(path: str):
    feature_array = []
    extracted_text = image_to_text(path)
    feature_array.append(spelling_accuracy(extracted_text))
    feature_array.append(gramatical_accuracy(extracted_text))
    feature_array.append(percentage_of_corrections(extracted_text))
    return feature_array

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


def generate_csv(folder: str, label: int, csv_name: str):
    arr = []
    for image in os.listdir(folder):
        path = os.path.join(folder, image)
        feature_array = get_feature_array(path)
        feature_array.append(label)
        # print(feature_array)
        arr.append(feature_array)
        print(feature_array)
    print(arr)
    pd.DataFrame(arr, columns=["spelling_accuracy", "gramatical_accuracy", " percentage_of_corrections",
                  "presence_of_dyslexia"]).to_csv("test1.csv")

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


def score(input):
    if input[0] <= 96.40350723266602:
        var0 = [0.0, 1.0]
    else:
        if input[1] <= 99.1046028137207:
            var0 = [0.0, 1.0]
        else:
            if input[2] <= 2.408450722694397:
                if input[2] <= 1.7936508059501648:
                    var0 = [1.0, 0.0]
                else:
                    var0 = [0.0, 1.0]
            else:
                var0 = [1.0, 0.0]
    return var0

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# deploying the model


st.set_page_config(page_title="Dyslexia Webapp")

hide_menu_style = """
<style>
#MainMenu {visibility: hidden; }
footer {visibility: hidden; }
</style>
"""


st.markdown(hide_menu_style, unsafe_allow_html=True)
st.header("Dyslexia Web APP")

tab1, tab2, tab3 = st.tabs(["Home", "Writing", "About"])

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

    img1 = Image.open("images\img1.jpg")
    st.image(img1)

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
    st.title("   Dyslexia Detection Using Handwriting Samples")
    st.write("This is a simple web app that works based on machine learning techniques. This application can predict the presence of dyslexia from the handwriting sample of a person.")
    with st.container():
        st.write("---")
        image = st.file_uploader("Upload the handwriting sample that you want to test", type=["jpg"])
        if image is not None:
            st.write("Please review the image selected")
            st.write(image.name)
            image_uploaded = Image.open(image)
            image_uploaded.save("temp.jpg")
            st.image(image_uploaded, width=224)

        if st.button("Predict", help="click after uploading the correct image"):
            try:
                feature_array = get_feature_array("temp.jpg")
                result = score(feature_array)
                if result[0] == 1:
                    st.write("From the tests on this handwriting sample there is very slim chance that this person is suffering from dyslexia or dysgraphia")
                else:
                    st.write("From the tests on this handwriting sample there is very high chance that this person is suffering from dyslexia or dysgraphia")
            except:
                st.write("Something went wrong at the server end please refresh the application and try again")



with tab3:
    st.header("About APP")
    st.write("""
    Dyslexia, also known as reading disorder, is a disorder characterized by reading below the expected level for ones age. 
    Different people are affected to different degrees.
    The common symptoms include: Frequently making the same kinds of mistakes, like reversing letters, Having poor spelling, like spelling the same word correctly and 
    incorrectly in the same exercise, Having trouble remembering how words are spelled and applying spelling rules in writing, etc.

    Based on the spelling, grammatic, contextual and phonetics error the app predicts whether the person with the wrting has 
    dyslexia or not. 
    """)
    st.subheader("Average corrections is less for a non-dyslexic child when compared to dyslexic child")
    st.image("images\percentage_of_corrections.jpg")
    
    st.subheader("Spelling accuracy for a dyslexic and a non-dyslexic child")
    st.image("images\spelling_accuracy.jpg")
    

