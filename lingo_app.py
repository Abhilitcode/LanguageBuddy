import streamlit as st 
import joblib
import re
import requests

#load all model, tokenizer and labelencoder
model = joblib.load('multinomialnb.pkl')
cv = joblib.load('your_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

#fucntion to display the meaning of sentence in English for users
# Function to display the meaning of the sentence in English for users
def translate_text(text, target_language='en'):
    if text:
        url = "https://translateai.p.rapidapi.com/google/translate/text"
        
        payload = {
            "target_language": target_language,
            "origin_language": "auto",  # Auto-detect source language
            "words_not_to_translate": "Para; Experimenta",  # Words not to translate
            "input_text": text
        }
        
        headers = {
            "x-rapidapi-key": "23a008335cmsh83fe99128fbf62bp167056jsn17a9baaa82eb",  # Replace with your actual API key
            "x-rapidapi-host": "translateai.p.rapidapi.com",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('translation', 'Translation not found')
        else:
            return f"Error: {response.status_code}"
        
# Sidebar for language information
with st.sidebar:
    st.title("Supported Languages")
    st.write("""
    Currently, the app supports a total of **19 languages** for detection and translation:
    - Marathi
    - English
    - Gujarati
    - French
    - Spanish
    - Portuguese
    - Italian
    - Russian
    - Hindi
    - Swedish
    - Malayalam
    - Dutch
    - Arabic
    - Turkish
    - German
    - Tamil
    - Danish
    - Kannada
    - Greek
    """)

#input on interface
st.title("Language Buddy 🌍")
st.write("Welcome to Language Buddy! This app detects the language of the text you input and provides an English translation.")
st.write("Enter a sentence, and the app will predict the language and provide its translation.")

#USER INPUT
input = st.text_input("Enter the text:")

#predict button
if st.button("Predict Language"):
    if input.strip() != "":
        text = re.sub(r'[!@#$(),"%^*?:;~`0-9।]',' ', input)
        text = re.sub(r'[[]]', ' ', text)
        text = text.lower()
        
        x = cv.transform([text])
        lang = model.predict(x)
        #le.inverse_transform(lang) converts the predicted numerical label back into the original string label (e.g., 'English', 'Hindi'). This is important for displaying human-readable predictions instead of just numerical outputs.
        # Inverse transform the predicted label to get the original language name
        lang = le.inverse_transform(lang)
        
        #display on production
        st.success(f"The predicted language is {lang[0]}")
        
        translate_text = translate_text(input)
        # Display translated text using markdown for better formatting
        st.markdown(f"### Translation in English: \n\n**{translate_text}**")
    else:
        st.warning("Please enter some text to predict!")
    
