# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:34:33 2023

@author: chaitanya
"""


from flask import Flask, render_template, request, redirect, url_for
from googletrans import Translator
from gtts import gTTS
import os
import cv2
import pytesseract
from PyPDF2 import PdfReader
from docx import Document
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import io
import base64
import nltk
import json
import fitz
nltk.download('punkt')

app = Flask(__name__)

# Set the path to the Tesseract executable (update this path based on your installation)
pytesseract.pytesseract.tesseract_cmd = r"F:\New folder\tesseract.exe"

LANGUAGE_MAPPING = {
    'telugu': 'te',
    'english': 'en',
    'assamese': 'as',
    'bengali': 'bn',
    'bodo': 'brx',
    'dogri': 'doi',
    'gujarati': 'gu',
    'hindi': 'hi',
    'kannada': 'kn',
    'kashmiri': 'ks',
    'konkani': 'kok',
    'maithili': 'mai',
    'malayalam': 'ml',
    'manipuri': 'mni',
    'marathi': 'mr',
    'nepali': 'ne',
    'odia':'or-IN',
    'punjabi': 'pa',
    'sanskrit': 'sa',
    'santali': 'sat',
    'sindhi': 'sd',
    'tamil': 'ta',
    'telugu': 'te',
    'urdu': 'ur',
}

def extract_text_from_image(scanned_image):
    gray_image = cv2.cvtColor(scanned_image, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray_image)
    print("Extracted Text from Image:")
    print(text)
    return text
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)

        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()

        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_text_file(text_file_path):
    with open(text_file_path, 'r') as file:
        text = file.read()
        print("Text from Text File:")
        print(text)
        return text

def extract_text_from_word_document(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    print("Text from Word Document:")
    print(text)
    return text

def summarize_text_sumy(text, sentences_count):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=sentences_count)
    summary_text = " ".join(str(sentence) for sentence in summary)
    print("Summary:")
    print(summary_text)
    return summary_text

def translate_text(text, target_language='en'):
    print("Text:", text)
    print("Target Language:", target_language)

    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    print(translation.text)

    return translation.text

def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language, slow=False)
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    encoded_audio = base64.b64encode(audio_data.read()).decode('utf-8')
    return encoded_audio


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    user_choice = request.form.get('user_choice')
    if user_choice == 'camera':
        return redirect(url_for('camera'))
    elif user_choice == 'file':
        return redirect(url_for('file'))
    else:
        return "Invalid source option. Please choose 'camera' or 'file'."

@app.route('/camera')
def camera():
    return render_template('camera1.html', language_mapping=LANGUAGE_MAPPING)

@app.route('/camera_processing', methods=['POST'])
def camera_processing():
    if request.method == 'POST':
        print(request.form)  # Add this line to print form data
        camera_index = int(request.form.get('camera_index', 0))  # Provide a default value
        sentences_count = int(request.form.get('sentences_count', 3))
        target_language = request.form.get('target_language')
        translation_choice = request.form['translation_choice']
        cap = cv2.VideoCapture(camera_index)  # Use the selected camera index

        if not cap.isOpened():
            return "ERROR: Could not open the camera"

        while True:
            ret, frame = cap.read()
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Press 'c' to capture an image
                cv2.imwrite("scanned_image.png", frame)
                print("Image captured successfully as 'scanned_image.png'.")
                scanned_image = cv2.imread("scanned_image.png")
                scanned_image = cv2.cvtColor(scanned_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                text = extract_text_from_image(scanned_image)
                
                break

        # Display the extracted text on the result page
        summarized_text = summarize_text_sumy(text, sentences_count)
        translated_text = translate_text(summarized_text, target_language) if translation_choice == 'yes'  else translate_text(text, target_language)

        # Generate audio data
        audio_data = text_to_speech(translated_text, language=target_language)

        return render_template('result.html', extracted_text=text, summarized_text=summarized_text, translated_text=translated_text, audio_data=audio_data, language_mapping=LANGUAGE_MAPPING)
    else:
        return "Error: No file provided."

@app.route('/file')
def file():
    return render_template('file.html', language_mapping=LANGUAGE_MAPPING)

@app.route('/process_file', methods=['POST'])
def process_file():
    file = request.files['file']
    sentences_count = int(request.form['sentences_count'])
    target_language = request.form['target_language']
    translation_choice = request.form['translation_choice']

    if file:
        # Save the file temporarily
        file_path = "temp_file" + os.path.splitext(file.filename)[-1]
        file.save(file_path)

        file_extension = os.path.splitext(file_path)[-1].lower()

        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension == '.txt':
            text = extract_text_from_text_file(file_path)
        elif file_extension == '.docx':
            text = extract_text_from_word_document(file_path)
        else:
            os.remove(file_path)  # Remove the temporary file
            return f"Unsupported file format: {file_extension}"

        os.remove(file_path)  # Remove the temporary file

        summarized_text = summarize_text_sumy(text, sentences_count)
        translated_text = translate_text(summarized_text, target_language) if translation_choice == 'yes' else translate_text(text,target_language)

        # Generate audio data
        audio_data = text_to_speech(translated_text, language=target_language)

        return render_template('result.html', extracted_text=text, summarized_text=summarized_text, translated_text=translated_text, audio_data=audio_data, language_mapping=LANGUAGE_MAPPING)
    else:
        return "Error: No file provided."

if __name__ == "__main__":
    app.run()
