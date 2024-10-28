

# FLASK Server:
# ------------

# This code to receive a single sentence and predict the output

import json
import nltk
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
import joblib
from keras.preprocessing.sequence import pad_sequences
import re
import string
from keras.preprocessing.text import Tokenizer
import csv
import numpy as np

app = Flask(__name__)

output_file_path = '/Users/sara/Desktop/Master/Thesis Project/FLASK/output.csv'
MLP_model = joblib.load('Sara_MLP_model.joblib')

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tweets_list = []
labels_list = []
with open(output_file_path, mode='r') as input_file:
    reader = csv.reader(input_file)
    next(reader)
    for row in reader:
        tweets_list.append(row[0])
        labels_list.append(int(row[1]))


tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(tweets_list)
sequences = tokenizer.texts_to_sequences(tweets_list)
padded_sequences = pad_sequences(sequences, padding='post', truncating='post')
labels = np.array(labels_list)


def remove_stopwords(sentence):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    custom_words = ['might', 'ima', 'amp', 'aint', 'never', 'always', 'plz', 'thats']
    stop_words.update(custom_words)
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    processed_sentence = ' '.join(filtered_words)

    return processed_sentence


def clean_sentence(sentence):
    processed_sentence = re.sub(r'@[a-zA-Z0-9_]+', '', sentence)
    processed_sentence = re.sub(r'#', '', processed_sentence)
    processed_sentence = re.sub(r'http\S+', '', processed_sentence)
    translator = str.maketrans('', '', string.punctuation)
    processed_sentence = processed_sentence.translate(translator)
    translator = str.maketrans('', '', '0123456789')
    processed_sentence = processed_sentence.translate(translator)
    processed_sentence = processed_sentence.strip()
    processed_sentence = remove_stopwords(processed_sentence)
    processed_sentence = re.sub(r'RT\s+', '', processed_sentence)
    processed_sentence = processed_sentence.lower()
    processed_sentence = re.sub(r'[^a-zA-Z\s]', '', processed_sentence)
    processed_sentence = processed_sentence.replace('""', '')
    processed_sentence = re.sub(r'\s+', ' ', processed_sentence).strip()

    return processed_sentence


def make_prediction(cleaned_sentence):

    # Tokenize and pad the new sentence
    new_sequence = tokenizer.texts_to_sequences([cleaned_sentence])
    print(f"Cleaned Sentence: {cleaned_sentence}")
    new_padded_sequence = pad_sequences(new_sequence, padding='post', truncating='post',
                                        maxlen=padded_sequences.shape[1])

    average = 0
    for i in range(5):
        probability_prediction = MLP_model.predict(new_padded_sequence)
        print(f"probability {i}: {probability_prediction}")
        average += probability_prediction

    average /= 5
    threshold = 0.5
    if average > threshold:
        class_ = "appropriate"
    else:
        class_ = "inappropriate"

    return class_, average







@app.route('/receive_text_dictionary', methods=['POST'])
def receive_data():
    data = request.json
    json_text_dictionary = data.get('json_text_dictionary')
    json_data = json.loads(json_text_dictionary)
    for key, value in json_data.items():
        sentence = value["text_content"]
        cleaned_sentence = clean_sentence(sentence)
        sentence_prediction, average = make_prediction(cleaned_sentence)
        print("---------------------------------------------------------------------------")
        print(f"Sentence: {sentence}, Probability: {average}, Class: {sentence_prediction}")
        print("---------------------------------------------------------------------------")
    return jsonify({'message': 'Data received by FLASK successfully'}), 200





if __name__ == '__main__':
    app.run(debug=True)

