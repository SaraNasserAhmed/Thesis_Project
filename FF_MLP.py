
# FF_MLP Model:
# ------------

import string
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
# For visualizing the outout accuracy:
import matplotlib.pyplot as plt

#for saving the trained model:
import joblib


def remove_stopwords(tweet):

    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    # Add custom words to the list of stopwords:
    custom_words = ['might', 'ima', 'amp', 'aint', 'never', 'always', 'plz', 'thats']
    stop_words.update(custom_words)

    # Tokenize the tweet into words
    words = tweet.split()

    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Join the filtered words back into a tweet
    filtered_tweet = ' '.join(filtered_words)

    return filtered_tweet

def clean_tweets(reader, writer):
    # Process each row in the input file
    for row in reader:
        # read the tweet
        tweet = row[0]
        tweet_class = int(row[1])



# Reference: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
# The class label has 3 values: 0, 1, or 2:
# 0-hate speech, 1-offensive language, 2-neither

        # Refactor the class labels: 0 and 1 to 0 "GOOD", class 2 to 1 "BAD"
        if tweet_class == 0 or tweet_class == 1:
            tweet_class = 1 # BAD
        elif tweet_class == 2:
            tweet_class = 0 # GOOD



        # 1- Remove accounts mentions (@account_name):
        processed_tweet = re.sub(r'@[a-zA-Z0-9_]+', '', tweet)

        # 2- Remove hashtags (#hashtag_name):
        processed_tweet = re.sub(r'#', '', processed_tweet)

        # 3- Remove URLs:
        processed_tweet = re.sub(r'http\S+', '', processed_tweet)

        # 4- Remove punctuation marks
        translator = str.maketrans('', '', string.punctuation)
        processed_tweet = processed_tweet.translate(translator)

        # 5- Remove numbers:
        translator = str.maketrans('', '', '0123456789')
        processed_tweet = processed_tweet.translate(translator)

        # 6- Remove beginning and end spaces:
        processed_tweet = processed_tweet.strip()

        # 7- Remove stop words:
        processed_tweet = remove_stopwords(processed_tweet)

        # 8- Remove the "RT" indicating a retweet:
        processed_tweet = re.sub(r'RT\s+', '', processed_tweet)

        # 9- Convert to lower case:
        processed_tweet = processed_tweet.lower()

        # 10- Remove special characters except for letters and spaces
        processed_tweet = re.sub(r'[^a-zA-Z\s]', '', processed_tweet)

        # 11- Remove "":
        processed_tweet = processed_tweet.replace('""', '')

        # 12- Remove extra spaces
        processed_tweet = re.sub(r'\s+', ' ', processed_tweet).strip()

        # If the processed tweet was not empty, write it to the output file:
        if processed_tweet.strip():
            writer.writerow([processed_tweet, tweet_class])  # Write the processed tweet to the output file

def tokenize_stem_tweet(tweet):
    #Tokenize: convert the string into separate words
    # Stem: reduce each word to its original representation: "played", "plays", "playing" ==> "play"

    # Tokenize the tweet:
    tokens = word_tokenize(tweet)

    # Stem each word and save them in a list:
    stemmer = PorterStemmer()
    stemmed_tokens_list = [stemmer.stem(token) for token in tokens]

    return stemmed_tokens_list


# File paths
input_file_path = '/Users/sara/Desktop/Master/Thesis Project/FLASK/FF_MLP/tweets.csv'
output_file_path = '/Users/sara/Desktop/Master/Thesis Project/FLASK/FF_MLP/output.csv'
stemmed_tokens_list = []


with open(input_file_path, mode='r') as input_file, open(output_file_path, mode='w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    # Read the header from the input file
    header = next(reader)
    writer.writerow(header)

    clean_tweets(reader, writer)

with open(output_file_path, mode='r') as input_file:
    reader = csv.reader(input_file)
    next(reader)  # Skip the header

    for row in reader:
        tweet = row[0]
        stemmed_tokens = tokenize_stem_tweet(tweet)
        stemmed_tokens_list.extend(stemmed_tokens)



# ------------------------------------------------------------------------------ NN:
# ------------------------------------------------------------------------------ NN:


# List of cleaned tweets
tweets_list = []

# List of associated labels
labels_list = []

# Read the cleaned tweets:
with open(output_file_path, mode='r') as input_file:
    reader = csv.reader(input_file)
    next(reader)
    for row in reader:
        tweets_list.append(row[0])
        labels_list.append(int(row[1]))



# Tokenize and Stem the tweets
# ----------------------------


# 1- Create a Tokenizer that will only consider the top 100 ...
# ... most frequent words (tokens) based on their occurrence in the provided data:
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

# 2- Apply the Tokenizer on the provided data:
tokenizer.fit_on_texts(tweets_list)

# 3- Replace each word in the tweets by its equivalent integer:
# (Sequences) contains the tweets from tweets_list, but each word in the tweets is replaced ...
# ... by its integer index from the tokenizer's vocabulary:
sequences = tokenizer.texts_to_sequences(tweets_list)


# To make sure that all sequences have the same length
padded_sequences = pad_sequences(sequences, padding='post', truncating='post')


labels = np.array(labels_list)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)




# Build the FF_MLP:
model = tf.keras.Sequential([
    # 1- Converting integer words into dense vectors:
    tf.keras.layers.Embedding(10000, 64, input_length=padded_sequences.shape[1]),

    # Flatten layer to transition from the convolutional layers to the fully connected layers:
    tf.keras.layers.Flatten(),


    # Fully connected dense layer with 100 neurons and 'tanh' activation:
    tf.keras.layers.Dense(100, activation='tanh'),

    # Another fully connected dense layer with 100 neurons and 'tanh' activation:
    tf.keras.layers.Dense(100, activation='tanh'),

    # Add more layers:
    tf.keras.layers.Dense(100, activation='tanh'),
    tf.keras.layers.Dense(50, activation='tanh'),




    # Output layer of single unit and a sigmoid activation function for binary classification:
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model:
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model:
trained_model = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)

# Save the model:
joblib.dump(model, 'Sara_FF_MLP_model.joblib')

# Evaluate the model:
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")





# Visualize the training history
plt.plot(trained_model.history['accuracy'], label='Training Accuracy')
plt.plot(trained_model.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



def make_prediction(sentence):
    # 1- Remove accounts mentions (@account_name):
    processed_sentence = re.sub(r'@[a-zA-Z0-9_]+', '', sentence)

    # 2- Remove hashtags (#hashtag_name):
    processed_sentence = re.sub(r'#', '', processed_sentence)

    # 3- Remove URLs:
    processed_sentence = re.sub(r'http\S+', '', processed_sentence)

    # 4- Remove punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    processed_sentence = processed_sentence.translate(translator)

    # 5- Remove numbers:
    translator = str.maketrans('', '', '0123456789')
    processed_sentence = processed_sentence.translate(translator)

    # 6- Remove beginning and end spaces:
    processed_tweet = processed_sentence.strip()

    # 7- Remove stop words:
    processed_tweet = remove_stopwords(processed_tweet)

    # 8- Remove the "RT" indicating a retweet:
    processed_tweet = re.sub(r'RT\s+', '', processed_tweet)

    # 9- Convert to lower case:
    processed_tweet = processed_tweet.lower()

    # 10- Remove special characters except for letters and spaces
    processed_tweet = re.sub(r'[^a-zA-Z\s]', '', processed_tweet)

    # 11- Remove "":
    processed_tweet = processed_tweet.replace('""', '')

    # 12- Remove extra spaces
    processed_tweet = re.sub(r'\s+', ' ', processed_tweet).strip()

    # Tokenize and pad the new tweet
    new_sequence = tokenizer.texts_to_sequences([processed_sentence])
    new_padded_sequence = pad_sequences(new_sequence, padding='post', truncating='post', maxlen=padded_sequences.shape[1])

    # Make the prediction: predict 5 times and take the average:
    average = 0
    for i in range(5):
        probability_prediction = model.predict(new_padded_sequence)
        average += probability_prediction

    average /= 5

    # The 'prediction' variable contains the probability of the tweet being classified BAD,
    # the closer it is to 1 ==> BAD, the closer it is to 0 ==> GOOD
    threshold = 0.6
    if average > threshold:
        class_ = "BAD"
    else:
        class_ = "GOOD"


    print(f"Sentence: {sentence}, Averaged Probability of being BAD: {average}, Class: {class_}")





# print(model.summary())
make_prediction("friday smiles around via ig user cookies make people")
make_prediction("ouchjunior angrygot junior yugyoem omg")
make_prediction("get see daddy today days gettingfed")