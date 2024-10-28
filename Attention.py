import joblib
import pandas as pd
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Attention
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
import string
from sklearn.model_selection import train_test_split


# loading a dataset of cleaned data:
cleaned_DS_path = '/Users/sara/Desktop/Master/Thesis Project/FLASK/Attention Based Model/trainDataset/output.csv'
df = pd.read_csv(cleaned_DS_path)




# Tokenize: break down the tweets into separate words and convert them into integers::
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['tweet'])
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(df['tweet'])


# apply padding to ensure all the tweets have the same length, max length = 100
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')




# Model:
# -----

# Layer 1: Input Layer:
embedding_dim = 50 # size of each vector
input_layer = Input(shape=(100,))

# Layer 2: Embedding Layer:
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=100)(input_layer)

# layer 3: LSTM Layer:
lstm_layer = Bidirectional(LSTM(100, return_sequences=True))(embedding_layer)

# Layer 4: Attention Layer:
attention_layer = Attention()([lstm_layer, lstm_layer])

# Layer 5: Context Layer:
context_layer = tf.reduce_sum(attention_layer * lstm_layer, axis=1)

# Layer 6: Output Layer:
output_layer = Dense(1, activation='sigmoid')(context_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# Replace 'df['label']' with your actual label data
model.fit(padded_sequences, df['class'], epochs=5, validation_split=0.2, batch_size=32)

joblib.dump(model, 'Sara_attention_model.joblib')

# from keras.utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# Assuming 'df' is your entire dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

val_padded_sequences = pad_sequences(tokenizer.texts_to_sequences(val_df['tweet']), maxlen=100, padding='post')
val_labels = val_df['class']

# Evaluate the model on the validation set
_, accuracy = model.evaluate(val_padded_sequences, val_labels)


# Print model summary:
print(model.summary())

# Print the accuracy
print(f"Model Accuracy on Validation Set: {accuracy * 100:.2f}%")



# ---------------------------------------------------------------------------------


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
    new_sequence = tokenizer.texts_to_sequences([processed_tweet])
    new_padded_sequence = pad_sequences(new_sequence, maxlen=100, padding='post')

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


make_prediction("think about ending your life")
make_prediction("i would cut you into pieces")
make_prediction("let's go to bed have some fun")
make_prediction("white people are the best")


