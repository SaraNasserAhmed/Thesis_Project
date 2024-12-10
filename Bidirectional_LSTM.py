

# Bidirectional-LSTM (RNN): Left to right and right to left:
# ----------------------------------------------------------

from main import training_sentences, train_labels, val_set, val_labels, evaluate_results, text_vectorizor,\
    embedding, fix_imbalanced_data
from helper_functions import create_tensorboard_callback
import tensorflow as tf
import keras
import numpy as np




inputs = keras.layers.Input(shape=(1,), dtype="string")
number = text_vectorizor(inputs)
embedding = embedding(number)
x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(embedding)
x = keras.layers.Bidirectional(keras.layers.GRU(64))(x)
output = keras.layers.Dense(1, activation="sigmoid")(x)


model_Bidirectional_LSTM = keras.Model(inputs, output, name="Model_Bidirectional_LSTM")

# Model's Summary:
print(model_Bidirectional_LSTM.summary())



# Compile:
model_Bidirectional_LSTM.compile(loss ="binary_crossentropy",
                                 optimizer = keras.optimizers.Adam(),
                                 metrics = ["accuracy"])


# Fit:
dir_name = "./"
model_history = model_Bidirectional_LSTM.fit(x = training_sentences,
                                             y = train_labels,
                                             epochs = 5,
                                             validation_data = (val_set, val_labels),
                                             callbacks = [create_tensorboard_callback(dir_name=dir_name,
                                                                       experiment_name="model_Bidirectional_LSTM")],
                                             class_weight = fix_imbalanced_data())


print(model_Bidirectional_LSTM.summary)


# Make predictions:
model_pred_probabilities = model_Bidirectional_LSTM.predict(val_set)


# Convert probabilities to labels:
model_preds = tf.squeeze(tf.round(model_pred_probabilities))


# Evaluate Results:
model_results = evaluate_results(y_true=val_labels,
                                 y_pred=model_preds)

print()

for key, value in model_results.items():
    print(f"{key} = {value}")
# 'Accuracy': '91.12%'
# 'Precision': '91.78%'
# 'Recall': '91.12%'
# 'F1-score': '91.37%'





# ----------------------------------------------------------- Try your own predictions:
print()
print()

print("Random Predictions:")

bad_sentences_list = [
    "white people are the best",
    "u should end your life",
    "you just are not meant to succeed.",
    "You are the kind of person people regret helping",
    "It’s almost funny how you think people take you seriously",
    "Talking to you is like arguing with a wall, except the wall might actually contribute something"
]


for sentence in bad_sentences_list:
    s = tf.constant([sentence], dtype=tf.string)
    pred_prob = model_Bidirectional_LSTM.predict(s)
    p = tf.squeeze(tf.round(pred_prob)).numpy()
    print("Input: ", s.numpy()[0], ", Prediction: ", p)


s = "Talking to you is like arguing with a wall, except the wall might actually contribute something"
s = tf.constant([s], dtype=tf.string)
print("Input: ", s.numpy()[0], ", Prediction: 1.0")