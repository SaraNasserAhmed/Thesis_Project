

# Model 6: Transfer Learning:
# --------------------------

# Where we can find pretrained embeddings? Tensorflow Hub


# Universal Sentence Encoder:
# Reference: https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder/2?tfhub-redirect=true
# Reference: https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder/2
# Reference: https://www.kaggle.com/models/google/universal-sentence-encoder/code





import tensorflow_hub as hub
import tensorflow as tf
import keras
from keras import Model
from keras import layers
from main import training_sentences, train_labels, val_labels, val_set, evaluate_results
from helper_functions import create_tensorboard_callback



class CustomUSELayer(layers.Layer):
    def __init__(self, hub_url, **kwargs):
        super(CustomUSELayer, self).__init__(**kwargs)
        self.encoder = hub.KerasLayer(hub_url, dtype=tf.string, trainable=False, name = "Model_6_USE")

    def call(self, inputs):
        return self.encoder(inputs)


USE_layer = CustomUSELayer("https://tfhub.dev/google/universal-sentence-encoder/4")




inputs = layers.Input(shape=(), dtype=tf.string)
x = USE_layer(inputs)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model_6_USE = Model(inputs, outputs, name="Model_6_USE")
model_6_USE.summary()



# Compile:
model_6_USE.compile(loss ="binary_crossentropy",
                       optimizer = keras.optimizers.Adam(),
                       metrics = ["accuracy"])


# Fit:
dir_name = "./"
model_6_USE_history = model_6_USE.fit(x = training_sentences,
                                    y = train_labels,
                                    epochs = 5,
                                    validation_data = (val_set, val_labels),
                                    callbacks = [create_tensorboard_callback(dir_name=dir_name,
                                    experiment_name="model_6_USE")])


# Make predictions:
model_6_pred_probabilities = model_6_USE.predict(val_set)


# Convert probabilities to labels:
model_6_preds = tf.squeeze(tf.round(model_6_pred_probabilities))


# Evaluate Results:
model_6_results = evaluate_results(y_true=val_labels,
                                   y_pred=model_6_preds)

for key, value in model_6_results.items():
    print(f"{key} = {value}")
# Accuracy = 93.85%
# Precision = 93.7%
# Recall = 93.85%
# F1-score = 93.74%





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
    input_tensor = tf.constant([sentence], dtype=tf.string)
    pred_prob = model_6_USE.predict(input_tensor)
    pred_label = tf.squeeze(tf.round(pred_prob)).numpy()
    print(f"Input: \"{sentence}\" \nPrediction: {pred_label}\n")





