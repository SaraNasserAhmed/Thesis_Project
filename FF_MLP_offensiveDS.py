

'Model 1: FF_MLP'
import keras
import matplotlib.pyplot as plt

# --------------------

from main_offensiveDS import training_sentences, train_labels, val_set, val_labels, evaluate_results,\
    text_vectorizor, embedding, fix_imbalanced_data
from helper_functions import create_tensorboard_callback

"As with all machine learning modelling experiments, it is important to create a baseline model so you've" \
"got a benchmark for future experiments."

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from keras import layers as lyr
import tensorflow as tf


# Build the model:
text_input = lyr.Input(shape=(1,), dtype=tf.string)  # 1D input = strings
number = text_vectorizor(text_input)                # text converted to number
embedding = embedding(number)                       # number converted to embedding
# Reduce dimension from a prediction per token (___, 8, 1) to a prediction per input(___, 1):
pooled_embedding = lyr.GlobalAveragePooling1D()(embedding)
outputs = lyr.Dense(1, activation="sigmoid")(pooled_embedding)

model_1 = keras.Model(text_input, outputs, name="Model_1_Dense")
print(model_1.summary())


# Compile
model_1.compile(loss = "binary_crossentropy",
                optimizer = keras.optimizers.Adam(),
                metrics = ["accuracy"])

dir_name = "./"

class_weights = fix_imbalanced_data()

# Fit:
model_1_history = model_1.fit(x = training_sentences,
                              y = train_labels,
                              epochs = 5,
                              validation_data = (val_set, val_labels),
                              callbacks = [create_tensorboard_callback(dir_name=dir_name,
                                                                       experiment_name="model_1_Dense")],
                              class_weight=class_weights)


# Check predictions results:
model_1_prediction_probabilities = model_1.predict(val_set)
# print(model_1_prediction_probabilities[:10])

# To convert preds from probabilities to binary classification:
model_1_preds = tf.squeeze(tf.round(model_1_prediction_probabilities))
# print(model_1_preds[:10])


# Check Model's Results on Test Data:
model_1_results = evaluate_results(y_true=val_labels,
                                   y_pred=model_1_preds)
for key, value in model_1_results.items():
    print(f"{key} = {value}")

# Output:
# Accuracy = 91.98%
# Precision = 0.93%
# Recall = 0.92%
# F1-score = 0.92%
