from datasets import load_dataset
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model

# Load the dataset
dataset = load_dataset("gagan3012/finer_ord")
train_data = dataset["train"]
test_data = dataset["test"]

# Convert Hugging Face dataset to Pandas DataFrame
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Processing the dataset: separating words and labels
def process_data(df):
    words = []
    labels = []
    for idx, row in df.iterrows():
        words.extend(row["text"].split())
        try:
            if isinstance(row["label"], str):
                labels.extend(eval(row["label"]))  # Convert string list to Python list
            else:
                # Handle cases where "label" is not a string (e.g., NaN)
                labels.extend(["O"] * len(row["text"].split()))
        except Exception as e:
            print(f"Error parsing label: {row['label']}, Error: {e}")
            labels.extend(["O"] * len(row["text"].split()))  # Default to "O" for unlabeled tokens
    return words, labels

train_words, train_labels = process_data(train_df)
test_words, test_labels = process_data(test_df)

# Create a vocabulary and label map
vocab = {word: idx + 2 for idx, word in enumerate(set(train_words))}
vocab["<PAD>"] = 0
vocab["<OOV>"] = 1

label_map = {label: idx for idx, label in enumerate(set(train_labels))}
num_labels = len(label_map)

# Encode words and labels
def encode_data(words, labels, vocab, label_map, max_len):
    encoded_sentences = [vocab.get(word, vocab["<OOV>"]) for word in words]
    encoded_labels = [label_map[label] for label in labels]
    padded_sentences = pad_sequences([encoded_sentences], maxlen=max_len, padding="post")[0]
    padded_labels = pad_sequences([encoded_labels], maxlen=max_len, padding="post")[0]
    return padded_sentences, padded_labels

# Define maximum sequence length
max_len = 128  # Adjust based on dataset or use the longest sentence length

print(f"Augmented training size: {len(train_data)}")

# Prepare training and testing data
train_sentences, train_labels = encode_data(train_words, train_labels, vocab, label_map, max_len)
test_sentences, test_labels = encode_data(test_words, test_labels, vocab, label_map, max_len)

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels, num_classes=num_labels)
test_labels = to_categorical(test_labels, num_classes=num_labels)

# Define the NER model
def build_model(vocab_size, num_labels, max_len):
    input_layer = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len)(input_layer)
    lstm_layer = Bidirectional(LSTM(units=64, return_sequences=True))(embedding_layer)
    output_layer = TimeDistributed(Dense(num_labels, activation="softmax"))(lstm_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_model(len(vocab), num_labels, max_len)

# Train the model
model.fit(
    np.array([train_sentences]),
    np.array([train_labels]),
    validation_data=(np.array([test_sentences]), np.array([test_labels])),
    epochs=5,
    batch_size=32
)

# Evaluate the model
loss, accuracy = model.evaluate(np.array([test_sentences]), np.array([test_labels]))
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predict on new sentences
def predict_ner(sentence, model, vocab, label_map, max_len):
    words = sentence.split()
    encoded_sentence = [vocab.get(word, vocab["<OOV>"]) for word in words]
    padded_sentence = pad_sequences([encoded_sentence], maxlen=max_len, padding="post")
    predictions = model.predict(padded_sentence)[0]
    idx_to_label = {idx: label for label, idx in label_map.items()}
    predicted_labels = [idx_to_label[np.argmax(tag)] for tag in predictions[:len(words)]]
    return list(zip(words, predicted_labels))

# Example prediction
sentence = "Barack Obama visited Google in California."
predicted_entities = predict_ner(sentence, model, vocab, label_map, max_len)
print(predicted_entities)
