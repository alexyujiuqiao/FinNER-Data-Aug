import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from datasets import load_dataset
from seqeval.metrics import classification_report, accuracy_score
import nltk
import random
import json

nltk.download('wordnet')

# ****************** Load the Financial-NER-NLP dataset ******************
ds = load_dataset("Josephgflowers/Financial-NER-NLP")
df = pd.DataFrame(ds["train"])

# Ensure assistant column contains XBRL labels
df = df[df["assistant"].str.contains(":")]

# Create sentence IDs for input and output alignment
df["sentence_id"] = df.index

# Separate data by sentences
def prepare_data(df):
    sentences = []
    labels = []
    for i, row in df.iterrows():
        tokens = row["user"].split()  # Tokenize user text
        xbrl_tags = ["O"] * len(tokens)  # Default label as "O" for non-entities
        
        # Map XBRL tags from assistant column to tokens
        if row["assistant"] != "No XBRL associated data.":
            try:
                entity_tags = eval(row["assistant"])  # Parse the assistant dictionary
                for entity, values in entity_tags.items():
                    for value in values:
                        for token in tokens:
                            if value in token:
                                xbrl_tags[tokens.index(token)] = entity
            except:
                pass
        
        sentences.append(tokens)
        labels.append(xbrl_tags)
    return sentences, labels

# Prepare sentences and labels
sentences, labels = prepare_data(df)

# Create vocabulary and label mapping
word_vocab = {word: i + 2 for i, word in enumerate(set(token for sent in sentences for token in sent))}
word_vocab["<PAD>"] = 0
word_vocab["<OOV>"] = 1
label_vocab = {label: i for i, label in enumerate(set(tag for label in labels for tag in label))}
id_to_label = {i: label for label, i in label_vocab.items()}  # For decoding predictions

# Encode sentences and labels
def encode_data(sentences, labels, word_vocab, label_vocab):
    encoded_sentences = [[word_vocab.get(word, word_vocab["<OOV>"]) for word in sent] for sent in sentences]
    encoded_labels = [[label_vocab[tag] for tag in lbl] for lbl in labels]
    return encoded_sentences, encoded_labels

encoded_sentences, encoded_labels = encode_data(sentences, labels, word_vocab, label_vocab)

# Pad sequences
max_len = max(len(sent) for sent in encoded_sentences)
padded_sentences = pad_sequences(encoded_sentences, maxlen=max_len, padding="post")
padded_labels = pad_sequences(encoded_labels, maxlen=max_len, padding="post")

# One-hot encode labels
padded_labels = [to_categorical(label, num_classes=len(label_vocab)) for label in padded_labels]

# Split the dataset
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    padded_sentences, padded_labels, test_size=0.2, random_state=42
)

# ******************** Build the NER Model ********************
input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=len(word_vocab), output_dim=128, input_length=max_len)(input_layer)
lstm_layer = Bidirectional(LSTM(units=64, return_sequences=True))(embedding_layer)
output_layer = TimeDistributed(Dense(len(label_vocab), activation="softmax"))(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ******************** Train the Model ********************
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

history = model.fit(
    train_sentences,
    train_labels,
    validation_data=(test_sentences, test_labels),
    batch_size=32,
    epochs=5,
    verbose=1
)

# ******************** Plot Loss and Accuracy ********************
import matplotlib.pyplot as plt

# Plot training & validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model and vocab
model.save("financial_ner_model.h5")
with open("word_vocab.json", "w") as f:
    json.dump(word_vocab, f)
with open("label_vocab.json", "w") as f:
    json.dump(label_vocab, f)

# ******************** Evaluate the Model ********************
def decode_predictions(predictions):
    """Convert predictions back to label format."""
    predicted_tags = []
    for prediction in predictions:
        tag_ids = np.argmax(prediction, axis=-1)  # Get the most probable tag ID
        tags = [id_to_label[tag_id] for tag_id in tag_ids if tag_id in id_to_label]
        predicted_tags.append(tags)
    return predicted_tags

# Decode predictions and true labels
y_pred = decode_predictions(model.predict(test_sentences))
y_true = [[id_to_label[np.argmax(tag)] for tag in label] for label in test_labels]

# Align lengths
y_pred = [pred[:len(true)] for pred, true in zip(y_pred, y_true)]

# Generate classification report
print("Classification Report:")
print(classification_report(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))
