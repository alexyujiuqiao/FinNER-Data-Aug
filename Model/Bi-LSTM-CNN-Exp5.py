import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer, Embedding, LSTM, Dense, Input, Bidirectional, TimeDistributed, Conv1D, Concatenate
from tensorflow.keras.models import Model
import random
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from transformers import BertTokenizer, TFBertForMaskedLM, pipeline
from datasets import load_dataset

'''
Parameters:
    * df: the dataset in pd.DataFrame
    * word_col: the column that contains words
    * label_col: the column that contains labels
    * sentence_identifier_col: different sentence should have different identifiers
'''

def separate_by_sentence(df, word_col, label_col, sentence_identifier_col):
    # Prepare the data
    data = []
    current_sentence = []
    current_labels = []
    current_sentence_id = df[sentence_identifier_col].iloc[0]

    for _, row in df.iterrows():
        if row['sentence_id'] != current_sentence_id:    # begin of a new sentence
            data.append((current_sentence, current_labels))
            current_sentence = []
            current_labels = []
            current_sentence_id = row[sentence_identifier_col]
        
        current_sentence.append(row[word_col])
        current_labels.append(row[label_col])

    # Append the last sentence
    data.append((current_sentence, current_labels))
    return data

# Create a number index to every word / label
def num_index(data):
    # take unique numbers in order
    global vocab
    global labels
    labels = set()
    vocab = set()

    for sentence, label in data:
        vocab.update(sentence)
        labels.update(label)
    
    # give a num index to every word
    vocab = {word: idx + 2 for idx, word in enumerate(vocab)}
    vocab["<PAD>"] = 0
    vocab["<OOV>"] = 1

    # a num index to every label
    label_map = {label: idx for idx, label in enumerate(sorted(labels))}
    
    return vocab, label_map

# Convert sentences and labels to integer sequences
def encode_data(data, vocab, label_map):
    encoded_sentences = []
    encoded_labels = []
    
    for sentence, label in data:
        encoded_sentence = [vocab.get(token, vocab["<OOV>"]) for token in sentence]
        encoded_label = [label_map[lbl] for lbl in label]
        
        encoded_sentences.append(encoded_sentence)
        encoded_labels.append(encoded_label)
    
    return encoded_sentences, encoded_labels

'''
Parameter - prediction_matrix: a 2*2 numpy.array
    - each element is a list of length 7 (that with the largest number is the predicted label)
    - each line is a sentence
Output: a 2*2 matrix, in which each element is a label
'''

def make_prediction(predicted_values: np.array):
    output = np.full((predicted_values.shape[0], predicted_values.shape[1]), np.nan)
    for i in range(predicted_values.shape[0]):
        for j in range(predicted_values.shape[1]):
            predicted = pd.Series(predicted_values[i, j])
            most_probable_value = predicted[predicted == predicted.max()].index
            output[i, j] = int(most_probable_value[0])
    return output

'''
Goal: The 0 (out of noun entity) labels are overwhelming in this dataset.
    We want to only keep a part of sentences with all 0, to make the model focus more on words within noun entities
Parameter:
    - train_labels: the input data
    - kept_percent: the remained percentage of data
'''

def undersample(train_labels, kept_percent):
    all_zero_sentences = np.where(np.array([np.sum(i) for i in train_labels]) == 0)[0]
    eliminated_index = np.random.choice(all_zero_sentences, int(len(all_zero_sentences) * kept_percent), replace=False)
    return eliminated_index

# ****************** Augmentation Functions *****************

# Synonym Replacement
def synonym_replacement(sentence, labels, p=0.1):
    new_sentence = []
    for token in sentence:
        if random.random() < p:
            synonyms = wordnet.synsets(token)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()  # Choose the first synonym
                new_sentence.append(synonym)
            else:
                new_sentence.append(token)
        else:
            new_sentence.append(token)
    return new_sentence, labels

# Random Token Insertion
def random_insertion(sentence, labels, p=0.1):
    new_sentence = []
    new_labels = []
    for token, label in zip(sentence, labels):
        new_sentence.append(token)
        new_labels.append(label)
        if random.random() < p:
            new_sentence.append("<INSERTED>")
            new_labels.append(0)  # Neutral label
    return new_sentence, new_labels

# Token Deletion
def token_deletion(sentence, labels, p=0.1):
    new_sentence = []
    new_labels = []
    for token, label in zip(sentence, labels):
        if random.random() > p:
            new_sentence.append(token)
            new_labels.append(label)
    return new_sentence, new_labels

# Noisy Label Augmentation
def noisy_labels(labels, p=0.05):
    return [random.choice(list(label_map.values())) if random.random() < p else lbl for lbl in labels]

# ****************** Contextual Replacement ******************
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')
# Define the fill-mask pipeline
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

def contextual_replacement(sentence, labels):
    new_sentence = sentence.copy()
    for i, token in enumerate(sentence):
        if random.random() < 0.15:  # Mask 15% of the tokens
            new_sentence[i] = '[MASK]'
    
    inputs = tokenizer(' '.join(new_sentence), return_tensors='tf')
    outputs = model(**inputs)
    predictions = tf.argmax(outputs.logits, axis=-1)
    
    predicted_tokens = tokenizer.convert_ids_to_tokens(predictions[0])
    augmented_sentence = [predicted_tokens[i] if token == '[MASK]' else token for i, token in enumerate(new_sentence)]
    
    return augmented_sentence, labels

def contextual_replacement(sentence, labels, mask_token="[MASK]", p=0.1):
    new_sentence = []
    for idx, token in enumerate(sentence):
        if random.random() < p and token.isalpha():
            masked_sentence = sentence[:]
            masked_sentence[idx] = mask_token
            predictions = fill_mask(" ".join(masked_sentence))
            new_word = predictions[0]["token_str"]  # Use the top prediction
            new_sentence.append(new_word)
        else:
            new_sentence.append(token)
    return new_sentence, labels

# Augmentation Pipeline
def augment_data(data, vocab, label_map):
    augmented_data = []
    for sentence, labels in data:
        # Apply contextual replacement to a subset of data
        # if random.random() < 0.5:  # Apply contextual replacement to 50% of the data
        #     augmented_sentence, augmented_labels = contextual_replacement(sentence, labels)
        #     augmented_data.append((augmented_sentence, augmented_labels))
        
        # Apply synonym replacement
        # if random.random() < 0.3:
        # augmented_sentence, augmented_labels = synonym_replacement(sentence, labels)
        # augmented_data.append((augmented_sentence, augmented_labels))
        
        # # Apply token deletion
        # if random.random() < 0.2:
        # augmented_sentence, augmented_labels = token_deletion(sentence, labels)
        # augmented_data.append((augmented_sentence, augmented_labels))
        
        # # Apply random insertion
        # if random.random() < 0.3:
        # augmented_sentence, augmented_labels = random_insertion(sentence, labels)
        # augmented_data.append((augmented_sentence, augmented_labels))
        
        # # Add noisy labels (optional)
        # if random.random() < 0.2:
        noisy_augmented_labels = noisy_labels(labels)
        augmented_data.append((sentence, noisy_augmented_labels))
    
    return augmented_data

# ****************** Load the dataset ******************
'''
The dataset is imported from GitHub, and every word is given a label (sentence_id) to show from which sentence it comes from.
'''

splits = {'train': 'train.csv', 'validation': 'val.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/gtfintechlab/finer-ord/" + splits["train"])
new_row = pd.DataFrame({'gold_label': 0, 'gold_token': '.', 'sent_idx': '0', 'doc_idx': 0}, index = [0])
df = pd.concat([new_row, df])
df.index = range(len(df))
# Combine document ID and sentence ID to create a unique sentence identifier
df['sentence_id'] = df['doc_idx'].astype(str) + "_" + df['sent_idx'].astype(str)

# ******************** pre-processing ********************
'''
Here words within a sentence is combined together. This would help the NN to capture the in-senetence relationships.
Then, words and labels are converted into numbers, to be understanable to computers.
'''

# Separate data by sentences
data = separate_by_sentence(df, 'gold_token', 'gold_label', 'sentence_id')

# Turn words into numbers
vocab, label_map = num_index(data)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# ****************** Data Augmentation ******************
# Augment the training data
augmented_train_data = augment_data(train_data, vocab, label_map)

# Combine original and augmented training data
train_data.extend(augmented_train_data)

# Check the size of the augmented training dataset
print(f"Original training size: {len(train_data) - len(augmented_train_data)}")
print(f"Augmented training size: {len(train_data)}")

# Convert sentences and labels to integer sequences
train_sentences, train_labels = encode_data(train_data, vocab, label_map)
test_sentences, test_labels = encode_data(test_data, vocab, label_map)


# eliminate some of the all 0 sentences (50%)
eliminated_indexs = undersample(train_labels, 0.5)

train_sentences_list = list(train_sentences)
train_sentences_list = [sentence for idx, sentence in enumerate(train_sentences_list) if idx not in eliminated_indexs]
train_sentences = np.array(train_sentences_list, dtype=object)

train_labels_list = list(train_labels)
train_labels_list = [sentence for idx, sentence in enumerate(train_labels_list) if idx not in eliminated_indexs]
train_labels = np.array(train_labels_list, dtype=object)

# ******************** one-hot coding **************************
'''
Used as inputs of a NN, each line in the matrix should have the same length. Therefore, we fill every sentence to the same length as the longest one.
Then, the numbers are further turned into hot-coding, since we don't want the model to capture the relative size of labels (0~6).
'''

# fill the sentence to the max length
max_train_len = max([len(sentence) for sentence in train_sentences])
max_test_len = max([len(sentence) for sentence in test_sentences])
max_len = max(max_train_len, max_test_len)  # Use the longest length from both training and testing


train_sentences = pad_sequences(train_sentences, maxlen=max_len, padding='post')
train_labels = pad_sequences(train_labels, maxlen=max_len, padding='post')
test_sentences = pad_sequences(test_sentences, maxlen=max_len, padding='post')
test_labels = pad_sequences(test_labels, maxlen=max_len, padding='post')

train_labels = [to_categorical(label, num_classes=len(label_map)) for label in train_labels]
test_labels = [to_categorical(label, num_classes=len(label_map)) for label in test_labels]


# ******************** define the deep model ********************
'''
input layer: receive a sentence
embedding layer: embed a word into 64 dimensions
CNN layer: extract local features
LSTM layer: 100 hidden units (forget gate + input gate + output gate)
output layer: 7 output dimensions
'''
input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=len(vocab), output_dim=64, input_length=max_len)(input_layer)

# Add CNN layers with different kernel sizes
conv_outputs = []
kernel_sizes = [3, 4, 5]
for kernel_size in kernel_sizes:
    conv = Conv1D(
        filters=64,
        kernel_size=kernel_size,
        padding='same',
        activation='relu',
        name=f'conv1d_{kernel_size}'
    )(embedding_layer)
    conv_outputs.append(conv)

# Concatenate CNN outputs
cnn_features = Concatenate(axis=-1)(conv_outputs)

lstm_layer = Bidirectional(LSTM(units=100, return_sequences=True))(cnn_features)
output_layer = TimeDistributed(Dense(len(label_map), activation="softmax"))(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# ******************** Train the model ********************
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

model.fit(
    train_sentences,
    train_labels,
    validation_data=(test_sentences, test_labels),
    batch_size=32,
    epochs=7
)

# Model summary
model.summary()


# ******************** Evaluate the model ********************
dataset = load_dataset("gagan3012/finer_ord", split = "train")

# Process the dataset to extract words and labels
test_words = []
test_labels = []

for example in dataset:
    # Extract text and labels
    text = example["text"]
    labels = example["label"]  # Convert string representation of list to Python list
    
    test_words.append(text.split())
    test_labels.append(labels)

# Convert words to their respective indices
encoded_test_sentences = [
    [vocab.get(word, vocab["<OOV>"]) for word in sentence] for sentence in test_words
]

# Define the label mapping
label_map = {
    "O": 0,
    "PER_B": 1,
    "PER_I": 2,
    "LOC_B": 3,
    "LOC_I": 4,
    "ORG_B": 5,
    "ORG_I": 6
}

# convert labels to numerical format
numerical_test_labels = [
    [label_map[label] if label in label_map else label_map[0] for label in sentence]
    for sentence in test_labels
]

# Pad sequences for evaluation
padded_test_sentences = pad_sequences(encoded_test_sentences, maxlen=max_len, padding="post")
padded_test_labels = pad_sequences(numerical_test_labels, maxlen=max_len, padding="post")

# Convert labels to categorical format
categorical_test_labels = [to_categorical(label, num_classes=len(label_map)) for label in padded_test_labels]

# Predict labels for the test data
predictions = model.predict(padded_test_sentences)

# Convert predictions to label indices
predicted_labels = np.argmax(predictions, axis=-1)

# Calculate metrics
true_labels_flat = padded_test_labels.flatten()
predicted_labels_flat = predicted_labels.flatten()

# Evaluate performance
print("Accuracy:", accuracy_score(true_labels_flat, predicted_labels_flat))
print("Classification Report:")
print(classification_report(true_labels_flat, predicted_labels_flat, target_names=list(label_map.keys())))