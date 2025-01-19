import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFBertModel, BertTokenizerFast
from datasets import load_dataset

# ******************** Load FinBERT ********************
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizerFast.from_pretrained(finbert_model_name)
finbert = TFBertModel.from_pretrained(finbert_model_name)

# ******************** Load the Dataset ********************
# 加载 finer-139 数据集
dataset = load_dataset("nlpaueb/finer-139", split="train")

# 转换为 Pandas DataFrame
df = pd.DataFrame(dataset)

# 生成 sentence_id 列（用行索引作为唯一标识符）
df["sentence_id"] = df.index.astype(str)

# ******************** Separate Sentences and Labels ********************
def separate_by_sentence(df, word_col, label_col):
    data = []
    for _, row in df.iterrows():
        sentence = row[word_col]  # 获取 tokens
        labels = row[label_col]  # 获取 ner_tags
        data.append((sentence, labels))
    return data

data = separate_by_sentence(df, "tokens", "ner_tags")

# ******************** Create Vocabulary and Label Map ********************
def num_index(data):
    global vocab
    global labels
    labels = set()
    vocab = set()

    for sentence, label in data:
        vocab.update(sentence)
        labels.update(label)

    vocab = {word: idx + 2 for idx, word in enumerate(vocab)}
    vocab["<PAD>"] = 0
    vocab["<OOV>"] = 1

    label_map = {label: idx for idx, label in enumerate(sorted(labels))}
    return vocab, label_map

vocab, label_map = num_index(data)

# ******************** Split Data ********************
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# ******************** Tokenization and Preprocessing ********************
def tokenize_data(sentences, labels, tokenizer, max_len):
    input_ids, attention_masks, token_labels = [], [], []

    for sentence, label in zip(sentences, labels):
        tokens = tokenizer(
            sentence, 
            padding="max_length", 
            truncation=True, 
            max_length=max_len, 
            return_tensors="tf", 
            is_split_into_words=True
        )
        # 不使用 tf.squeeze，直接使用返回的张量
        input_ids.append(tokens["input_ids"][0])
        attention_masks.append(tokens["attention_mask"][0])

        word_ids = tokens.word_ids()
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        token_labels.append(label_ids)

    return (
        tf.convert_to_tensor(input_ids),
        tf.convert_to_tensor(attention_masks),
        tf.convert_to_tensor(token_labels)
    )

# 准备数据
max_len = 128
train_sentences = [sentence for sentence, _ in train_data]
train_labels = [label for _, label in train_data]
test_sentences = [sentence for sentence, _ in test_data]
test_labels = [label for _, label in test_data]

train_inputs, train_masks, train_labels = tokenize_data(train_sentences, train_labels, tokenizer, max_len)
test_inputs, test_masks, test_labels = tokenize_data(test_sentences, test_labels, tokenizer, max_len)

# ******************** Build the Model ********************
input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

finbert_output = finbert(input_ids, attention_mask=attention_mask)[0]
output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(label_map), activation="softmax"))(finbert_output)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ******************** Train the Model ********************
model.fit(
    {"input_ids": train_inputs, "attention_mask": train_masks},
    train_labels,
    validation_data=({"input_ids": test_inputs, "attention_mask": test_masks}, test_labels),
    batch_size=16,
    epochs=3
)

# Save the model
model.save("finbert_ner_model")

# ******************** Predict ********************
test_sentence = "John works at Morgan Stanley"
tokens = tokenizer(test_sentence, padding="max_length", truncation=True, max_length=max_len, return_tensors="tf")
predictions = model.predict({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})
predicted_labels = tf.argmax(predictions, axis=-1)
print("Predicted labels:", predicted_labels)
