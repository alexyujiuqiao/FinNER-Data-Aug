import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch

# Load SEC-BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/sec-bert-base')
model = TFAutoModelForTokenClassification.from_pretrained(
    'nlpaueb/sec-bert-base',
    num_labels=7,  # O, PER_B, PER_I, LOC_B, LOC_I, ORG_B, ORG_I
    from_pt=True
)

# Define label mapping
label_map = {
    "O": 0,
    "PER_B": 1,
    "PER_I": 2,
    "LOC_B": 3,
    "LOC_I": 4,
    "ORG_B": 5,
    "ORG_I": 6
}
reverse_label_map = {v: k for k, v in label_map.items()}

def preprocess_data_from_csv(file_path):
    """
    Preprocess the gtfintechlab/finer-ord dataset
    """
    df = pd.read_csv(file_path)
    texts = []
    labels = []
    current_text = []
    current_labels = []
    current_sent_id = None
    
    for _, row in df.iterrows():
        sent_id = f"{row['doc_idx']}_{row['sent_idx']}"
        
        if current_sent_id is not None and sent_id != current_sent_id:
            # Process completed sentence
            text = ' '.join(current_text)
            tokenized = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='tf'
            )
            
            padded_labels = pad_sequences(
                [current_labels],
                maxlen=128,
                padding='post',
                value=label_map["O"]
            )[0]
            
            texts.append(tokenized)
            labels.append(padded_labels)
            
            # Reset collectors
            current_text = []
            current_labels = []
        
        current_sent_id = sent_id
        current_text.append(str(row['gold_token']))
        current_labels.append(row['gold_label'])
    
    # Process the last sentence
    if current_text:
        text = ' '.join(current_text)
        tokenized = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='tf'
        )
        
        padded_labels = pad_sequences(
            [current_labels],
            maxlen=128,
            padding='post',
            value=label_map["O"]
        )[0]
        
        texts.append(tokenized)
        labels.append(padded_labels)
    
    # Add this check to ensure all tokenized inputs have the same length as labels
    assert len(texts) == len(labels), f"Mismatch in number of texts ({len(texts)}) and labels ({len(labels)})"
    
    return texts, labels

def preprocess_evaluation_data(dataset):
    """
    Preprocess the gagan3012/finer_ord dataset for evaluation
    """
    texts = []
    labels = []
    
    for example in dataset:
        text = example["text"]
        label = example["label"]
        
        tokenized = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='tf'
        )
        
        numerical_labels = [label_map[l] if l in label_map else label_map["O"] 
                          for l in label]
        
        padded_labels = pad_sequences(
            [numerical_labels],
            maxlen=128,
            padding='post',
            value=label_map["O"]
        )[0]
        
        texts.append(tokenized)
        labels.append(padded_labels)
    
    return texts, labels

def create_tf_dataset(texts, labels, batch_size=16):
    """
    Create TensorFlow dataset from preprocessed data
    """
    # Add input validation
    input_ids = [t['input_ids'][0] for t in texts]
    attention_mask = [t['attention_mask'][0] for t in texts]
    
    assert len(input_ids) == len(labels), f"Mismatch in number of input_ids ({len(input_ids)}) and labels ({len(labels)})"
    assert len(attention_mask) == len(labels), f"Mismatch in number of attention_masks ({len(attention_mask)}) and labels ({len(labels)})"
    
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        },
        labels
    ))
    
    return dataset.shuffle(1000).batch(batch_size)

# Load training data from gtfintechlab/finer-ord
splits = {'train': 'train.csv', 'validation': 'val.csv', 'test': 'test.csv'}
base_path = "hf://datasets/gtfintechlab/finer-ord/"

# Preprocess training datasets
train_texts, train_labels = preprocess_data_from_csv(base_path + splits['train'])
val_texts, val_labels = preprocess_data_from_csv(base_path + splits['validation'])

# Create TF datasets for training
train_tf_dataset = create_tf_dataset(train_texts, train_labels)
val_tf_dataset = create_tf_dataset(val_texts, val_labels)

# Load evaluation dataset (gagan3012/finer_ord)
eval_dataset = load_dataset("gagan3012/finer_ord")
eval_texts, eval_labels = preprocess_evaluation_data(eval_dataset['test'])
eval_tf_dataset = create_tf_dataset(eval_texts, eval_labels)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Add callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]

# Train the model
history = model.fit(
    train_tf_dataset,
    validation_data=val_tf_dataset,
    epochs=10,
    callbacks=callbacks
)

# Evaluate the model on gagan3012/finer_ord test set
test_results = model.evaluate(eval_tf_dataset)
print(f"\nEvaluation on gagan3012/finer_ord test set:")
print(f"Test loss: {test_results[0]:.4f}")
print(f"Test accuracy: {test_results[1]:.4f}")

# Function to predict on new text
def predict_ner(text):
    # Tokenize input text
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )
    
    # Get predictions
    outputs = model(inputs)
    predictions = tf.argmax(outputs.logits, axis=-1)
    
    # Convert predictions to labels
    predicted_labels = [reverse_label_map[pred] for pred in predictions[0].numpy() 
                       if pred in reverse_label_map]
    
    # Align predictions with tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Create output with aligned tokens and labels
    result = []
    for token, label in zip(tokens, predicted_labels):
        if token.startswith('##'):
            continue
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        result.append((token, label))
    
    return result

# Example usage
example_text = "Apple Inc. CEO Tim Cook announced new iPhone in Cupertino."
predictions = predict_ner(example_text)
print("\nExample prediction:")
print(example_text)
print("\nPredicted entities:")
for token, label in predictions:
    if label != 'O':
        print(f"{token}: {label}")

# Generate detailed classification report
def generate_classification_report(dataset):
    all_true_labels = []
    all_predicted_labels = []
    
    for batch in dataset:
        inputs = batch[0]
        true_labels = batch[1]
        
        outputs = model(inputs)
        predictions = tf.argmax(outputs.logits, axis=-1)
        
        # Remove padding from true labels and predictions
        mask = inputs['attention_mask']
        
        for i in range(len(true_labels)):
            true = true_labels[i][mask[i] == 1].numpy()
            pred = predictions[i][mask[i] == 1].numpy()
            
            all_true_labels.extend(true)
            all_predicted_labels.extend(pred)
    
    # Generate classification report
    target_names = list(label_map.keys())
    report = classification_report(
        all_true_labels,
        all_predicted_labels,
        target_names=target_names,
        digits=4
    )
    
    return report

print("\nDetailed Classification Report on gagan3012/finer_ord test set:")
print(generate_classification_report(eval_tf_dataset))