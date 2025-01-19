import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

'''
Parameters:
    * df: the dataset in pd.DataFrame
    * word_col: the column that contains words
    * label_col: the column that contains labels
    * sentence_identifier_col: different sentence should have different identifiers
'''
def separate_by_sentence(df, word_col, label_col, sentence_identifier_col):
    data = []
    current_sentence = []
    current_labels = []
    current_sentence_id = df[sentence_identifier_col].iloc[0]

    for _, row in df.iterrows():
        if row[sentence_identifier_col] != current_sentence_id:
            data.append((current_sentence, current_labels))
            current_sentence = []
            current_labels = []
            current_sentence_id = row[sentence_identifier_col]
        
        current_sentence.append(row[word_col])
        current_labels.append(row[label_col])

    # Append the last sentence
    data.append((current_sentence, current_labels))
    return data

'''
Convert labels from FINER-ORD format to BERT-NER format
'''
def convert_labels(labels):
    finer_to_bert_labels = {
        "O": "O",
        "PER_B": "B-PER",
        "PER_I": "I-PER",
        "LOC_B": "B-LOC",
        "LOC_I": "I-LOC",
        "ORG_B": "B-ORG",
        "ORG_I": "I-ORG"
    }
    return [finer_to_bert_labels.get(str(label), "O") for label in labels]

'''
Create a number index to every label
'''
def create_label_map():
    label_map = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-LOC": 3,
        "I-LOC": 4,
        "B-ORG": 5,
        "I-ORG": 6
    }
    return label_map, {v: k for k, v in label_map.items()}

'''
Data augmentation techniques
'''
def random_deletion(sentence, labels, p=0.1):
    """Randomly delete words from the sentence"""
    if len(sentence) <= 1:
        return sentence, labels
    
    new_sentence = []
    new_labels = []
    for word, label in zip(sentence, labels):
        if random.random() > p:
            new_sentence.append(word)
            new_labels.append(label)
    
    if not new_sentence:  # If all words were deleted
        rand_idx = random.randint(0, len(sentence)-1)
        return [sentence[rand_idx]], [labels[rand_idx]]
    
    return new_sentence, new_labels

def augment_data(data, augment_ratio=0.3):
    """Augment the dataset"""
    augmented_data = []
    for sentence, labels in data:
        # Original data
        augmented_data.append((sentence, labels))
        
        # Random deletion
        if random.random() < augment_ratio:
            aug_sentence, aug_labels = random_deletion(sentence, labels)
            augmented_data.append((aug_sentence, aug_labels))
    
    return augmented_data

'''
Load and preprocess the dataset
'''
def load_training_data(file_path):
    # Load CSV data
    df = pd.read_csv(file_path)
    
    # Add sentence identifier
    df['sentence_id'] = df['doc_idx'].astype(str) + "_" + df['sent_idx'].astype(str)
    
    # Separate into sentences
    data = separate_by_sentence(df, 'gold_token', 'gold_label', 'sentence_id')
    
    # Convert labels to BERT format
    processed_data = [(sent, convert_labels(labels)) for sent, labels in data]
    
    return processed_data

'''
Model and prediction functions
'''
class BERTModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        self.label_map, self.reverse_label_map = create_label_map()

    def predict_text(self, text):
        results = self.nlp(text)
        entities = []
        for result in results:
            entity = {
                'word': result['word'],
                'entity': result['entity'],
                'confidence': result['score'],
                'span': (result['start'], result['end'])
            }
            entities.append(entity)
        return entities

    def evaluate_on_dataset(self, texts, true_labels):
        predictions = []
        processed_true_labels = []
        
        for text, true_label in zip(texts, true_labels):
            text = ' '.join(text) if isinstance(text, list) else text
            results = self.nlp(text)
            
            pred_labels = ['O'] * len(text.split())
            for result in results:
                word_idx = len(text[:result['start']].split())
                # Map unknown labels to 'O'
                if result['entity'] in self.label_map:
                    pred_labels[word_idx] = result['entity']
                else:
                    pred_labels[word_idx] = 'O'
            
            predictions.extend(pred_labels)
            processed_true_labels.extend(true_label)
        
        true_numerical = [self.label_map[label] for label in processed_true_labels]
        pred_numerical = [self.label_map[label] for label in predictions]
        
        return accuracy_score(true_numerical, pred_numerical), classification_report(
            true_numerical, 
            pred_numerical,
            target_names=list(self.label_map.keys()),
            digits=4
        )

def main():
    # Initialize model
    bert_model = BERTModel()
    
    # Load training data
    print("Loading and preprocessing training data...")
    train_data = load_training_data("hf://datasets/gtfintechlab/finer-ord/train.csv")
    
    # Augment training data
    print("Augmenting training data...")
    augmented_train_data = augment_data(train_data)
    
    # Load evaluation dataset
    print("Loading evaluation data...")
    eval_dataset = load_dataset("gagan3012/finer_ord", split='test')
    eval_texts = [example["text"].split() for example in eval_dataset]
    eval_labels = [convert_labels(example["label"]) for example in eval_dataset]
    
    # Evaluate model
    print("\nEvaluating on gagan3012/finer_ord test set...")
    accuracy, report = bert_model.evaluate_on_dataset(eval_texts, eval_labels)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Example prediction
    example_text = "Apple Inc. CEO Tim Cook announced new iPhone in Cupertino."
    print("\nExample prediction:")
    print(f"Text: {example_text}")
    predictions = bert_model.predict_text(example_text)
    
    print("\nPredicted entities:")
    for pred in predictions:
        print(f"{pred['word']}: {pred['entity']} (confidence: {pred['confidence']:.4f})")

if __name__ == "__main__":
    main()

