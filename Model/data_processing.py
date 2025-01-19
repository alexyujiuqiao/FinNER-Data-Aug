# ****************** Augmentation Functions ******************
import random
from nltk.corpus import wordnet

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

# Augmentation Pipeline
def augment_data(data, vocab, label_map):
    augmented_data = []
    for sentence, labels in data:
        # Apply one or more augmentation techniques
        if random.random() < 0.5:  # Apply synonym replacement to 50% of the data
            augmented_sentence, augmented_labels = synonym_replacement(sentence, labels)
            augmented_data.append((augmented_sentence, augmented_labels))
        if random.random() < 0.3:  # Apply token deletion to 30% of the data
            augmented_sentence, augmented_labels = token_deletion(sentence, labels)
            augmented_data.append((augmented_sentence, augmented_labels))
        if random.random() < 0.3:  # Apply random insertion to 30% of the data
            augmented_sentence, augmented_labels = random_insertion(sentence, labels)
            augmented_data.append((augmented_sentence, augmented_labels))
        # Add noisy labels (optional)
        if random.random() < 0.2:  # Apply noisy labels to 20% of the data
            noisy_augmented_labels = noisy_labels(labels)
            augmented_data.append((sentence, noisy_augmented_labels))
    
    return augmented_data
