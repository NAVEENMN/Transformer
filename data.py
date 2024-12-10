import os
import torch
import re
from collections import Counter
from torch.utils.data import Dataset

# File paths
train_file = os.path.join("dataset", "train.dat")
test_file = os.path.join("dataset", "test.dat")

# Preprocessing and Vocabulary Building
def tokenize_and_build_vocab(texts, vocab_size=10000):
    tokens = []
    for text in texts:
        tokens.extend(text.split())
    counter = Counter(tokens)
    # Reserve tokens for <PAD> and <UNK>
    most_common = counter.most_common(vocab_size - 2)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphanumeric chars
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def encode_text(text, vocab, seq_length=10):
    tokens = text.split()
    encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    sequences = []
    for i in range(len(encoded) - seq_length):
        input_seq = encoded[i:i + seq_length]
        target = encoded[i + seq_length]
        sequences.append((input_seq, target))
    return sequences

def prepare_data():
    vocab_size = 10000
    seq_length = 10

    with open(train_file, "r") as f:
        train_texts = [preprocess_text(line) for line in f.readlines()]
    with open(test_file, "r") as f:
        test_texts = [preprocess_text(line) for line in f.readlines()]

    vocab = tokenize_and_build_vocab(train_texts, vocab_size)
    train_sequences = [seq for text in train_texts for seq in encode_text(text, vocab, seq_length)]
    test_sequences = [seq for text in test_texts for seq in encode_text(text, vocab, seq_length)]

    # Create validation split from train_sequences
    val_ratio = 0.2
    train_size = int((1 - val_ratio) * len(train_sequences))
    val_sequences = train_sequences[train_size:]
    train_sequences = train_sequences[:train_size]

    return train_sequences, val_sequences, test_sequences, vocab

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.inputs = torch.tensor([seq[0] for seq in sequences], dtype=torch.long)
        self.targets = torch.tensor([seq[1] for seq in sequences], dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def get_dataset():
    print("Preparing data...")
    train_sequences, val_sequences, test_sequences, vocab = prepare_data()
    train_dataset = TextDataset(train_sequences)
    val_dataset = TextDataset(val_sequences)
    test_dataset = TextDataset(test_sequences)
    return train_dataset, val_dataset, test_dataset, vocab