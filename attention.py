import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import get_dataset
import logging
import matplotlib.pyplot as plt
from setproctitle import setproctitle

# TODO: Organize the code and extract reusable components into separate modules and source control it.

# Set up logging
setproctitle("proc_nmysore_exp")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
os.makedirs("results", exist_ok=True)
file_handler = logging.FileHandler("results/logs.txt", mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class AttentionMechanism(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionMechanism, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        Q = self.query(inputs)
        K = self.key(inputs)
        V = self.value(inputs)

        attention_scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / torch.sqrt(
            torch.tensor(Q.size(-1), dtype=torch.float32, device=Q.device)
        )
        attention_probs = self.softmax(attention_scores)
        weighted_sum = torch.matmul(attention_probs, V)

        return weighted_sum, attention_scores, attention_probs

class NextWordPredictorWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(NextWordPredictorWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = AttentionMechanism(embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeddings = self.embedding(x)  # (batch, seq_length, embed_dim)
        attn_output, attn_scores, attn_probs = self.attention(embeddings)
        _, (hidden, _) = self.lstm(attn_output)
        logits = self.fc(hidden.squeeze(0))
        return logits, attn_scores, attn_probs

def evaluate(model, data_loader, criterion):
    logger.info("Evaluating...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def visualize_fixed_attention(model, device, vertical_words, horizontal_words, vocab, epoch, results_dir="results"):
    # Create a reverse vocabulary map
    inv_vocab = {idx: word for word, idx in vocab.items()}

    # Encode the fixed 10-word sequence
    # The sequence is vertical_words followed by horizontal_words
    full_words = vertical_words + horizontal_words
    input_ids = [vocab.get(w, vocab["<UNK>"]) for w in full_words]
    input_seq = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1,10)

    model.eval()
    with torch.no_grad():
        _, attn_scores, attn_probs = model(input_seq)
        # attn_probs: (1, 10, 10)
        attn_scores = attn_scores[0].cpu().numpy()  # (10,10)

    # Extract the 5x5 block:
    # Rows 0-4 correspond to vertical words
    # Columns 5-9 correspond to horizontal words
    attn_block = attn_scores[0:5, 5:10]

    # Plot the attention block
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_block, cmap='viridis', aspect='auto')
    plt.colorbar()

    plt.xticks(range(5), horizontal_words, rotation=45, ha='right')
    plt.yticks(range(5), vertical_words)

    plt.title(f"Fixed Attention Map at Epoch {epoch+1}")
    plt.tight_layout()

    # Save the figure
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"fixed_attention_epoch_{epoch+1}.png"))
    plt.close()


def train():
    vocab_size = 10000
    seq_length = 10
    embed_dim = 12
    hidden_dim = 256
    batch_size = 32
    epochs = 20
    learning_rate = 0.001

    train_dataset, val_dataset, test_dataset, vocab = get_dataset()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = NextWordPredictorWithAttention(vocab_size, embed_dim, hidden_dim)

    device_ids = [0]
    if torch.cuda.is_available() and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        device = torch.device(f"cuda:{device_ids[0]}")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{device_ids[0]}")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define fixed words to track across epochs
    # 5 words for vertical axis and 5 for horizontal axis
    # Choose common words that likely appear in the vocab
    vertical_words = ["after", "intravenous", "fdg", "administration", "metastases"]
    horizontal_words = ["nodal", "and", "peritoneal", "also", "showed"]

    train_losses = []
    val_losses = []

    # Pre-select a fixed set of samples from val_dataset to visualize each epoch
    # For example, the first 10 samples (or fewer if not enough)
    num_visualizations = min(10, len(val_dataset))
    visualization_samples = [val_dataset[i] for i in range(num_visualizations)]

    logger.info("Training started...")
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = evaluate(model, val_loader, criterion)
        epoch_duration = time.time() - epoch_start
        avg_train_loss = train_loss / len(train_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Time: {epoch_duration:.2f}s")

        # Visualize fixed attention map for the same 10 words each epoch
        visualize_fixed_attention(model, device, vertical_words, horizontal_words, vocab, epoch, results_dir="results")

    # After training, evaluate on test set
    test_loss = evaluate(model, test_loader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}")

    # Plotting the losses
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/loss_plot.png")  # Save the figure locally
    plt.close()  # Close the figure
    logger.info("Training completed.")

if __name__ == "__main__":
    train()