import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
DATA_PATH = "data/train.txt"
MODEL_PATH = "model/context_model.pth"

context_size = 8        # how many previous characters the model can see
embedding_dim = 32      # size of the vector that represents each character
hidden_size = 256       # number of neurons in the hidden layer
batch_size = 64
learning_rate = 0.003
max_steps = 5000
print_every = 250
seed = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed)

# ----------------------------------------------------------------------
# Data, vocabulary, encoder/decoder (same ideas as train.py)
# ----------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"ERROR: Could not find '{DATA_PATH}'. Run from the project folder.")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()
if len(text.strip()) <= context_size + 1:
    raise SystemExit("ERROR: The dataset is empty or shorter than context_size.")

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batch():
    # Input: `context_size` characters. Target: the ONE character that follows them.
    starts = torch.randint(0, len(data) - context_size, (batch_size,))
    x = torch.stack([data[i : i + context_size] for i in starts])   # [B, context_size]
    y = torch.stack([data[i + context_size] for i in starts])       # [B]
    return x.to(device), y.to(device)


# ----------------------------------------------------------------------
# Model: looks at the last `context_size` characters, not just one
# ----------------------------------------------------------------------
class ContextModel(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(context_size * embedding_dim, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)                         # [B, context_size, embedding_dim]
        flat = emb.reshape(x.shape[0], -1)              # [B, context_size * embedding_dim]
        h = torch.tanh(self.hidden(flat))               # [B, hidden_size]
        return self.output(h)                           # [B, vocab_size] logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_text(model, prompt, max_new_tokens=200, temperature=0.8):
    unknown = sorted(set(c for c in prompt if c not in stoi))
    if len(prompt) == 0 or unknown:
        return None
    model.eval()
    ids = [stoi[c] for c in prompt]
    pad_id = stoi["\n"]    # used if the prompt is shorter than context_size
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-context_size:]
            context = [pad_id] * (context_size - len(context)) + context
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)[0] / temperature          # lower temperature = safer choices
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            ids.append(next_id)
    model.train()
    return "".join([itos[i] for i in ids])


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------
model = ContextModel(vocab_size, context_size, embedding_dim, hidden_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("========== Model information ==========")
print("Device:              ", device)
print("Dataset size (chars):", len(text))
print("Vocabulary size:     ", vocab_size)
print("Context size:        ", context_size)
print("Trainable parameters:", count_parameters(model))
print("Batch size:          ", batch_size)
print("Learning rate:       ", learning_rate)
print("Training steps:      ", max_steps)
print("=======================================")

loss_history = []
for step in range(max_steps):
    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if step % print_every == 0 or step == max_steps - 1:
        print(f"Step: {step}  Loss: {loss.item():.4f}")

# ----------------------------------------------------------------------
# Save model and loss plot
# ----------------------------------------------------------------------
os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "chars": chars,
        "context_size": context_size,
        "embedding_dim": embedding_dim,
        "hidden_size": hidden_size,
    },
    MODEL_PATH,
)
print(f"Saved model to {MODEL_PATH}")

losses = np.array(loss_history)
window = min(50, len(losses))
smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
plt.figure(figsize=(8, 5))
plt.plot(losses, alpha=0.3, label="loss per step (noisy)")
plt.plot(np.arange(window - 1, len(losses)), smooth, label=f"moving average ({window} steps)")
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training Step vs Loss (context model)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("outputs/loss_curve_context.png", dpi=120)
plt.close()
print("Saved loss plot to outputs/loss_curve_context.png")

# ----------------------------------------------------------------------
# Generate samples
# ----------------------------------------------------------------------
print("-" * 50)
for prompt in ["Machine", "Artificial", "Generative", "Python", "Neural", "Learning"]:
    for temperature in [0.5, 0.8]:
        output = generate_text(model, prompt, max_new_tokens=200, temperature=temperature)
        print(f"Prompt: {prompt}   (temperature {temperature})")
        print(output)
        print()
