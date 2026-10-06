import os
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Settings (change these for experiments)
# ----------------------------------------------------------------------
DATA_PATH = "data/train.txt"
MODEL_PATH = "model/bigram_model.pth"
EXPERIMENT_NAME = "baseline"      # used in output file names

batch_size = 16
block_size = 32
learning_rate = 0.001
max_steps = 5000
print_every = 100
seed = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed)           # reproducibility


# ----------------------------------------------------------------------
# Step 3: read the dataset
# ----------------------------------------------------------------------
def read_dataset(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"ERROR: Could not find '{path}'.\n"
            "Fix: run this script from the project folder "
            "(character-text-generator) and make sure data/train.txt exists."
        )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text.strip()) == 0:
        raise SystemExit(f"ERROR: '{path}' is empty. Add some training text to it.")
    return text


text = read_dataset(DATA_PATH)

if len(text) <= block_size + 1:
    raise SystemExit(
        f"ERROR: The dataset has only {len(text)} characters, but block_size is "
        f"{block_size}. Add more text or use a smaller block_size."
    )

print("First 200 characters:")
print(text[:200])
print("-" * 50)

# ----------------------------------------------------------------------
# Step 4-5: vocabulary, encoder, decoder
# ----------------------------------------------------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # string  -> integer
itos = {i: ch for i, ch in enumerate(chars)}   # integer -> string


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join([itos[i] for i in ids])


assert decode(encode(text[:50])) == text[:50], "Encode/decode round trip failed!"

# ----------------------------------------------------------------------
# Step 6: the whole dataset as one tensor of integer IDs
# ----------------------------------------------------------------------
data = torch.tensor(encode(text), dtype=torch.long)


# ----------------------------------------------------------------------
# Step 8: mini-batches
# ----------------------------------------------------------------------
def get_batch():
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


# ----------------------------------------------------------------------
# Step 9-11: the model (embedding lookup table + cross-entropy loss)
# ----------------------------------------------------------------------
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)        # [B, T, C]
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = self.loss_fn(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# Text generation
# ----------------------------------------------------------------------
def generate_text(model, prompt, max_new_tokens=150):
    unknown = sorted(set(c for c in prompt if c not in stoi))
    if len(prompt) == 0 or unknown:
        return None
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)  # [1, T]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(idx)
            last_logits = logits[:, -1, :]                  # [1, C]
            probs = torch.softmax(last_logits, dim=-1)      # scores -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)  # sample
            idx = torch.cat([idx, next_id], dim=1)          # append
    model.train()
    return decode(idx[0].tolist())


# ----------------------------------------------------------------------
# Create model + optimizer
# ----------------------------------------------------------------------
model = BigramModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("========== Model information ==========")
print("Device:              ", device)
print("Dataset size (chars):", len(text))
print("Vocabulary size:     ", vocab_size)
print("Trainable parameters:", count_parameters(model))
print("Batch size:          ", batch_size)
print("Block size:          ", block_size)
print("Learning rate:       ", learning_rate)
print("Training steps:      ", max_steps)
print("=======================================")

xb, yb = get_batch()
_, initial_loss = model(xb, yb)
print(f"Loss before training: {initial_loss.item():.4f}")
print(f"(Pure random guessing would give about ln({vocab_size}) = {math.log(vocab_size):.4f})")
print("-" * 50)

# ----------------------------------------------------------------------
# Step 13: training loop
# ----------------------------------------------------------------------
loss_history = []

for step in range(max_steps):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()      # clear old gradients
    loss.backward()            # backpropagation
    optimizer.step()           # update weights

    loss_history.append(loss.item())
    if step % print_every == 0 or step == max_steps - 1:
        print(f"Step: {step}  Loss: {loss.item():.4f}")

print("-" * 50)

# ----------------------------------------------------------------------
# Save loss values and plot Training Step vs Loss
# ----------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("model", exist_ok=True)

steps = np.arange(len(loss_history))
losses = np.array(loss_history)
np.savetxt(
    f"outputs/loss_history_{EXPERIMENT_NAME}.csv",
    np.column_stack([steps, losses]),
    delimiter=",", header="step,loss", comments="", fmt=["%d", "%.6f"],
)

window = min(50, len(losses))
smooth = np.convolve(losses, np.ones(window) / window, mode="valid")

plt.figure(figsize=(8, 5))
plt.plot(steps, losses, alpha=0.3, label="loss per step (noisy)")
plt.plot(steps[window - 1:], smooth, label=f"moving average ({window} steps)")
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title(f"Training Step vs Loss ({EXPERIMENT_NAME})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"outputs/loss_curve_{EXPERIMENT_NAME}.png", dpi=120)
print(f"Saved loss plot to outputs/loss_curve_{EXPERIMENT_NAME}.png")

# ----------------------------------------------------------------------
# Step 15: save the trained weights (+ vocabulary, needed to decode later)
# ----------------------------------------------------------------------
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "chars": chars,
        "block_size": block_size,
    },
    MODEL_PATH,
)
print(f"Saved model to {MODEL_PATH}")

# ----------------------------------------------------------------------
# Quick generation test
# ----------------------------------------------------------------------
print("-" * 50)
for prompt in ["Machine", "Artificial", "Generative", "Python", "Neural", "Learning"]:
    output = generate_text(model, prompt, max_new_tokens=150)
    print("Prompt:   ", prompt)
    print("Generated:", output)
    print()

plt.show()   # close the plot window to finish
