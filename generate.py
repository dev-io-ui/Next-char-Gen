import os
import sys

import torch
import torch.nn as nn

MODEL_PATH = "model/bigram_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"


# The architecture must be recreated EXACTLY as in train.py.
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = self.loss_fn(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise SystemExit(
            f"ERROR: Model file '{path}' was not found.\n"
            "Fix: train the model first by running:  python train.py"
        )
    checkpoint = torch.load(path, map_location=device)
    chars = checkpoint["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    model = BigramModel(len(chars)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()          # inference mode
    return model, stoi, itos


def generate_text(model, prompt, stoi, itos, max_new_tokens=150):
    if len(prompt) == 0:
        raise ValueError("The prompt is empty. Type at least one character.")
    unknown = sorted(set(c for c in prompt if c not in stoi))
    if unknown:
        raise ValueError(
            f"The prompt contains characters the model never saw in training: {unknown}\n"
            f"Characters it knows: {''.join(sorted(stoi.keys()))!r}"
        )

    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(idx)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
    return "".join([itos[i] for i in idx[0].tolist()])


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Machine"
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 150

    model, stoi, itos = load_model()
    try:
        output = generate_text(model, prompt, stoi, itos, max_new_tokens=length)
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}")

    print("Prompt:   ", prompt)
    print("Generated:", output)
