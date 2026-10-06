import os
import string

import torch

from generate import load_model, generate_text

DATA_PATH = "data/train.txt"
REPORT_PATH = "outputs/evaluation_report.txt"
TEST_PROMPTS = ["Machine", "Artificial", "Generative", "Python", "Neural", "Learning"]
LENGTH = 200

torch.manual_seed(42)   # so the report is repeatable


def clean_words(text):
    words = []
    for w in text.split():
        w = w.strip(string.punctuation).lower()
        if w:
            words.append(w)
    return words


def longest_copied_chunk(generated, train_text):
    """Length of the longest piece of generated text found verbatim in training data."""
    best = 0
    for start in range(len(generated)):
        length = best + 1
        while start + length <= len(generated) and generated[start:start + length] in train_text:
            best = length
            length += 1
    return best


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"ERROR: '{DATA_PATH}' not found. Run from the project folder.")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        train_text = f.read()
    train_words = set(clean_words(train_text))

    model, stoi, itos = load_model()
    lines = []
    lines.append("EVALUATION REPORT (basic qualitative evaluation, NOT a professional LLM benchmark)")
    lines.append("=" * 80)

    for prompt in TEST_PROMPTS:
        out1 = generate_text(model, prompt, stoi, itos, max_new_tokens=LENGTH)
        out2 = generate_text(model, prompt, stoi, itos, max_new_tokens=LENGTH)
        new_text = out1[len(prompt):]

        words = clean_words(new_text)
        # Drop the first/last word: they may be cut off by the generation window.
        inner_words = words[1:-1] if len(words) > 2 else words
        real_ratio = (
            sum(1 for w in inner_words if w in train_words) / len(inner_words)
            if inner_words else 0.0
        )
        repeat_ratio = (
            1 - len(set(words)) / len(words) if words else 0.0
        )
        copied = longest_copied_chunk(new_text, train_text)
        differs = out1 != out2

        notes = []
        notes.append(
            f"Related to training data: {real_ratio:.0%} of words are real training words."
        )
        if real_ratio >= 0.6:
            notes.append("Fairly readable.")
        elif real_ratio >= 0.3:
            notes.append("Partly readable; a mix of real words and gibberish.")
        else:
            notes.append("Mostly unreadable gibberish.")
        notes.append(f"Repeated words: {repeat_ratio:.0%} of words are repeats.")
        notes.append(
            f"Longest chunk copied from training text: {copied} characters"
            + (" (looks like copying)." if copied >= 40 else " (not memorizing long passages).")
        )
        notes.append(
            "Output changes between generations: " + ("yes." if differs else "NO (suspicious).")
        )

        lines.append(f"Prompt: {prompt}")
        lines.append(f"Generated: {out1!r}")
        lines.append("Observations:")
        for n in notes:
            lines.append(f"  - {n}")
        lines.append("-" * 80)

    report = "\n".join(lines)
    print(report)

    os.makedirs("outputs", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
