# Character-Level Language Model From Scratch

A tiny generative AI model built from scratch with Python and PyTorch, for learning how language models are trained.

## Project Objective
Understand the full pipeline of training and using a language model: data, vocabulary, encoding, batches, model, loss, backpropagation, optimizer, saving, loading and sampling. No pretrained models or APIs are used.

## What We Build
A character-level bigram model that learns which character tends to follow which, then generates text one character at a time.

## Architecture
Input character ID -> Embedding table (vocab_size x vocab_size) -> Logits -> Softmax -> Sample next character.
The model sees only the previous character, so it cannot memorize whole sentences.

## Technologies
Python 3.10/3.11, PyTorch, NumPy, Matplotlib, VS Code.

## Project Structure
- `data/train.txt` - training text
- `model/bigram_model.pth` - saved weights and vocabulary (created by training)
- `outputs/` - loss plot, loss log, evaluation report
- `train.py` - train and save the model
- `generate.py` - load the model and generate text
- `evaluate.py` - qualitative evaluation
- `requirements.txt` - Python packages
- `.gitignore` - files Git should ignore

## Installation (Windows)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Dataset
About 60 short educational sentences on AI, machine learning, generative AI, neural networks, Python and deep learning.

## Training
```
python train.py
```
Prints model info and the loss every 100 steps, then saves the loss curve and loss log to `outputs/`.

## Model Saving
Weights and the character vocabulary are saved to `model/bigram_model.pth`.

## Text Generation
```
python generate.py "Machine" 200
```

## Evaluation
```
python evaluate.py
```
Writes `outputs/evaluation_report.txt`. This is a basic qualitative check, not a professional LLM benchmark.

## Experiments
Vary learning rate (0.01 / 0.001 / 0.0001), training steps (100 / 1000 / 5000), dataset size, `block_size` and `batch_size`. Change one thing at a time and set `EXPERIMENT_NAME` in `train.py` so plots are not overwritten.

## Limitations
Tiny data, tiny model, one-character context, character-level tokens. Output is word-like gibberish with some real fragments.

## What I Learned
(Write your own notes here: what loss means, why we sample, training vs inference.)

## Future Improvements
Add a train/validation split, temperature control, a longer context (for example an MLP over several characters), and only then attention and Transformers.
