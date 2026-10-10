# SMS Spam Classifier with Naive Bayes

A small, dependency-free Python mini-project that labels SMS-style messages as **ham** or **spam** using a Multinomial Naive Bayes model implemented from scratch. It includes a stratified hold-out evaluation and reports spam precision, recall, F1, accuracy, and confusion counts.

## What it demonstrates

- Text tokenization and bag-of-words features
- Multinomial Naive Bayes with Laplace smoothing and log probabilities
- Reproducible stratified train/test splitting
- Precision and recall, including false-positive and false-negative counts
- A tiny local CSV dataset and standard-library unit tests

The included dataset is synthetic and for learning only. It is not representative of real-world messaging, and this model should not be used to make consequential decisions.

## Requirements

- Python 3.10 or newer
- No third-party packages (see `requirements.txt`)

## Setup

```bash
git clone https://github.com/mohan843/spam-classifier-naive-bayes.git
cd spam-classifier-naive-bayes
python -m venv .venv
```

Activate the virtual environment if desired. There are no packages to install. If a tooling workflow expects a requirements install, run `python -m pip install -r requirements.txt` (the file intentionally declares no external dependencies).

## Usage

Evaluate on the bundled sample messages:

```bash
python spam_classifier.py
```

Classify a message after the evaluation run (the model is refit on all bundled examples for this optional prediction):

```bash
python spam_classifier.py --predict "Claim your free prize now"
```

Use a different CSV with `label` and `message` columns, adjust the held-out fraction, or change the deterministic split seed:

```bash
python spam_classifier.py --data data/messages.csv --test-size 0.25 --seed 7
```

The output includes **spam precision** (how many messages predicted spam really are spam) and **spam recall** (how many actual spam messages were caught). Tune the threshold/model or inspect the confusion counts according to which error is more costly.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Project files

- `spam_classifier.py` - tokenizer, classifier, CSV loader, evaluation metrics, and CLI
- `data/messages.csv` - small synthetic ham/spam training and evaluation sample
- `tests/test_spam_classifier.py` - unit tests for model behavior, metrics, splitting, and CSV validation
- `requirements.txt` - notes that runtime dependencies are standard-library only
