# BGE EVA Intent Classifier

Fine-tunes `BAAI/bge-small-en-v1.5` to classify EVA voice commands into the intent labels in `intents.json`. The dataset contains 3,600 labeled phrases: 100 per intent, split evenly between normal language and astronaut-style radio phrasing.

## Setup

From this directory, install the dependencies into your Python environment:

```sh
python -m pip install -r requirements.txt
```

The first training or prediction run downloads the BGE tokenizer and encoder from Hugging Face.

## Train

```sh
python train.py
```

Training saves `runs/model.pt`, `runs/label2id.json`, and `runs/metrics.json`. Paraphrases sharing a `family` value stay together in one split; rows whose family starts with `training_only_` are kept out of validation and test.

Options include `--epochs`, `--batch-size`, `--learning-rate`, `--patience`, `--output-dir`, and a positional path to a JSON or JSONL training file.

## Predict

After training:

```sh
python predict.py "How much oxygen is in the primary tank?"
```

The JSON response contains the top five intents and their softmax scores. The `unhandled` intent is trained for speech that is not an EVA command.

## Data format

`trainingData.json` rows use `text`, `intent`, `style`, and `family` fields. `intents.json` is the canonical label and description list. The 1,800 selected original examples are preserved in `normal_examples.json`; run `python generate_data.py` to rebuild the balanced dataset from that source and the included astronaut-radio phrase patterns. Generated astronaut examples use a `training_only_...` family name, keeping them out of held-out metrics.

After changing the dataset, run `python train.py` to update the saved model and metrics.

## Astronaut-domain encoder

`python train_primer.py ../astronaut_primer.json` continues pretraining the BGE encoder
with a symmetric in-batch contrastive objective over each row's `anchor` and `positive`
text. The adapted encoder and matching tokenizer are saved to
`runs/astronaut-primer-encoder/`; the intent-classifier artifacts in `runs/` are not
modified. Load the adapted encoder for embedding or downstream fine-tuning with
`AutoModel.from_pretrained("runs/astronaut-primer-encoder")` and
`AutoTokenizer.from_pretrained("runs/astronaut-primer-encoder")`.
