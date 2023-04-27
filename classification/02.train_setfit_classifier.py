# %%

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset, load_metric
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AutoTokenizer
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss

from src.plotting_functions import plot_cm, roc_curve_plot

# %%
dataset = load_dataset("Brian-M-Collins/retracted_articles")

# %%
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")

# %%
max_length = 512


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


# %%
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# %%
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# %%
trainer = SetFitTrainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    loss_class=CosineSimilarityLoss,
    batch_size=4,
    num_iterations=20,  # Number of text pairs to generate for contrastive learning
    num_epochs=1,  # Number of epochs to use for contrastive learning
)

trainer.train()

# %%
metrics = trainer.evaluate()

# %%
# issues pushing this model to the HF hub, saved locally and loaded for evaluation
# model = SetFitModel.from_pretrained("sgugger/tiny-distilbert-classification")

# %%
preds_df = pd.DataFrame(
    {"text": tokenized_datasets["test"]["text"], "label": tokenized_datasets["test"]["label"]}
)


# %%
def get_predicted_probabilities(model: SetFitModel, text: list[str]):
    embeddings = model.model_body.encode(
        text, normalize_embeddings=model.normalize_embeddings, convert_to_tensor=False
    )
    output = np.array(
        model.model_head.predict_proba(embeddings)
    )  # Extract only the probs for the positive class
    outputs_pos = output[:, 1].T
    return outputs_pos  # Get predicted probabilities


# %%
predicted_probabilities = get_predicted_probabilities(model, preds_df["text"].tolist())


# %%
preds = model(preds_df["text"].tolist())


# %%
def return_label(val):
    if val == 1:
        result = 1
    else:
        result = 0
    return result


preds = map(return_label, preds)
final_preds = list(preds)

# %%
preds_df["model_predictions"] = final_preds

# %%
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")


def compute_metrics(predictions, labels):
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# %%
metrics = compute_metrics(preds_df["model_predictions"].tolist(), preds_df["label"].tolist())

# %%
cm_plt = plot_cm(
    preds_df["label"].tolist(),
    preds_df["model_predictions"].tolist(),
    "Retraction Prediction Confusion Matrix for SetFit Model",
)

# %%
fpr, tpr, _ = roc_curve(preds_df["label"].tolist(), preds_df["model_predictions"].tolist())
roc_auc = auc(fpr, tpr)
roc_plot = roc_curve_plot(fpr, tpr, roc_auc)

# %%
model.push_to_hub("Brian-M-Collins/setfit_retractions_e_2")

# %%
