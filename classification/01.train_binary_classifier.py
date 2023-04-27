# %%
import string

import nltk
import numpy as np
import pandas as pd
import torch
import evaluate

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")

from datasets import Dataset
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import auc, roc_curve
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from plotting_functions import plot_cm, roc_curve_plot

# %%
retracted_articles = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet"
)
retracted_articles["retracted"] = True
negative_class = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/negative_class.parquet"
)

cols = [
    "doi",
    "article_title",
    "full_source_title",
    "subject_cat_desc",
    "publisher_group",
    "year_published",
    "article_open_access_status",
    "fwci",
    "citations",
    "concat_abstract",
    "embeddings",
    "tsne_1",
    "tsne_2",
    "keywords",
    "cluster_label",
    "retracted",
]

merged = pd.concat([retracted_articles[cols], negative_class[cols]], axis=0)

merged["retracted"] = merged["retracted"].apply(lambda x: 1 if x == True else 0)


# %%
def remove_punctuation(text):
    return "".join([i for i in text if i not in string.punctuation])


# remove punctuation and lower concat abstract
merged["concat_abstract"] = merged["concat_abstract"].apply(
    lambda x: remove_punctuation(x.lower())
)

stop_words = set(stopwords.words("english"))


def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])


merged["concat_abstract"] = merged["concat_abstract"].apply(lambda x: remove_stopwords(x))

lemmatizer = WordNetLemmatizer()


def lemm_abstract(text):
    return " ".join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)])


merged["concat_abstract"] = merged["concat_abstract"].apply(lambda x: lemm_abstract(x))

# stemmer = PorterStemmer()
# def stemm_abstract(text):
#     return ' '.join([stemmer.stem(w) for w in nltk.word_tokenize(text)])

# merged["concat_abstract"] = merged["concat_abstract"].apply(
#     lambda x: stemm_abstract(x)
# )

# %%
dataset = (
    Dataset.from_pandas(
        merged[["doi", "concat_abstract", "retracted"]]
        .set_index("doi")
        .rename(columns={"retracted": "label", "concat_abstract": "text"})
    )
    .class_encode_column("label")
    .train_test_split(test_size=0.3, seed=42, stratify_by_column="label")
)

# %%
# dataset.push_to_hub("Brian-M-Collins/retracted_articles", private=True)

# %%
model = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model)

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
model = AutoModelForSequenceClassification.from_pretrained(
    model, num_labels=2, ignore_mismatched_sizes=True
)


# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# %%
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# %%
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=1
    ),
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# %%
train_results = trainer.train()
test_results = trainer.predict(tokenized_datasets["test"])


# %%
predictions = torch.from_numpy(test_results.predictions)
probabilities = nn.functional.softmax(predictions, dim=-1)

# %%
max_probabilities, max_indices = torch.max(probabilities, dim=1)

# %%
predicted_labels = test_results.predictions.argmax(-1)


# %%
def compute_final_metrics(predictions, labels):
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


metrics = compute_final_metrics(predicted_labels, tokenized_datasets["test"]["label"])

# %%
cm_plt = plot_cm(
    tokenized_datasets["test"]["label"],
    predicted_labels,
    "Retraction Prediction Confusion Matrix - Binary Classifier",
)

# %%
fpr, tpr, _ = roc_curve(tokenized_datasets["test"]["label"], predicted_labels)
roc_auc = auc(fpr, tpr)
roc_plot = roc_curve_plot(fpr, tpr, roc_auc)

# %%
# model.push_to_hub("Brian-M-Collins/predicting_retractions", private=True)

# %%
