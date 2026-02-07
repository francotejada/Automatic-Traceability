Repository: francotejada/automatic-traceability
File: 3_Prediction_components_2025.ipynb
Lines: 113

Estimated tokens: 871

Directory structure:
└── 3_Prediction_components_2025.ipynb


================================================
FILE: Prediction/3_Prediction_components_2025.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
<a href="https://colab.research.google.com/github/francotejada/Automatic-Traceability/blob/main/Prediction/3_Prediction_components_2025.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# Instalación de paquetes necesarios
!pip install -q transformers datasets evaluate accelerate openpyxl scikit-learn

import os
import torch
import pandas as pd
import evaluate

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Desactivar logging de Weights & Biases
os.environ["WANDB_DISABLED"] = "true"

# Cargar datos desde Excel
df = pd.read_excel('/content/data_bugzilla.xlsx')
df.columns = ['component', 'title', 'description']
df.dropna(inplace=True)

# Preparar textos y etiquetas
df['text'] = df['title'].astype(str) + ' ' + df['description'].astype(str)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['component'])
num_labels = len(label_encoder.classes_)

# División en conjuntos de entrenamiento y prueba
train_df, test_df = train_test_split(df[['text', 'label']], test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Leer nuevo vocabulario
with open("vocab.txt", "r") as text_file:
    new_tokens = text_file.readlines()
print(new_tokens)
print(len(new_tokens))

# Cargar modelo y tokenizer DeBERTa v3
model_checkpoint = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

print("[ BEFORE ] tokenizer vocab size:", len(tokenizer))
added_tokens = tokenizer.add_tokens(new_tokens)
print("[ AFTER ] tokenizer vocab size:", len(tokenizer))
print("added_tokens:", added_tokens)

# Redimensionar embedding del modelo
model.resize_token_embeddings(len(tokenizer))

# Función de tokenización
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
test_dataset = test_dataset.map(tokenize_function, batched=True).remove_columns(["text"])

# Métrica
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=15,
    weight_decay=0.01
)

# Inicializar entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Entrenamiento y evaluación
trainer.train()
trainer.evaluate()

# Guardar modelo, tokenizer y clases
model.save_pretrained("./deberta-bugzilla")
tokenizer.save_pretrained("./deberta-bugzilla")
pd.Series(label_encoder.classes_).to_csv("label_classes.csv", index=False)

# Liberar memoria
from torch.cuda import empty_cache
empty_cache()

