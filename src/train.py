import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.data_loader import load_data
from src.dataset import IntentDataset

def train():
    # 1️⃣ المسارات
    data_path = "data/raw/intents.csv"
    model_name = "bert-base-multilingual-cased"
    output_dir = "models/bert_intent_model"

    # 2️⃣ تحميل البيانات
    df, label_encoder = load_data(data_path)

    # 3️⃣ تحميل النموذج والمحول (Tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_)
    )

    # 4️⃣ تجهيز Dataset
    dataset = IntentDataset(
        df["text"].tolist(),
        df["label"].tolist(),
        tokenizer
    )

    # 5️⃣ إعدادات التدريب
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,                 # لجعل التدريب سريع
        per_device_train_batch_size=4,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none"
    )

    # 6️⃣ تهيئة Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # 7️⃣ بدء التدريب
    trainer.train()

    # 8️⃣ حفظ النموذج بعد التدريب
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("تم تدريب النموذج وحفظه بنجاح ✅")

if __name__ == "__main__":
    train()
