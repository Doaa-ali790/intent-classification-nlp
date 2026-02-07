# Intent Classification NLP Project

This project implements **Intent Classification** for Arabic text using **Multilingual BERT**.  
It can classify user messages into different predefined intents such as booking a ticket, password reset, or saying thanks.

---

## 📂 Project Structure

intent-classification-nlp/
├── data/
│ └── raw/
│ └── intents.csv # Training dataset
├── src/
│ ├── init.py
│ ├── data_loader.py # Load and preprocess the data
│ ├── dataset.py # Custom Dataset class
│ ├── model.py # Load BERT model and tokenizer
│ ├── train.py # Training script
│ └── predict.py # Prediction script
├── requirements.txt # Python dependencies
└── README.md


> **Note:** The `models/` folder containing the trained BERT model is **not included** in this repository due to its large size (>400MB).  
> The trained model can be downloaded separately from Google Drive or HuggingFace Hub.

---

## ⚡ Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```
🚀 How to Run
1️⃣ Training the Model
After installing the dependencies, you can train the model using:

python src/train.py
The model will be saved to:

models/bert_intent_model
2️⃣ Using the Trained Model
You can predict the intent of any Arabic sentence using:

from src.predict import predict_intent

predict_intent("أريد حجز رحلة")
predict_intent("نسيت كلمة المرور")
predict_intent("شكرا لك")
