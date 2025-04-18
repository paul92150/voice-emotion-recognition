# 🎙️ Voice Emotion Recognition

This project implements a machine learning pipeline to classify **emotions in human voice** using MFCC features extracted from audio recordings.

It supports multiple models (SVM, Logistic Regression) and provides tools for feature selection (Mutual Information), hyperparameter tuning, and AutoML experiments (TPOT, H2O).

---

## 🚀 Features

- 🎧 MFCC feature extraction using Librosa  
- 📊 Feature selection (Mutual Information)  
- 🔍 Grid Search with cross-validation for SVM  
- 🤖 AutoML with TPOT and H2O  
- 📈 Model evaluation with precision, recall, F1-score  
- 📁 Modular and organized codebase (no notebooks!)  
- ✅ Clean training/testing split and preprocessing  

---

## 🗂️ Project Structure

```
voice-emotion-recognition/
├── data/
│   ├── raw/                ← Place your audio dataset here (ignored by Git)
├── models/                 ← Trained models are saved here
├── scripts/                ← All run scripts (training, grid search, etc.)
├── src/                    ← Core logic (feature extraction, preprocessing...)
│   ├── features.py
│   ├── preprocessing.py
│   ├── automl.py
│   ├── evaluate.py
│   └── utils.py
├── README.md               ← This file
├── requirements.txt        ← Python dependencies
└── .gitignore              ← Keeps datasets and model files out of Git
```

---

## 📦 Installation

1. **Clone the repo**:

```bash
git clone https://github.com/Paul92150/voice-emotion-recognition.git
cd voice-emotion-recognition
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## 📁 Data Setup

This repo assumes your dataset is stored in:

```
data/raw/wav/
```

You can use datasets like:
- **Emo-DB** (Berlin Database of Emotional Speech)
- **CREMA-D**, **RAVDESS**, etc.

> ⚠️ The `data/raw/wav/` folder is ignored by Git to keep the repo light and personal-data-free.

---

## 🏷️ Emotion Label Mapping

This project assumes emotion labels are encoded in the filenames using a known convention (e.g., Emo-DB or CREMA-D).

If you're using Emo-DB, you can use the default mapping:

```python
emotion_mapping = {
    'W': 'anger',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'anxiety',
    'F': 'happiness',
    'T': 'sadness',
    'N': 'neutral'
}
```

📌 **Using a different dataset?** Make sure to update both:
- the `emotion_mapping` dictionary (character → emotion name)
- the `label_position` (index of the label character in filenames)

These are typically defined at the top of each script (e.g., `train_svm.py`, `automl_tpot.py`, etc.).

> ⚠️ If no mapping is provided or a label is missing from the mapping, the corresponding file will be **skipped** during training.

---

## 🧪 How to Run

### 🧠 Train an SVM:

```bash
python scripts/train_svm.py
```

### 🔍 Run Grid Search for Best SVM Parameters:

```bash
python scripts/grid_search.py
```

### 🤖 Run TPOT AutoML:

```bash
python scripts/automl_tpot.py
```

### 📊 Analyze MFCCs via Mutual Information:

```bash
python scripts/feature_selection_mi.py
```

---

## 🧠 Example Results (TPOT AutoML)

```
Accuracy: 73%
F1-score (macro avg): 0.69
Best pipeline: MinMaxScaler → RFE → XGBoostClassifier
```

📌 The dataset was relatively small, and AutoML reached results comparable to hand-tuned SVM.

---

## 👤 Author

**Paul Lemaire**  
Student at CentraleSupélec  
[LinkedIn](https://www.linkedin.com/in/paul-lemaire-aa0369289) • [GitHub](https://github.com/Paul92150)

---

## 🛠️ Future Improvements

- 📦 Add support for metadata-based emotion labeling (e.g., from CSV)
- 📊 Add ROC curve visualizations
- 🎯 Integrate more datasets (e.g., RAVDESS, TESS)
- ⚙️ Add configuration file (e.g., `config.yaml`) to make scripts more flexible
- 🧪 Add test suite for pipeline robustness

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
