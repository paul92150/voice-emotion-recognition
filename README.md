# Voice Emotion Recognition

This project implements a machine learning pipeline for the classification of emotions from human voice recordings, using MFCC features as input.

It supports multiple supervised models (SVM, Logistic Regression), provides feature selection via Mutual Information, and includes AutoML experiments with TPOT and H2O. The codebase is modular, script-based (no notebooks), and designed for reproducibility.

## Features

- MFCC feature extraction with Librosa
- Feature selection using Mutual Information
- Hyperparameter tuning via grid search and cross-validation
- AutoML experiments using TPOT and H2O
- Evaluation metrics: precision, recall, F1-score, accuracy
- Clean train/test split and preprocessing pipeline
- Structured, modular codebase without reliance on notebooks

## Project Structure

```text
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

## Installation

```bash
git clone https://github.com/Paul92150/voice-emotion-recognition.git
cd voice-emotion-recognition
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Setup

The dataset should be placed in:

```text
data/raw/wav/
```

Supported datasets include:
- Emo-DB (Berlin Database of Emotional Speech)
- CREMA-D
- RAVDESS
- Other compatible WAV-based datasets

The `data/raw/wav/` folder is excluded from version control via `.gitignore` to prevent uploading personal or large files.

## Emotion Label Mapping

By default, the system expects emotion labels to be encoded in the filenames (as is common with datasets like Emo-DB or CREMA-D). For example, using Emo-DB:

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

If you are using a different dataset, make sure to:
- Update the `emotion_mapping` dictionary
- Adjust the `label_position` (character index in the filename)

These values are typically configured at the top of the relevant scripts (e.g., `train_svm.py`, `automl_tpot.py`).

Files with missing or unmapped labels are automatically skipped during training.

## How to Run

### Train an SVM Classifier

```bash
python scripts/train_svm.py
```

### Run Grid Search for Optimal SVM Hyperparameters

```bash
python scripts/grid_search.py
```

### Run TPOT AutoML

```bash
python scripts/automl_tpot.py
```

### Perform Feature Selection (Mutual Information)

```bash
python scripts/feature_selection_mi.py
```

## Example Results (TPOT AutoML)

```text
Accuracy: 73%
F1-score (macro avg): 0.69
Best pipeline: MinMaxScaler → RFE → XGBoostClassifier
```

Note: These results were obtained on a relatively small dataset and are comparable to those of manually tuned SVMs.

## Author

**Paul Lemaire**  
MSc Candidate, CentraleSupélec  
LinkedIn: https://www.linkedin.com/in/paul-lemaire-aa0369289  
GitHub: https://github.com/Paul92150

## Future Improvements

- Add support for CSV-based emotion labeling metadata
- Integrate ROC curve visualizations for evaluation
- Add support for additional datasets (e.g., TESS, SAVEE)
- Implement a configuration file (`config.yaml`) for parameter control
- Develop a test suite to validate pipeline integrity

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

