# TrustLens

### Explainable Machine Learning for Financial Fraud Detection in UK E-Commerce

TrustLens is an explainable machine learning system developed as part of an MSc research project investigating the use of machine learning and Explainable AI (XAI) for financial fraud detection.

The repository contains the code, notebooks, models, results, and application required to run and reproduce the project.

---

## 1. Repository Structure

```text
TrustLens/
│
├── app/                # TrustLens application
├── data/               # Dataset and data preparation files
├── notebooks/          # Data analysis and machine learning notebooks
├── src/                # Project source code
├── models/             # Saved trained models
├── results/            # Generated results and visualisations
├── requirements.txt    # Python dependencies
└── README.md
```

The exact contents of each directory may vary depending on the final implementation.

---

## 2. Requirements

The project requires:

* Python 3.10+
* Git
* pip
* Jupyter Notebook or JupyterLab
* A web browser

The required Python packages are listed in:

```text
requirements.txt
```

---

## 3. Clone the Repository

Clone the repository using:

```bash
git clone <REPOSITORY_URL>
cd TrustLens
```

---

## 4. Create a Virtual Environment

Creating a virtual environment is recommended.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Confirm that the environment is active and check the Python version:

```bash
python --version
```

---

## 5. Install Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

If Jupyter is not already included in the requirements:

```bash
pip install jupyter
```

---

# 6. Dataset

TrustLens uses the **IEEE-CIS Fraud Detection** dataset.

The dataset is available through the original Kaggle competition:

**IEEE-CIS Fraud Detection — Kaggle**

[Download the IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

The dataset contains separate transaction and identity files which can be joined using `TransactionID`.

### Required files

The training data consists of:

```text
train_transaction.csv
train_identity.csv
```

The Kaggle dataset also provides:

```text
test_transaction.csv
test_identity.csv
sample_submission.csv
```

The complete dataset is approximately 1.35 GB, so it is **not included directly in this repository**.

### Dataset placement

After downloading the dataset, place the required files in the project's data directory:

```text
data/
├── train_transaction.csv
└── train_identity.csv
```

If the repository uses a different data path, update the paths in the relevant configuration or notebook accordingly.

---

# 7. Running the Project

There are two main components:

1. **Research pipeline** — data preparation, analysis, model training, evaluation and explainability.
2. **TrustLens application** — the interactive interface for generating and interpreting fraud predictions.

The research pipeline should be run before the application if the required trained models or processed data have not already been generated.

---

# 8. Research Pipeline

The notebooks are intended to be run in numerical order.

```text
01 → Data Preparation
02 → Exploratory Data Analysis
03 → Model Training
04 → Model Evaluation
05 → Explainability
```

---

## 8.1 Data Preparation

Open:

```text
notebooks/01_data_preparation.ipynb
```

Run the notebook from the first cell to the last.

This stage prepares the raw IEEE-CIS dataset for subsequent analysis and modelling.

The notebook performs the relevant data preparation operations used by the project, including data loading, merging, cleaning and feature preparation.

---

## 8.2 Exploratory Data Analysis

Open:

```text
notebooks/02_exploratory_analysis.ipynb
```

Run all cells.

This notebook examines the prepared dataset and produces the exploratory analysis and visualisations used during the project.

---

## 8.3 Model Training

Open:

```text
notebooks/03_model_training.ipynb
```

Run all cells.

This notebook contains the machine learning training pipeline.

Depending on the final implementation, this stage may generate or update files in:

```text
models/
```

---

## 8.4 Model Evaluation

Open:

```text
notebooks/04_model_evaluation.ipynb
```

Run all cells.

This notebook evaluates the trained model(s) using the metrics implemented in the project.

Generated evaluation outputs may include:

* Confusion matrices
* Classification metrics
* ROC curves
* Precision-Recall curves
* Model comparison results

Results are saved or displayed according to the implementation.

---

## 8.5 Explainability

Open:

```text
notebooks/05_explainability.ipynb
```

Run all cells.

This notebook contains the explainability component of TrustLens.

It demonstrates how the trained model's predictions can be interpreted using the XAI methods implemented in the project.

The outputs include the relevant feature contributions and explanation visualisations.

---

# 9. Running the TrustLens Application

Once the required model and supporting files have been generated, the TrustLens application can be launched.

Run the application using the project's startup command:

```bash
<APPLICATION_START_COMMAND>
```

For example, if the final application uses Streamlit:

```bash
streamlit run app/app.py
```

The terminal will provide a local URL.

Open the URL in a web browser.

---

# 10. Using TrustLens

The general application workflow is:

```text
Start application
       ↓
Enter transaction information
       ↓
Submit transaction
       ↓
Generate prediction
       ↓
View fraud probability
       ↓
View prediction explanation
       ↓
Inspect contributing features
```

The application produces a fraud prediction based on the trained machine learning model.

The explanation component provides additional information about the factors contributing to the prediction.

---

# 11. Reproducing the Complete Pipeline

To reproduce the project from the raw dataset, use the following sequence:

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd TrustLens
```

### 2. Create the virtual environment

```bash
python -m venv .venv
```

Activate the environment using the instructions above.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the IEEE-CIS Fraud Detection dataset from:

[Kaggle — IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

### 5. Place the required dataset files in `data/`

```text
data/
├── train_transaction.csv
└── train_identity.csv
```

### 6. Run the notebooks in order

```text
01_data_preparation.ipynb
02_exploratory_analysis.ipynb
03_model_training.ipynb
04_model_evaluation.ipynb
05_explainability.ipynb
```

### 7. Launch TrustLens

```bash
<APPLICATION_START_COMMAND>
```

---

# 12. Running the Application Without Retraining

If the repository contains the required trained models and processed files, the complete training pipeline does not need to be rerun.

In that case:

```text
Clone repository
      ↓
Install dependencies
      ↓
Download/place dataset if required by the application
      ↓
Start TrustLens
      ↓
Use the application
```

The saved models in the `models/` directory can be used by the application where applicable.

---

# 13. Troubleshooting

### Python version

Check the installed version:

```bash
python --version
```

Use Python 3.10 or the version specified in the project configuration.

### Dependency errors

Upgrade pip and reinstall:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset not found

Check that the required files are located in the expected directory:

```text
data/
```

and that the filenames match those expected by the notebooks.

### Notebook cannot find project files

Start Jupyter from the project root:

```bash
cd TrustLens
jupyter notebook
```

### Model file not found

If the application requires a model that has not been generated, run:

```text
03_model_training.ipynb
```

before starting the application.

### Application does not start

Make sure the virtual environment is active and all dependencies have been installed:

```bash
pip install -r requirements.txt
```

Then run the application using the startup command specified above.

---

# 14. Important Notes

* The raw IEEE-CIS dataset is not stored in this repository.
* The dataset is subject to the terms and rules specified by Kaggle.
* Do not commit the raw dataset to the repository.
* Run the notebooks in the specified order when reproducing the complete pipeline.
* Saved models and generated results should not be unnecessarily regenerated if the purpose is only to use the completed application.
* TrustLens is a research prototype and is not intended to be used as a production fraud detection system.

---

# 15. Project Information

**Project:** TrustLens

**Research Title:**
*Explainable Machine Learning for Financial Fraud Detection in UK E-Commerce: Evaluating the Trade-off Between Accuracy, Transparency and Operational Performance.*

**Programme:** MSc Computer Science and Technology with Business Development

**Institution:** Ulster University

**Research Project:** COM748

**Author:** Oluwayomi Bamidele
