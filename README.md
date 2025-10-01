# AI Demo Project

This repository contains a demonstration of a simple Machine Learning project using Python.  
The goal is to show hands-on experience with data preprocessing, training models, and evaluating results in a clean, production-friendly structure.

## Features

- Data preprocessing and handling (Pandas, NumPy)
- Machine Learning model training (scikit-learn)
- Example dataset: Iris, SMS Dataset, Titanic Dataset
- Model evaluation with accuracy metric
- Structured code for production (`src/`, `tests/`)
- Easy to extend with new datasets or models

## Test Application / Dataset

We use the Iris dataset from `sklearn.datasets` as a demo dataset.  
The Iris dataset contains 150 samples of iris flowers with 4 features (sepal length, sepal width, petal length, petal width) and 3 classes (species).  

Optional datasets for future experiments:  
- Titanic dataset (survival prediction)  
- MNIST dataset (handwritten digits)  
- Any Kaggle dataset suitable for classification or regression  

## Project Structure

```text
ai-demo-project/
├── src/            # Code for training and inference
│   └── train.py
├── tests/          # Unit tests for models
│   └── test_model.py
├── data/           # Example datasets (optional)
├── requirements.txt
└── README.md 
```

## Instalation
```bash
# Clone the repository
git clone https://github.com/aleon120894/ai-demo-project.git
cd ai-demo-project

# Create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the training script
python src/train.py

# Run unit tests
pytest tests/
```
