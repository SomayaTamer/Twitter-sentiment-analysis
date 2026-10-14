# Twitter Sentiment Analysis with Machine Learning and Deep Learning

A Natural Language Processing (NLP) project that classifies tweets as **positive** or **negative** sentiment. The project compares classical machine learning models, neural networks, and a fine-tuned BERT transformer using the Sentiment140 dataset.

## Project Overview

Social media contains large amounts of opinionated text. Automatically identifying sentiment can help summarize public reactions and understand opinions at scale.

This project builds and evaluates multiple approaches to binary sentiment classification, from traditional TF-IDF features to pretrained word embeddings and transformer-based transfer learning.

> **Scope note:** The implementation documented here focuses on sentiment analysis. Although the original notebook title mentions fake news detection, a separate fake-news classification pipeline is not documented in the current implementation.

## Dataset

- **Name:** Sentiment140
- **Source:** [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Original size:** 1.6 million tweets
- **Working sample:** 50,000 tweets for the experiments documented in the notebook
- **Labels:** `0` = negative, `4` = positive; positive labels are converted to `1` for binary classification.

The dataset is downloaded with `kagglehub`. Dataset files are not included in this repository; please follow Kaggle's terms and dataset conditions.

## Workflow

1. **Data loading and preprocessing**
   - Load the Sentiment140 dataset.
   - Convert sentiment labels to binary values.
   - Normalize text to lowercase.
   - Remove URLs, punctuation, numbers, and English stop words.

2. **Exploratory data analysis**
   - Inspect dataset information and class distribution.
   - Analyze tweet lengths and frequent words.
   - Generate a word cloud.

3. **Feature extraction and dimensionality reduction**
   - Represent text using TF-IDF with up to 5,000 features.
   - Apply Truncated SVD / Latent Semantic Analysis (LSA).
   - Visualize the text data in two dimensions and examine explained variance.

4. **Classical machine learning**
   - K-Nearest Neighbors (KNN)
   - Multinomial Naive Bayes
   - Random Forest
   - XGBoost
   - Experiment with selected hyperparameters.

5. **Deep learning**
   - Tokenize tweets and pad sequences.
   - Use pretrained GloVe 100-dimensional word embeddings.
   - Build a 1D Convolutional Neural Network (CNN).
   - Compare Adam, SGD, and RMSprop optimizers.
   - Experiment with Simple RNN, GRU, and LSTM architectures.

6. **Transfer learning**
   - Fine-tune `bert-base-uncased` for binary sentiment classification using Hugging Face Transformers and PyTorch.

7. **Evaluation**
   - Accuracy, precision, recall, and F1-score.
   - ROC curve and AUC where calculated.
   - Confusion matrices, training curves, and model comparisons.

## Technologies

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- NLTK
- scikit-learn
- XGBoost
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- GloVe word embeddings
- KaggleHub

## Reported Results

The notebook reports the following representative accuracy values across its experiments. These runs do not all use an identical experimental setup, so treat this as an overview of the notebook results rather than a perfectly controlled benchmark.

| Model / experiment | Reported accuracy |
|---|---:|
| Fine-tuned BERT | 79.51% |
| Multinomial Naive Bayes | 74.57% |
| CNN (best checkpoint) | 72.57% |
| CNN + RMSprop | 72.49% |
| CNN + Adam | 71.82% |
| XGBoost (initial configuration) | 71.97% |
| Random Forest (initial configuration) | 70.49% |
| Simple RNN | 51.42% |
| GRU | 50.23% |
| LSTM | 49.77% |
| KNN (k = 3, initial run) | 58.85% |

**Best reported model:** Fine-tuned BERT, with approximately **79.5% accuracy** in the documented evaluation.

### Key observations

- BERT achieved the strongest reported accuracy among the models evaluated.
- Naive Bayes provided a competitive baseline with TF-IDF features.
- The CNN experiments showed different results depending on the optimizer.
- The RNN, GRU, and LSTM experiments did not learn useful sentiment patterns in their documented runs, with accuracy near 50%.
- The results highlight the importance of preprocessing, architecture choices, and careful evaluation.

## Getting Started

The notebook was developed in a Google Colab-style environment. The easiest way to reproduce the experiments is to open the notebook in Google Colab and run its cells in order.

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

Replace `<your-username>` with your GitHub username.

### 2. Create a virtual environment (optional, recommended for local use)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Access the dataset

The notebook uses KaggleHub to download the dataset:

```python
import kagglehub

path = kagglehub.dataset_download("kazanova/sentiment140")
print(path)
```

Depending on your environment, Kaggle authentication or configuration may be required.

### 5. Run the notebook

Open `notebooks/sentiment_analysis.ipynb` in Google Colab or Jupyter Notebook and execute the cells in order. The GloVe-based experiments also download pretrained GloVe embeddings, which require additional disk space and download time.

## Suggested Repository Structure

```text
twitter-sentiment-analysis/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── sentiment_analysis.ipynb
```

Keep datasets, downloaded GloVe files, virtual environments, model checkpoints, and other large generated artifacts out of Git unless there is a specific reason to version them.

## Important Evaluation Note

In the notebook, the test split is also used as validation data in some neural-network experiments, and test-set performance is used during some hyperparameter comparisons. This can make reported results optimistic and means the comparisons are not fully independent.

For a more rigorous experiment, split the data into separate training, validation, and test sets. Use the validation set for model selection and hyperparameter tuning, and evaluate the final selected model on the test set only once. Also use consistent splits and metrics when comparing models.

## Future Improvements

- Use consistent stratified train, validation, and test splits for all models.
- Tune models using validation data rather than the test set.
- Add a simple interface for entering a tweet and predicting its sentiment.
- Save and document the final model and preprocessing pipeline for reproducible inference.
- If fake news detection is a project goal, add a separate labeled fake-news dataset and a clearly evaluated classification pipeline.

## Acknowledgements

- [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---

**Disclaimer:** This project is for educational and experimental purposes. Model predictions reflect patterns learned from the dataset and should not be treated as definitive interpretations of a person's opinions.
