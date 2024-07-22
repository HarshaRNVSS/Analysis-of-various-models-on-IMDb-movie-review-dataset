# NLP Final Project

This repository contains the final project for the Natural Language Processing (NLP) course. The project focuses on sentiment analysis using the IMDB dataset.

## Project Structure

- `Analysis-of-various-models-on-IMDb-movie-review-dataset.ipynb`: The main Jupyter notebook containing all the code and analysis for the project.

## Dataset

The dataset used for this project is the IMDB Dataset, which contains 50,000 movie reviews categorized into positive and negative sentiments.

## Steps Involved

### 1. Importing the Dataset

The dataset is imported using `pandas`.

```python
import pandas as pd
df = pd.read_csv("IMDB Dataset.csv")
```
### 2. Data Exploration

The distribution of sentiments in the dataset is checked.
``` python
print(df['sentiment'].value_counts())
```
### 3. Preprocessing the Data

Preprocessing steps include:

* Converting text to lowercase
* Removing punctuation
* Tokenizing the text
* Removing stopwords
  
Libraries used: `string`, `nltk`

``` python
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if not word in stop_words]
    text = ' '.join(tokens)
    return text

df['review'] = df['review'].apply(preprocess_text)
```
### 4. TF-IDF
TF-IDF (Term Frequency-Inverse Document Frequency) converts the text data into a numerical format suitable for machine learning models.

## How to Run
* Clone this repository.
* Open NLP_Final_Project.ipynb in Jupyter Notebook or JupyterLab.
* Run the cells in the notebook to reproduce the results.

## Requirements
* Python 3.x
* pandas
* nltk
* scikit-learn (if used for model building)
* Install the necessary packages using:
  ```bash
  pip install pandas nltk scikit-learn
```
