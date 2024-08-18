import re
from pathlib import Path

import contractions
import pickle
import string

import torch
import torch.nn as nn
import torch.optim as optim

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import word_tokenize

from textblob import TextBlob
import dill

# pip install spellchecker, pyspellchecker
# python3 -m nltk.downloader all

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def convert_to_lowercase(text):
    return text.lower()

def remove_extra_whitespace(text):
    return ' '.join(text.split())

def text_cleaning(text):
    lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def correct_spelling(text):
    b = TextBlob(text)
    return b.correct().string

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

# remove emails
def remove_emails(text):
    return re.sub(r'\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,6}\b', '', text)

# remove special characters
def remove_special_characters(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_specs(text):
    return remove_special_characters(remove_emails(remove_urls(text)))

# expanding contractions to full forms for more consistent analysis
def expand_contractions(text):
    return contractions.fix(text)

clean_function_list = [ 
    remove_extra_whitespace,
    remove_punctuation,
    correct_spelling,
    expand_contractions,
    remove_stop_words, 
    lemmatize_words,
    convert_to_lowercase,
    remove_specs
]

def clean_data(text):
    for clean_func in clean_function_list:
        text = clean_func(text)

    return text

# NLTK
def word_tokenizer(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in string.punctuation]
    return tokens

def to_features(words):
    return {word: True for word in words}

def analyze_text_nltk_model(text):
    file_path = Path.cwd() / "sentiment_analysis" / "models" / "naive_bayes_model_with_simple_tokenizer.pkl"
    print(f'Model file: {file_path.resolve()}')

    with open(file_path.resolve(), 'rb') as f:
        naive_bayes_model = pickle.load(f)
    
    predictions = naive_bayes_model.classify((to_features(word_tokenizer(text))))
    print("Predictions naive_bayes_model:", predictions)    
    return -1 if predictions == 0 else 1    


# SKLERN
def analyze_text_sklern_model(text):
    file_path = Path.cwd() / "sentiment_analysis" / "models" / "StackingClassifierLogisticRegressionPlusSVC.pkl"
    print(f'Model file: {file_path.resolve()}')

    with open(file_path.resolve(), 'rb') as f:
        loaded_pipeline = pickle.load(f)
    
    predictions = loaded_pipeline.predict([text])
    print("Predictions StackingClassifier:", predictions)    
    return -1 if predictions[0] == 0 else 1


# PyTorch
def pytorch_preprocess_text(text, vocab, max_length=1024):
    tokens = text.split()
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    
    if len(indices) < max_length:
        indices += [vocab['<pad>']] * (max_length - len(indices))
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

def pytorch_predict(model, text, vocab):
    model.eval()
    with torch.no_grad():
        input_tensor = pytorch_preprocess_text(text, vocab)
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

def load_pytorch_model_and_vocab(filename):
    with open(filename, 'rb') as f:
        data = dill.load(f)
    model = data['model']
    vocab = data['vocab']
    model.eval()
    return model, vocab

def analyze_text_pytorch_model(text):
    file_path = Path.cwd() / "sentiment_analysis" / "models" / "PyTorchNBoWModel.pkl"
    print(f'Model file: {file_path.resolve()}')

    model, vocab = load_pytorch_model_and_vocab(file_path.resolve())
    return -1 if pytorch_predict(model, text, vocab) == 0 else 1


def analyze_text(text, model_id):
    clean_text = clean_data(text)
    print(f'Clean text: {clean_text}')

    res = 0
    if model_id == 1:
        res = analyze_text_nltk_model(clean_text)
    elif model_id == 2:
        res = analyze_text_sklern_model(clean_text)
    elif model_id == 3:
        res = analyze_text_pytorch_model(clean_text)

    return res