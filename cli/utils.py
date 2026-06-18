import json
import string
import argparse
from pathlib import Path
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def get_word_stemmer(word):
    return (stemmer.stem(word))


def load_movies():
    json_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "movies.json"
    )
    with open(json_path, "r") as f:
        movies = json.load(f)
    return (movies)


def process_token(query, stopwords):
    text = rm_punctuation_upper(query)
    tokens = text.split()
    tokens = [get_word_stemmer(t) for t in tokens if t and t not in stopwords]
    return (tokens)

def rm_punctuation_upper(text):
    text = text.lower()
    table = str.maketrans("", "", string.punctuation)
    return (text.translate(table))

def load_stopwords():
    stopwords_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "stopwords.txt"
    )
    with open(stopwords_path, 'r') as f:
        stopword =  f.read()
    stopwords = stopword.splitlines()
    clean_stpword = [rm_punctuation_upper(w) for w in stopwords]
    return (clean_stpword)


stopwords = load_stopwords()