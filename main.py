import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from cleaning import text_cleaner
from stemming import stem_text



def text_tokenizer(text):
    text = text_cleaner(text)
    text = stem_text(text)
    return [word for word in text if len(word)>3]
        

true_df = pd.read_csv('true.csv')

feature_extraction = TfidfVectorizer(tokenizer=text_tokenizer)

X_transform = feature_extraction.fit_transform(true_df['title'])
arr = X_transform.toarray()

count_features = feature_extraction.get_feature_names_out(true_df['title'])

sum_words = X_transform.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in  feature_extraction.vocabulary_.items()]
words_freq =dict(sorted(words_freq, key = lambda x: x[1], reverse=True))

n_words = 15
top_n_words = list(words_freq.keys())[:n_words]
top_n_words_count = list(words_freq.values())[:n_words]

plt.subplots(figsize=(11, 5))
plt.xlabel("Term")
plt.ylabel("Count")
plt.title("Count of top 15 most important words")
plt.bar(top_n_words,top_n_words_count)
plt.show()

pretty_table = PrettyTable()
pretty_table.title = "Count of top 15 most important words"
pretty_table.add_column("Term", top_n_words[:10])
pretty_table.add_column("Weight", top_n_words_count[:10])
pretty_table