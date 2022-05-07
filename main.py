from regex import F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from cleaning import text_cleaner
from stemming import stem_text
import numpy as np


def text_tokenizer(text):
    text = text_cleaner(text)
    text = stem_text(text)
    return [word for word in text if len(word)>3]
        


with open('true.csv', 'r', encoding='utf-8') as file:
    text = []
    for  i, row  in enumerate(file.readlines()):
        # text.append(row)
        if i == 1:
            text.append(row)
            break



# Top 10 najczesciej wystepujacych tokenow

count_vectorizer = CountVectorizer(tokenizer=text_tokenizer)

X_transform = count_vectorizer.fit_transform(text)
arr = X_transform.toarray()

count_features = count_vectorizer.get_feature_names_out(text)

print("Najczesciej wystepujace tokeny: ")
for i in arr[0].argsort()[:10]:
    print(count_features[i])





# Top 10 najwazniejszych tokenow

tfid_vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)

X_transform = tfid_vectorizer.fit_transform(text)
arr = X_transform.toarray()

tfid_features = tfid_vectorizer.get_feature_names_out(text)

print("\nNajwazniejsze tokeny: ")
for i in arr[0].argsort()[:10]:
    print(tfid_features[i])

