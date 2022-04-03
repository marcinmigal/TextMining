from sklearn.feature_extraction.text import TfidfVectorizer
from cleaning import text_cleaner
from stemming import stem_text


def text_tokenizer(text):
    text = text_cleaner(text)
    text = stem_text(text)
    return [word for word in text if len(word)>3]
        


with open('true.csv', 'r', encoding='utf-8') as file:
    text = []
    for  i, row  in enumerate(file.readlines()):
        text.append(row)
        if i == 3:
            break

vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)


print(X_transform.toarray())