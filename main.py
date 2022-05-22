import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
from cleaning import text_cleaner
from stemming import stem_text


def text_tokenizer(text):
    text = text_cleaner(text)
    text = stem_text(text)
    return [word for word in text if len(word)>3]


reviews_df = pd.read_csv('alexa_reviews.csv', delimiter=";")
reviews_df = reviews_df[['rating','verified_reviews']]


text_list = []

for row in reviews_df.verified_reviews:
    line = text_cleaner(row) # Czyszczenie regex + lematyzacja
    line = stem_text(line) # Stemming
    text_list.extend(line)


bow = dict([[x,text_list.count(x)] for x in set(text_list)])

wc = WordCloud()
wc.generate_from_frequencies(bow)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()



#Czesc zwiazana z klasyfikacja:

feature_extraction = TfidfVectorizer(tokenizer=text_tokenizer)

X_df = reviews_df['verified_reviews']
Y_df = reviews_df['rating']

#Podzial na train i test set
X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=3)


X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)


# Tworzenie i uczenie modelu
model = DecisionTreeClassifier(random_state=0)

model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Training data accuracy: ', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Test data accuracy: ', accuracy_on_test_data)

print(classification_report(Y_test, prediction_on_test_data))