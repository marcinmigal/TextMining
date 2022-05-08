import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from prettytable import PrettyTable
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from cleaning import text_cleaner
from stemming import stem_text

plt.style.use('seaborn-dark')


def text_tokenizer(text):
    text = text_cleaner(text)
    text = stem_text(text)
    return [word for word in text if len(word)>3]


# Wczytanie i przygotowanie danych
mail_data = pd.read_csv('mail_data.csv')

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
feature_extraction = TfidfVectorizer(tokenizer=text_tokenizer)


count_vectorizer = CountVectorizer(min_df = 1, stop_words='english', lowercase='True')
X_transform = count_vectorizer.fit_transform(mail_data[mail_data['Category']==0]['Message'])
arr = X_transform.toarray()

count_features = count_vectorizer.get_feature_names_out(mail_data[mail_data['Category']==0]['Message'])

sum_words = X_transform.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in  count_vectorizer.vocabulary_.items()]
words_freq =dict(sorted(words_freq, key = lambda x: x[1], reverse=True))

n_words = 15
top_n_words = list(words_freq.keys())[:n_words]
top_n_words_count = list(words_freq.values())[:n_words]


plt.subplots(figsize=(11, 5))
plt.xlabel("Term")
plt.ylabel("Count")
plt.title("Count of most common words")
plt.bar(top_n_words,top_n_words_count)
plt.show()


pretty_table = PrettyTable()
pretty_table.title = "Count of most common words"
pretty_table.add_column("Term", top_n_words[:10])
pretty_table.add_column("Count", top_n_words_count[:10])
pretty_table


X_df = mail_data['Message']
Y_df = mail_data['Category']


X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=3)


X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)


# Tworzenie i uczenie modelu
model = LogisticRegression()

model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Training data accuracy: ', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Test data accuracy: ', accuracy_on_test_data)

#False Posives Testing
logit_roc_auc = roc_auc_score(Y_test, model.predict(X_test_features))
fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test_features)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# Testowanie modelu
input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)
print(prediction)

if (prediction[0]==1):
  print('Not Spam')
else:
  print('Spam')