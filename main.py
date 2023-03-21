# -*- coding: utf-8 -*-
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import pandas as pd
import string

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def preprocess_single_mail(mail):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    tokens = word_tokenize(mail)

    clean = []
    for word in tokens:
        if word not in stopwords_english and word not in string.punctuation:
            clean.append(stemmer.stem(word))

    return " ".join(clean)

df = pd.read_csv('train.csv')
df = df[['Message', 'Label']]
df['Message'] = df['Message'].apply(lambda x: preprocess_single_mail(x))


text = " ".join(["".join(i) + " " for i in df['Message']])
plt.figure(figsize=(20, 20))

wc = WordCloud(max_words=2000, width=1600, height=800)
wc.generate(text)
plt.imshow(wc, interpolation='bilinear')
plt.title("Spam Word Cloud")

cv = CountVectorizer(max_features=1500, analyzer='word', lowercase=False)
X = cv.fit_transform(df['Message'])

encoder = LabelEncoder()
df['Label'] = encoder.fit_transform(df['Label'])
print(encoder.classes_)
y = df['Label']

classifier = LogisticRegression(random_state=0)
classifier.fit(X, y)
classifier.score(X, y)

test_df = pd.read_csv('test.csv')
test_df = test_df[['Message', 'Label']]
test_df['Label'] = encoder.transform(test_df['Label'])

test_df['Message'] = test_df['Message'].apply(lambda x: preprocess_single_mail(x))

X_test = cv.transform(test_df['Message'])
y_test = test_df['Label']
score = classifier.score(X_test, y_test)

mails = [
    "Congrats! You got a free gift. Just text GIFT to 56789",
    "Congrats one more time! Your wedding was awesome!",
    "Hey could you send me a link for homework?",
    "You were chosen for a prize - free homework! Just click the link and download pdf for free!"
]

mails_processed = [preprocess_single_mail(x) for x in mails]

mails_processed = cv.transform(mails_processed)
results = encoder.inverse_transform(classifier.predict(mails_processed))

for text, score in zip(mails, results):
  print(f'{score} : {text}')

model = Sequential()
model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X.toarray(), to_categorical(y), epochs=10, verbose=1, validation_split=0.2)

test_results = model.evaluate(X_test.toarray(), to_categorical(y_test), verbose=1)
print("Keras", test_results)

results2 = model.predict(mails_processed)
for text, score in zip(mails, results2):
  print(f'{encoder.inverse_transform([np.argmax(score)])} : {text}')