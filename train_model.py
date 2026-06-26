import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Algorithms
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

# Shuffle dataset
data = data.sample(frac=1, random_state=42)

# Input and Output
x = data["title"] + " " + data["text"]
y = data["label"]

# Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

x_vectorized = vectorizer.fit_transform(x)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x_vectorized,
    y,
    test_size=0.25,
    random_state=42
)

# -------------------------------
# 1. Naive Bayes
# -------------------------------
nb_model = MultinomialNB()

nb_model.fit(x_train, y_train)

nb_pred = nb_model.predict(x_test)

nb_score = accuracy_score(y_test, nb_pred)

print("Naive Bayes Accuracy:", nb_score * 100)

# Save model
pickle.dump(nb_model, open("model/naive_bayes.pkl", "wb"))

# -------------------------------
# 2. SVM
# -------------------------------
svm_model = LinearSVC()

svm_model.fit(x_train, y_train)

svm_pred = svm_model.predict(x_test)

svm_score = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", svm_score * 100)

# Save model
pickle.dump(svm_model, open("model/svm_model.pkl", "wb"))

# -------------------------------
# 3. Logistic Regression
# -------------------------------
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(x_train, y_train)

lr_pred = lr_model.predict(x_test)

lr_score = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_score * 100)

# Save model
pickle.dump(lr_model, open("model/logistic_model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("All models saved successfully")