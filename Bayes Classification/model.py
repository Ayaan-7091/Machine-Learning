import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score

df = pd.read_csv('re',sep="\t", header=None, names=['Label','Message'])

# Encode the labels
df['Label'] = df['Label'].map({'ham':0,'spam':1})
X = df['Message']
Y= df["Label"]

# Split the data 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# model

model = MultinomialNB()
model.fit(X_train_vec,Y_train)

Y_pred = model.predict(X_test_vec)

result_df = pd.DataFrame({'Predicted':Y_pred,'Actual':Y_test})
print(result_df)
result_df.to_csv('result')

# Evaluation
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# print(df.head())

def classify_mails(model,mail):
    mail_vec = vectorizer.transform(mail)
    prediction = model.predict(mail_vec)
    return prediction

print(classify_mails(model,["Urgent, you won a lottery call at 9658 4578 to claim it"]))