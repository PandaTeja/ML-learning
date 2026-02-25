import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report

df = pd.read_csv("resources/spam.csv",encoding="latin1",usecols=[0,1])

df.columns = ["label", "message"]


df['label'] = df['label'].map({'ham': 0, 'spam': 1})


X=df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec,y_train)

predictions = model.predict(X_test_vec)


print("Accuracy:", accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))