from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

texts = [
    "this movie is amazing","I love this film","fantastic experience",
    "worst movie ever","very boring and dull","terrible storyline",
    "great acting and story","bad direction","awesome movie","not good"
]*50

labels = ["positive","positive","positive",
          "negative","negative","negative",
          "positive","negative","positive","negative"]*50

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

X_train,X_test,y_train,y_test = train_test_split(X,labels,test_size=0.2)

model = MultinomialNB()
model.fit(X_train,y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test,pred)

pickle.dump(model,open("model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

plt.figure()
plt.bar(["Accuracy"],[acc])
plt.title("Model Accuracy")
plt.savefig("static/accuracy.png")

print("Accuracy:",acc)
