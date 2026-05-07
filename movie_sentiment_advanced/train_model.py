from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

texts = [

    "this movie is amazing",
    "excellent acting and story",
    "i loved this movie",
    "fantastic film",
    "wonderful experience",
    "great direction",
    "super hit movie",
    "awesome screenplay",
    "best movie ever",
    "very entertaining",
    "good movie",

    "worst movie ever",
    "terrible acting",
    "boring film",
    "bad screenplay",
    "waste of time",
    "very disappointing",
    "poor direction",
    "awful movie",
    "not good",
    "horrible experience"

]

labels = [

    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",

    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative"
]
vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True
)
X = vectorizer.fit_transform(texts)

X_train,X_test,y_train,y_test = train_test_split(X,labels,test_size=0.2)

model = MultinomialNB(alpha=0.5)
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
