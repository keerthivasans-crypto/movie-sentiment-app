from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/",methods=["GET","POST"])
def home():
    prediction=""
    accuracy = 0.85
    if request.method=="POST":
        review=request.form["review"]
        data=vectorizer.transform([review])
        prediction=model.predict(data)[0]
    return render_template("index.html",prediction=prediction, accuracy=accuracy)

if __name__=="__main__":
    app.run(debug=True)

