import requests
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# HuggingFace API
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
headers = {
    "Authorization": "Bearer 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"
}


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    accuracy = 0.85

    if request.method == "POST":

        review = request.form["review"]

        data = vectorizer.transform([review])

        prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        accuracy=accuracy
    )


@app.route('/chat', methods=['POST'])
def chat():

    user_message = request.form['message']

    payload = {
    "inputs": "You are a movie assistant. " + user_message
}

    try:

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        result = response.json()

        if isinstance(result, list):

            if 'generated_text' in result[0]:

                bot_reply = result[0]['generated_text']

            else:

                bot_reply = str(result[0])

        elif isinstance(result, dict):

            if 'error' in result:

                bot_reply = result['error']

            else:

                bot_reply = str(result)

        else:

            bot_reply = "No response from AI."

    except Exception as e:

        bot_reply = "AI server busy. Try again."

    return {"reply": bot_reply}

if __name__ == "__main__":
    app.run(debug=True)
