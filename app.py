import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

app = Flask(__name__)

# Load and prepare the model
data = pd.read_csv("data3.csv")
X = data.drop(["Career"], axis=1)
y = data["Career"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SVC(kernel="linear")
model.fit(X_train, y_train)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    user_responses = request.form.to_dict()
    print(user_responses)
    user_input = []
    for question in X.columns[::2]:  # Matching questions exactly as in X.columns
        response = user_responses.get(question, "").lower()
        if response == "yes":
            user_input.extend([1, 0])
        elif response == "no":
            user_input.extend([0, 1])
        else:
            return "Please enter 'yes' or 'no' for all questions."

    # Create a DataFrame with the correct columns
    question_vectorized = pd.DataFrame([user_input], columns=X.columns)

    # Predict using the trained model
    prediction = model.predict(question_vectorized)[0]
    return redirect(url_for("result", career_path=prediction))


@app.route("/result")
def result():
    career_path = request.args.get("career_path", None)
    return render_template("result.html", career_path=career_path)


if __name__ == "_main_":
    app.run(debug=True)
