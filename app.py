from flask import Flask, render_template, request
import pandas as pd
import os
import re
import nltk
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------------------
# APP SETUP
# ----------------------------------------

app = Flask(__name__)

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ----------------------------------------
# TEXT CLEANING
# ----------------------------------------

def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

# ----------------------------------------
# TRAIN MODEL ON STARTUP
# ----------------------------------------

def train_model():

    print("Training model...")

    df = pd.read_csv(
        "data/sentiment140.csv",
        encoding="latin-1",
        header=None,
        sep=",",
        usecols=[0, 5],
        names=["sentiment", "tweet"]
    )

    # Convert labels (0 = Negative, 4 = Positive)
    df["sentiment"] = df["sentiment"].map({0: 0, 4: 2})

    # Balanced dataset
    df = pd.concat([
        df[df["sentiment"] == 0].sample(20000, random_state=42),
        df[df["sentiment"] == 2].sample(20000, random_state=42)
    ])

    df["cleaned"] = df["tweet"].apply(clean_tweet)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["cleaned"])
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("Model training completed.\n")

    return model, vectorizer


model, vectorizer = train_model()

# ----------------------------------------
# MAIN ROUTE
# ----------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["file"]
        df = pd.read_csv(file)

        if "tweet" not in df.columns:
            return "CSV must contain a column named 'tweet'"

        df["cleaned"] = df["tweet"].apply(clean_tweet)

        X = vectorizer.transform(df["cleaned"])

        # Get prediction probabilities
        probs = model.predict_proba(X)

        sentiments = []
        confidences = []

        for prob in probs:
            neg_prob = prob[0]
            pos_prob = prob[1]

            confidence = max(neg_prob, pos_prob)

            if abs(neg_prob - pos_prob) < 0.15:
                sentiments.append("Neutral")
            elif neg_prob > pos_prob:
                sentiments.append("Negative")
            else:
                sentiments.append("Positive")

            confidences.append(round(confidence * 100, 2))

        df["sentiment"] = sentiments
        df["confidence"] = confidences

        # Save analyzed CSV
        os.makedirs("static/results", exist_ok=True)
        analyzed_path = "static/results/analyzed_output.csv"
        df.to_csv(analyzed_path, index=False)

        # Count results
        counts = df["sentiment"].value_counts()

        negative = counts.get("Negative", 0)
        neutral = counts.get("Neutral", 0)
        positive = counts.get("Positive", 0)

        total = negative + neutral + positive

        # Charts
        os.makedirs("static/charts", exist_ok=True)

        # Bar Chart
        plt.figure()
        plt.bar(
            ["Negative", "Neutral", "Positive"],
            [negative, neutral, positive]
        )
        plt.title("Sentiment Distribution")
        plt.ylabel("Count")
        bar_chart = "static/charts/bar_chart.png"
        plt.savefig(bar_chart)
        plt.close()

        # Pie Chart
        plt.figure()
        plt.pie(
            [negative, neutral, positive],
            labels=["Negative", "Neutral", "Positive"],
            autopct="%1.1f%%"
        )
        plt.title("Sentiment Distribution")
        pie_chart = "static/charts/pie_chart.png"
        plt.savefig(pie_chart)
        plt.close()

        # Preview table (top 5 rows)
        preview_data = df.head(5).to_dict(orient="records")

        return render_template(
            "index.html",
            bar_chart=bar_chart,
            pie_chart=pie_chart,
            total=total,
            negative=negative,
            neutral=neutral,
            positive=positive,
            preview=preview_data,
            download_path=analyzed_path
        )

    return render_template("index.html")


# ----------------------------------------
# RUN APP
# ----------------------------------------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)