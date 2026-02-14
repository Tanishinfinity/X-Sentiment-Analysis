🚀 X-Sentiment-Analysis
AI Powered Sentiment Analysis Dashboard with Login, CSV Upload, PDF Export & Confidence Score
<p align="center"> <img src="assets/dashboard.png" width="900"> </p>
📌 Overview

X-Sentiment-Analysis is a full-stack AI web application that allows users to:

🔐 Secure Login

📂 Upload CSV files containing tweets

🤖 Perform AI-based Sentiment Analysis

📊 Visualize results using charts

📄 Export analyzed data as CSV

📝 Generate downloadable PDF reports

🎯 View prediction confidence scores

Built using Flask + Machine Learning + Modern Premium UI

✨ Features

✅ Logistic Regression Sentiment Model

✅ TF-IDF Vectorization

✅ Confidence Score Calculation

✅ Neutral Sentiment Detection

✅ Animated Dashboard Counters

✅ Bar & Pie Chart Visualization

✅ CSV Download

✅ PDF Report Export

✅ Login Authentication System

✅ Premium Dark UI with Glassmorphism

🧠 Machine Learning Details

Dataset: Sentiment140

Algorithm: Logistic Regression

Vectorizer: TF-IDF (5000 features)

Train/Test Split: 70/30 (Stratified)

Average Accuracy: ~75%

Confidence score is calculated using:

max(prediction_probability) × 100

Neutral sentiment is assigned when probability difference < 0.15.

🏗️ Project Structure
X-Sentiment-Analysis/
│
├── app.py
├── data/
│   └── sentiment140.csv
├── templates/
│   ├── index.html
│   └── login.html
├── static/
│   ├── charts/
│   ├── results/
│   └── reports/
├── assets/
│   ├── login.png
│   ├── dashboard.png
│   ├── charts.png
│   └── report.png
└── README.md
📊 Project Screenshots
🔐 Login Page
<p align="center"> <img src="assets/login.png" width="800"> </p>
📊 Dashboard Overview
<p align="center"> <img src="assets/dashboard.png" width="800"> </p>
📈 Sentiment Charts
<p align="center"> <img src="assets/charts.png" width="800"> </p>
📄 Export Reports
<p align="center"> <img src="assets/report.png" width="800"> </p>
🛠️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/Tanishinfinity/X-Sentiment-Analysis.git
cd X-Sentiment-Analysis
2️⃣ Create Virtual Environment
python -m venv sentiment_env

Activate environment:

Windows

sentiment_env\Scripts\activate

Mac/Linux

source sentiment_env/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt

If requirements.txt not created:

pip install flask pandas scikit-learn nltk matplotlib fpdf
4️⃣ Run Application
python app.py

Open in browser:

http://127.0.0.1:5000
🔐 Default Login Credentials
Username: admin
Password: admin123

(You can modify in app.py)

📂 CSV Format Example

Your uploaded CSV must contain a column named:

tweet

Example:

tweet
I love this product!
This is terrible service
I am not sure about this
📤 Export Options

After analysis:

⬇ Download analyzed CSV

📄 Download PDF Report

📊 View Charts

🧩 Technologies Used

Python

Flask

Pandas

Scikit-Learn

NLTK

Matplotlib

Bootstrap 5

HTML / CSS / JS

🔮 Future Improvements

🔑 Database-based authentication

🌍 Deployment on cloud

📈 Advanced ML models (LSTM / BERT)

📊 Interactive charts (Plotly)

👥 Multi-user accounts

👨‍💻 Author

Tanish Infinity

GitHub: https://github.com/Tanishinfinity

Passionate about AI, ML & Full Stack Development

⭐ Support

If you like this project:

⭐ Star the repository

🍴 Fork it

🛠️ Contribute

🚀 X-Sentiment-Analysis

Transforming raw tweets into actionable insights using AI.