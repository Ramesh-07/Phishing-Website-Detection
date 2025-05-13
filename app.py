from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("dataset.csv")

# Separate features and target variable
X = df.drop(columns=['Phising'])
y = df['Phising'].astype(int)  # Convert target to integer if needed

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


def extract_features(url):
    """Extracts relevant features from the given URL."""
    def has_ip_address(url):
        ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
        return 1 if re.search(ip_pattern, url) else 0

    def url_length(url):
        return len(url)

    def at_symbol(url):
        return 1 if '@' in url else 0

    def path_level(url):
        return urlparse(url).path.count('/')

    def num_dots(url):
        return url.count('.')

    def https_in_hostname(url):
        hostname = urlparse(url).hostname
        return 1 if hostname and 'https' in hostname else 0

    def num_dash(url):
        return url.count('-')

    return np.array([
        has_ip_address(url),
        url_length(url),
        at_symbol(url),
        path_level(url),
        num_dots(url),
        https_in_hostname(url),
        num_dash(url)
    ]).reshape(1, -1)


# Initialize Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("url", "")

    if not url:
        return render_template("index.html", prediction_text="⚠️ Please enter a valid URL!")

    features = extract_features(url)
    prediction = clf.predict(features)
    result = "🔴 Phishing" if prediction[0] == 1 else  "🟢 Legitimate"

    return render_template("index.html", prediction_text=f"Prediction: {result}")



if __name__ == "__main__":
    app.run(debug=True, port=5000)
