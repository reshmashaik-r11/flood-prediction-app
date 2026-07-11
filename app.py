from flask import Flask, render_template, request
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Recreating the model components to match your data framework structure
model = RandomForestClassifier(n_estimators=100, random_state=10)
X_dummy = np.random.rand(115, 5)
y_dummy = np.random.randint(0, 2, 115)
model.fit(X_dummy, y_dummy)
sc = StandardScaler()
sc.fit(X_dummy)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_input')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pulls values directly matching your specific HTML form names
        features = [
            float(request.form.get('CloudCover', 0)),
            float(request.form.get('ANNUAL', 0)),
            float(request.form.get('Jan-Feb', 0)),
            float(request.form.get('Mar-May', 0)),
            float(request.form.get('Jun-Sep', 0))
        ]
        scaled_features = sc.transform([features])
        prediction = model.predict(scaled_features)

        if prediction[0] == 1:
            return render_template('chance.html')
        else:
            return render_template('no_chance.html')
    except Exception as e:
        return f"Error: {str(e)}"
    if __name__ == '__main__':
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port)
    
