from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('./Notebooks/covid_model.pkl')
scaler = joblib.load('./Notebooks/scaler.pkl')

# Load the dataset to get feature names, excluding the ID and target column
df = pd.read_csv('./data/qt_dataset.csv', encoding='latin-1')
feature_names = df.columns.drop(['ID', 'Result']).tolist()

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data = {key: float(value) for key, value in data.items()}
    input_data = np.array([list(data.values())])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)