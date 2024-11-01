from flask import Flask, request, render_template
import pickle
import pandas as pd


with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encoders.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)

app = Flask(__name__)


weather_data = pd.read_csv('Weather Test Data.csv')


input_columns = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 
                 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 
                 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']


unique_values = {col: sorted(weather_data[col].dropna().unique().tolist()) for col in input_columns}

@app.route('/')
def home():
    return render_template('index.html', unique_values=unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    
    input_data = {}
    for column in input_columns:
        input_data[column] = request.form.get(column)
    
    
    data = pd.DataFrame([input_data], columns=input_columns)

    
    for col in encoders:
        if col in data:
            data[col] = encoders[col].transform(data[col])
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    
    prediction = model.predict(data)[0]
    
    
    result = " " if prediction == 1 else "Not"
    return render_template('index.html', prediction=result, unique_values=unique_values)

if __name__ == '__main__':
    app.run(debug=True)
