from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [
        float(request.form['Time_spent_Alone']),
        int(request.form['Stage_fear']),
        float(request.form['Social_event_attendance']),
        float(request.form['Going_outside']),
        int(request.form['Drained_after_socializing']),
        float(request.form['Friends_circle_size']),
        float(request.form['Post_frequency'])
    ]
    
    # Convert to numpy array for prediction
    features_array = np.array([features])
    
    # Make prediction
    prediction = model.predict(features_array)[0]
    
    # Interpret prediction
    result = 'Introvert' if prediction == 1 else 'Extrovert'
    
    return render_template('index.html', prediction_text=f'Predicted Personality: {result}')

if __name__ == '__main__':
    app.run(debug=True)