from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_airline.pkl')
model = joblib.load(MODEL_PATH)

airlines = {
    0: 'AirAsia',
    1: 'GO_FIRST',
    2: 'Indigo',
    3: 'SpiceJet',
    4: 'Vistara',
    5: 'Air India'
}

cities = {
    0: 'Delhi',
    1: 'Mumbai',
    2: 'Bangalore',
    3: 'Kolkata',
    4: 'Hyderabad',
    5: 'Chennai'
}

classes = {
    0: 'Economy',
    1: 'Business'
}

stops_map = {
    0: 'Non Stop',
    1: '1 Stop',
    2: '2+ Stops'
}

SUMMARY = {
    'model_name': 'Random Forest Regressor',
    'dataset_name': 'Airline Ticket Price',
    'data_count': '±9000',
    'r2_score': '96.34%'
}

@app.route('/')
def dashboard():
    return render_template('dashboard.html', summary=SUMMARY)

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    prediksi_text = None
    detail = None

    if request.method == 'POST':
        try:
            airline = int(request.form['airline'])
            source = int(request.form['source'])
            destination = int(request.form['destination'])
            ticket_class = int(request.form['class'])
            duration = float(request.form['duration'])
            days_left = int(request.form['days_left'])
            stops = int(request.form['stops'])

            if source == destination:
                prediksi_text = '❌ Kota asal dan tujuan tidak boleh sama.'
            elif duration <= 0:
                prediksi_text = '❌ Durasi perjalanan harus lebih dari 0 jam.'
            elif days_left < 0:
                prediksi_text = '❌ Hari menuju keberangkatan tidak boleh negatif.'
            elif stops not in [0, 1, 2]:
                prediksi_text = '❌ Jumlah transit tidak valid.'
            else:
                input_features = np.array([[
                    airline,
                    source,
                    destination,
                    ticket_class,
                    duration,
                    days_left,
                    stops
                ]])

                prediction = model.predict(input_features)
                output = round(float(prediction[0]), 2)

                detail = {
                    'airline': airlines[airline],
                    'source': cities[source],
                    'destination': cities[destination],
                    'class': classes[ticket_class],
                    'duration': duration,
                    'days_left': days_left,
                    'stops': stops_map[stops]
                }

                prediksi_text = f'₹ {output:,.2f}'
        except Exception as e:
            prediksi_text = f'Error: {str(e)}'

    return render_template('index.html', prediksi_text=prediksi_text, detail=detail, summary=SUMMARY)

if __name__ == '__main__':
    app.run(debug=True)