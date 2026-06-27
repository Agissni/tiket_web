from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('model_airline.pkl')

print("Jumlah fitur model:", model.n_features_in_)

# Mapping agar tampil nama, bukan angka
airlines = {
    0: "AirAsia",
    1: "GO_FIRST",
    2: "Indigo",
    3: "SpiceJet",
    4: "Vistara",
    5: "Air India"
}

cities = {
    0: "Delhi",
    1: "Mumbai",
    2: "Bangalore",
    3: "Kolkata",
    4: "Hyderabad",
    5: "Chennai"
}

classes = {
    0: "Economy",
    1: "Business"
}

stops_map = {
    0: "Non Stop",
    1: "1 Stop",
    2: "2+ Stops"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        airline = int(request.form['airline'])
        source = int(request.form['source'])
        destination = int(request.form['destination'])
        ticket_class = int(request.form['class'])
        duration = float(request.form['duration'])
        days_left = int(request.form['days_left'])
        stops = int(request.form['stops'])

        # VALIDASI

        if source == destination:
            return render_template(
                'index.html',
                prediksi_text='❌ Kota asal dan tujuan tidak boleh sama.'
            )

        if duration <= 0:
            return render_template(
                'index.html',
                prediksi_text='❌ Durasi perjalanan harus lebih dari 0 jam.'
            )

        if days_left < 0:
            return render_template(
        'index.html',
        prediksi_text='❌ Hari menuju keberangkatan tidak boleh negatif.'
    )

        if stops not in [0, 1, 2]:
            return render_template(
                'index.html',
                prediksi_text='❌ Jumlah transit tidak valid.'
            )

        # Urutan fitur sesuai training
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
            "airline": airlines[airline],
            "source": cities[source],
            "destination": cities[destination],
            "class": classes[ticket_class],
            "duration": duration,
            "days_left": days_left,
            "stops": stops_map[stops]
        }

        return render_template(
            'index.html',
            prediksi_text=f"₹ {output:,.2f}",
            detail=detail
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediksi_text=f'Error: {str(e)}'
        )

if __name__ == '__main__':
    app.run(debug=True)