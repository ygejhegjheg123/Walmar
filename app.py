from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Logging
logging.basicConfig(level=logging.DEBUG)

# Cargar modelo
model = joblib.load('wolmar.pkl')  # Asegúrate de usar el modelo correcto
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('Wolmar.html')

@app.route('/predict-ventas', methods=['POST'])
def predict_ventas():
    try:
        # Extraer variables
        datos = {
            'Store': int(request.form['Store']),
            'CPI': float(request.form['CPI']),
            'Unemployment': float(request.form['Unemployment']),
            'Week': int(request.form['Week']),
            'Temperature': float(request.form['Temperature']),
            'Fuel_Price': float(request.form['Fuel_Price'])
        }

        # Crear DataFrame
        df = pd.DataFrame([datos])
        app.logger.debug(f'Datos recibidos: {df}')

        # Predecir
        prediccion = model.predict(df)[0]
        app.logger.debug(f'Predicción: {prediccion}')

        return jsonify({'prediccion': float(prediccion)})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {e}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
