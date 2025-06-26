from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Logging
logging.basicConfig(level=logging.DEBUG)

# ✅ Cargar modelo usando joblib
with open('wolmar.pkl', 'rb') as f:
    model = joblib.load(f)
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('Wolmar.html')

@app.route('/predict-ventas', methods=['POST'])
def predict_ventas():
    try:
        # Extraer variables del formulario
        datos = {
            'Store': int(request.form['Store']),
            'CPI': float(request.form['CPI']),
            'Unemployment': float(request.form['Unemployment']),
            'Week': int(request.form['Week']),
            'Temperature': float(request.form['Temperature']),
            'Fuel_Price': float(request.form['Fuel_Price'])
        }

        # Crear DataFrame con una sola fila
        df = pd.DataFrame([datos])
        app.logger.debug(f'Datos recibidos: {df}')

        # Realizar predicción
        prediccion = model.predict(df)[0]
        app.logger.debug(f'Predicción: {prediccion}')

        return jsonify({'prediccion': float(prediccion)})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {e}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
