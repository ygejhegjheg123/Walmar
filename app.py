from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)


# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('model.pkl')
# model = joblib.load('modelo_svc.joblib')
app.logger.debug('Modelo cargado correctamente.')


@app.route('/')
def insectos():
    return render_template('insectos.html')

@app.route('/iris')
def home():
    return render_template('iris.html')



@app.route('/predict-insecto', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        abdomen = float(request.form['abdomen'])
        antena = float(request.form['antena'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[abdomen, antena]], columns=['abdomen', 'antena'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400


# esto siempre va al final
if __name__ == '__main__':
    app.run(debug=True)

