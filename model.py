from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

def procesar_archivo(file):
    file.save('data.csv')    
    data = pd.read_csv('data.csv')
    if 'url' in data.columns:
        data.drop('url', axis=1, inplace=True)
    return data['news']

def hacer_predicciones(texto, model):
    vectorizador = CountVectorizer()
    X = vectorizador.fit_transform(texto)
    predictions = model.predict(X)
    return predictions.tolist()

@app.route('/predict', methods=['POST'])
def predicts():
    ann = load('ann.joblib')
    knn = load('knn.joblib')
    file = request.files['file']
    texto_noticias = procesar_archivo(file)
    predictions_ann = hacer_predicciones(texto_noticias, ann)
    predictions_knn = hacer_predicciones(texto_noticias, knn)
    
    # agregan al archivo las predicciones
    
    
    # # contar ocurrencias
    # predictions_ann = {i: predictions_ann.count(i) for i in set(predictions_ann)}
    # predictions_knn = {i: predictions_knn.count(i) for i in set(predictions_knn)}
    return jsonify({'prediction_ann': predictions_ann, 'prediction_knn': predictions_knn})

if __name__ == '__main__':
    app.run(debug=True)
