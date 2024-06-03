from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Charger le modèle
model = joblib.load('model.pkl')

# Définir une route pour effectuer des prédictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    return jsonify({'prediction': prediction.tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True)
