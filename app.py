from flask import Flask, request, jsonify
from predict_outbreak import predict_outbreak

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [
        data["bacteria_count"],
        data["virus_rna_level"],
        data["antibiotic_resistance"]
    ]
    prediction = predict_outbreak(features)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
