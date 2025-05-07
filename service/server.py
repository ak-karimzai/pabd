from flask import Flask, request, jsonify, render_template
from logging.config import dictConfig
import config as cfg
import joblib
from models.PredictPriceRequest import PredictPriceRequest
from models.PredictPriceResponse import PredictPriceResponse

dictConfig(cfg.log_config)
app = Flask(__name__)

def predict_price(area) -> float:
    model = joblib.load(cfg.LRM_TRAINED_PATH)
    return float(model.predict([[area]])[0])

@app.route("/api/predict", methods=["POST"])
def predict():
    request_body = request.get_json()
    app.logger.info(f"handling \"process_numbers\": request body: {request_body}")

    try:
        dto = PredictPriceRequest(**request_body)
        price = predict_price(
            area=dto.area)
        response = PredictPriceResponse(price=price)
    except (ValueError, KeyError):
        app.logger.warning("Received invalid request body: %s", request_body)
        return jsonify({'error': 'Invalid input'}), 422

    app.logger.info(f"client response \"process_numbers\": {response.model_dump()}")
    return jsonify(response.model_dump()), 200

@app.route("/")
def home():
    app.logger.debug("server started")
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)