from flask import Flask, request, jsonify, render_template
from logging.config import dictConfig
import config as cfg
import pandas as pd
import joblib
from models.PredictPriceRequest import PredictPriceRequest
from models.PredictPriceResponse import PredictPriceResponse


dictConfig(cfg.log_config)
app = Flask(__name__)

def predict_price(request: PredictPriceRequest) -> float:
    model = joblib.load(cfg.LRM_TRAINED_PATH)
    preprocessor = joblib.load(cfg.PREPROCESSOR_PATH)

    input_df = pd.DataFrame([{
        'total_meters': request.area,
        'floor': request.floor,
        'floors_count': request.floors_count,
        'rooms_count': request.rooms_count,
        'floor_ratio': request.floor / request.floors_count,
    }])

    input_processed = preprocessor.transform(input_df)

    predicted_price = model.predict(input_processed)[0]

    return predicted_price

@app.route("/api/predict", methods=["POST"])
def predict():
    request_body = request.get_json()
    app.logger.info(f"handling \"process_numbers\": request body: {request_body}")

    try:
        dto = PredictPriceRequest(**request_body)
        print(dto)
        price = predict_price(dto)
        print(price)
        response = PredictPriceResponse(price=price)
    except (ValueError, KeyError) as e:
        app.logger.warning("Received invalid request body: %s", e)
        return jsonify({'error': 'Invalid input'}), 422

    app.logger.info(f"client response \"process_numbers\": {response.model_dump()}")
    return jsonify(response.model_dump()), 200

@app.route("/")
def home():
    app.logger.debug("server started")
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)