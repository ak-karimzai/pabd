import os
import logging
import joblib
import pandas as pd
import numpy as np
import argparse

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(file_dir, "logs", "model_evaluation.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_artifacts(model_path, preprocessor_path):
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Loading preprocessor from: {preprocessor_path}")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def load_test_data(input_path):
    logger.info(f"Loading test data from: {input_path}")
    data = pd.read_csv(input_path, index_col='url_id')
    return data


def evaluate_model(model, preprocessor, data):
    logger.info("Evaluating model...")

    data['floor_ratio'] = data['floor'] / data['floors_count']
    data['rooms_count'] = data['rooms_count'].astype(int).clip(upper=3)

    X = data[['total_meters', 'floor_ratio', 'rooms_count']]
    y = data['price']

    X_processed = preprocessor.transform(X)
    predictions = model.predict(X_processed)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    logger.info(f"Evaluation Results:")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"R² Score: {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def predict_price(model, preprocessor, features):
    logger.info("Making prediction...")
    input_df = pd.DataFrame([features])
    input_df['floor_ratio'] = input_df['floor'] / input_df['floors_count']
    input_processed = preprocessor.transform(input_df)
    predicted_price = model.predict(input_processed)[0]
    logger.info(f"Predicted price: {predicted_price:,.0f} руб")
    return predicted_price


def main():
    parser = argparse.ArgumentParser(description='Evaluate apartment price prediction model.')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(file_dir, "..", "models", "apartment_price_model.pkl"),
                        help='Path to trained model')
    parser.add_argument('--preprocessor_path', type=str,
                        default=os.path.join(file_dir, "..", "models", "preprocessor.pkl"),
                        help='Path to preprocessor')
    parser.add_argument('--test_data_path', type=str,
                        default=os.path.join(file_dir, "..", "data", "processed", "processed_data.csv"),
                        help='Path to test data')

    args = parser.parse_args()

    try:
        logger.info("Model evaluation started.")
        model, preprocessor = load_model_artifacts(args.model_path, args.preprocessor_path)
        test_data = load_test_data(args.test_data_path)

        metrics = evaluate_model(model, preprocessor, test_data)

        sample_features = {
            'total_meters': 60,
            'floor': 10,
            'floors_count': 25,
            'rooms_count': 2
        }
        predicted_price = predict_price(model, preprocessor, sample_features)

        logger.info("Model evaluation completed successfully.")
    except Exception as e:
        logger.critical(f"Critical error during model evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()