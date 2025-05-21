import os
import logging
import pandas as pd
import joblib
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

file_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(file_dir, "logs", "model_training.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_processed_data(input_path):
    logger.info(f"Loading processed data from: {input_path}")
    data = pd.read_csv(input_path, index_col='url_id')
    logger.info(f"Loaded data shape: {data.shape}")
    return data


def preprocess_data(data, test_size=0.2):
    logger.info("Preprocessing data...")

    if data.empty:
        logger.error("No data available for preprocessing.")
        raise ValueError("Input DataFrame is empty.")

    data['floor_ratio'] = data['floor'] / data['floors_count']
    data['rooms_count'] = data['rooms_count'].astype(int).clip(upper=3)

    X = data[['total_meters', 'floor_ratio', 'rooms_count']]
    y = data['price']

    if len(data) < 10:
        logger.warning(f"Only {len(data)} samples left before train/test split. Consider adjusting filters.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['total_meters', 'floor_ratio']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['rooms_count'])
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    logger.info("Data preprocessed.")
    return X_train, X_test, y_train, y_test, preprocessor


def train_model(X_train, y_train, model_type="random_forest"):
    logger.info(f"Training {model_type} model...")
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "linear":
        model = LinearRegression()
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    logger.info("Model trained successfully.")
    return model


def save_model_artifacts(model, preprocessor, model_path, preprocessor_path):
    logger.info(f"Saving model artifacts to: {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Model artifacts saved successfully.")


def main():
    parser = argparse.ArgumentParser(description='Train apartment price prediction model.')
    parser.add_argument('--input_path', type=str,
                        default=os.path.join(file_dir, "..", "data", "processed", "processed_data.csv"),
                        help='Path to processed data')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of dataset to use for testing')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(file_dir, "..", "models", "apartment_price_model.pkl"),
                        help='Path to save trained model')
    parser.add_argument('--preprocessor_path', type=str,
                        default=os.path.join(file_dir, "..", "models", "preprocessor.pkl"),
                        help='Path to save preprocessor')

    args = parser.parse_args()

    try:
        logger.info("Model training started.")
        data = load_processed_data(args.input_path)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data, args.test_size)
        model = train_model(X_train, y_train)
        save_model_artifacts(model, preprocessor, args.model_path, args.preprocessor_path)
        logger.info("Model training completed successfully.")
    except Exception as e:
        logger.critical(f"Critical error during model training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()