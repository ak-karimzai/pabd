import os
import glob
import joblib
import argparse
import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(file_dir, "logs", "ml_logs.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(train_data_path):
    logger.info(f"Loading CSV files from: {train_data_path}")
    csv_files = glob.glob(os.path.join(train_data_path, "*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files.")

    dfs = []
    for i, file in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            logger.debug(f"Loaded file {i}: {file}")
        except Exception as e:
            logger.warning(f"Failed to load file {file}: {str(e)}")

    main_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total data loaded: {main_df.shape[0]} rows")
    return main_df


def process_cian(data):
    logger.info("Processing CIAN dataset...")

    required_cols = ['url', 'total_meters', 'price', 'floor', 'floors_count', 'rooms_count']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise KeyError(f"Missing required columns: {missing_cols}")

    try:
        data['url_id'] = data['url'].apply(lambda x: x.split('/')[-2])
    except Exception as e:
        logger.error(f"Error extracting 'url_id': {str(e)}")
        raise

    filtered_df = data[required_cols + ['url_id']].set_index('url_id')

    for col in required_cols:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

    logger.info(f"Data after filtering: {len(filtered_df)} rows")
    return filtered_df


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

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    except ValueError as e:
        logger.critical("Train/Test split failed due to insufficient samples.")
        raise

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['total_meters', 'floor_ratio']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['rooms_count'])
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    logger.info("Data preprocessed.")
    return X_train, X_test, y_train, y_test, preprocessor


def train_model(X_train, y_train, model_type="linear"):
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

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model performance...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    logger.info(f"Evaluation Results:")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"R² Score: {r2:.4f}")

    logger.info(f"\nMean Absolute Error (MAE): {mae:,.0f} руб")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:,.0f} руб")
    logger.info(f"R² Score: {r2:.4f}\n")

    return {"mae": mae, "rmse": rmse, "r2": r2}

def predict_price(model, preprocessor, total_meters, floor, floors_count, rooms_count):
    input_df = pd.DataFrame([{
        'total_meters': total_meters,
        'floor': floor,
        'floors_count': floors_count,
        'rooms_count': rooms_count,
        'floor_ratio': floor / floors_count,
    }])
    input_processed = preprocessor.transform(input_df)
    predicted_price = model.predict(input_processed)[0]
    logger.info(f"🔮 Predicted price: {predicted_price:,.0f} руб")
    return predicted_price

def save_model(model, preprocessor, model_path, preprocessor_path=None):
    if preprocessor_path is None:
        preprocessor_path = model_path.replace(".pkl", "_preprocessor.pkl")

    logger.info(f"Saving model to: {model_path}")
    logger.info(f"Saving preprocessor to: {preprocessor_path}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info("Model and preprocessor saved successfully.")

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate apartment price prediction model.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of dataset to use for testing')
    parser.add_argument('--train_data_path', type=str,
                        default=os.path.join(file_dir, "..", "data", "raw"),
                        help='Path to training data (CSV files)')
    parser.add_argument('--trained_model_path', type=str,
                        default=os.path.join(file_dir, "..", "models", "apartment_price_model.pkl"),
                        help='Path to save the trained model.')

    args = parser.parse_args()

    try:
        logger.info("Pipeline started.")
        raw_data = load_data(args.train_data_path)
        cleaned_data = process_cian(raw_data)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(cleaned_data, args.test_size)

        model = train_model(X_train, y_train, model_type="random_forest")
        metrics = evaluate_model(model, X_test, y_test)

        save_model(model, preprocessor, args.trained_model_path)

        predict_price(model, preprocessor,
                      total_meters=60,
                      floor=10,
                      floors_count=25,
                      rooms_count=2)

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Critical error during pipeline execution: {e}", exc_info=True)
        logger.error(f"🚨 Critical Error: {e}")

if __name__ == "__main__":
    main()