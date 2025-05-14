import glob
import joblib
import argparse
import os
import logging
import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(file_dir, "logs", "ml_logs.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def fetch_data(room_numb):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path_dir = os.path.join(file_dir, "..", "data", "raw")
    if not os.path.exists(csv_path_dir):
        os.makedirs(csv_path_dir, exist_ok=True)

    n_rooms = 1
    while n_rooms <= 4:
        t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        csv_path = os.path.join(csv_path_dir, f'{n_rooms}_{t}.csv')

        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": 10,
                "object_type": "secondary"
            })
        df = pd.DataFrame(data)

        df.to_csv(csv_path,
                  encoding='utf-8',
                  index=False)
        n_rooms += 1


def load_data(train_data_path):
    file_pattern = os.path.join(train_data_path, "*.csv")
    logger.debug(f"Looking for CSV files matching pattern: {file_pattern}")

    file_list = glob.glob(file_pattern)
    logger.info(f"Found {len(file_list)} CSV files.")

    dfs = []
    for i, file in enumerate(file_list, start=1):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            logger.debug(f"Loaded file {i}: {file}")
        except Exception as e:
            logger.warning(f"Failed to load file {file}: {str(e)}")

    main_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded data with shape: {main_df.shape}")
    return main_df


def process_cian(data):
    logger.info("Starting data processing...")

    if 'url' not in data.columns:
        logger.error("'url' column is missing from the dataset.")
        raise KeyError("'url' column not found in DataFrame")

    try:
        data['url_id'] = data['url'].apply(lambda x: x.split('/')[-2])
        logger.debug("Extracted 'url_id' from 'url'")
    except Exception as e:
        logger.error(f"Error extracting 'url_id': {str(e)}")
        raise

    new_df = data[['url_id', 'total_meters', 'price']].set_index('url_id')
    filtered_df = new_df[(new_df['price'] < 100_000_000) & (new_df['total_meters'] < 100)]
    logger.info(f"Data processed and filtered. Remaining rows: {len(filtered_df)}")
    return filtered_df


def preprocess_data(data, test_size=0.2):
    logger.info("Starting data preprocessing...")
    X = data[['total_meters']]
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info("Data split and scaled.")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    logger.info("Training linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Model training completed.")
    return model


def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model performance...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    logger.info(f"Evaluation results:")
    logger.info(f"Mean Absolute Error (MAE): {mae:.2f}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    logger.info(f"R² Score: {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def save_model(model, model_path):
    logger.info(f"Saving trained model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved successfully.")


def main():
    parser = argparse.ArgumentParser(description='Automate machine learning pipeline.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of dataset to use for testing (default: 0.2)')
    parser.add_argument('--train_data_path', type=str,
                        default=os.path.join(file_dir, "..", "data", "raw"),
                        help='Path to training data (CSV files)')
    parser.add_argument('--trained_model_path', type=str,
                        default=os.path.join(file_dir, "..", "models", "linear_reg_model.pkl"),
                        help='Path to save the trained model (default: ../models/linear_reg_model.pkl)')
    args = parser.parse_args()

    try:
        logger.info("Pipeline started.")
        raw_data = load_data(args.train_data_path)
        cleaned_data = process_cian(raw_data)
        X_train, X_test, y_train, y_test = preprocess_data(cleaned_data, args.test_size)
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        save_model(model, args.trained_model_path)
        logger.info("Pipeline finished successfully.")
    except Exception as e:
        logger.critical(f"Critical error during pipeline execution: {e}", exc_info=True)
        print(f"Critical Error: {e}")


if __name__ == "__main__":
    main()