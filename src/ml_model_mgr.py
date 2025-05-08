import glob
import joblib
import argparse
import os.path
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(file_dir, "logs", "ml_logs.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=log_file_path,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


def load_data(train_data_path):
    file_pattern = os.path.join(train_data_path, "*.csv")
    logger.debug(f"Looking for CSV files matching pattern: {file_pattern}")
    
    file_list = glob.glob(file_pattern)
    logger.info(f"Found {len(file_list)} CSV files.")

    main_df = pd.DataFrame()
    for i, file in enumerate(file_list, start=1):
        try:
            logger.debug(f"Loading file {i}: {file}")
            df = pd.read_csv(file)
            main_df = pd.concat([main_df, df], axis=0)
        except Exception as e:
            logger.warning(f"Failed to load file {file}: {str(e)}")

    logger.info(f"Loaded data with shape: {main_df.shape}")
    return main_df


def process_cian(main_df):
    logger.info("Starting data processing...")
    max_price, max_meters = 100_000_000, 100

    logger.info(f"main_df columns: {main_df.head()}")
    if 'url' not in main_df.columns:
        logger.error("'url' column is missing from the dataset.")
        raise KeyError("'url' column not found in DataFrame")

    try:
        main_df['url_id'] = main_df['url'].apply(lambda x: x.split('/')[-2])
        logger.debug("Extracted 'url_id' from 'url'")
    except Exception as e:
        logger.error("Error extracting 'url_id': %s", str(e))
        raise

    new_df = main_df[['url_id', 'total_meters', 'price']].set_index('url_id')
    filtered_df = new_df[(new_df['price'] < max_price) & (new_df['total_meters'] < max_meters)]

    logger.info(f"Data processed and filtered. Remaining rows: {len(filtered_df)}")
    return filtered_df


def preprocess_data(data, test_size):
    logger.info("Starting data preprocessing...")
    X = data[['total_meters']]
    y = data['price']

    logger.debug(f"Splitting data into train/test sets with test size={test_size}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    logger.debug("Scaling features using StandardScaler")
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


def test_model(model, X_test, y_test):
    logger.info("Evaluating model performance...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    logger.info(f"Mean Squared Error: {mse:.2f}")
    return mse


def save_trained_model(model, model_path):
    logger.info(f"Saving trained model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved successfully.")


def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Automate the machine learning model lifecycle.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Size of the test set (default: 0.2)')
    parser.add_argument('--train_data_path', type=str, default=os.path.join(file_dir, "..", "data", "raw"),
                        help='Path to training data (not used currently)')
    parser.add_argument('--trained_model_path', type=str,
                        default=os.path.join(file_dir, "..", "models", "linear_reg_model.pkl"),
                        help='Path to save the trained model (default: ../models/linear_reg_model.pkl)')
    args = parser.parse_args()

    try:
        logger.info("Pipeline started.")
        data = load_data(args.train_data_path)
        data = process_cian(data)
        X_train, X_test, y_train, y_test = preprocess_data(data, args.test_size)
        model = train_model(X_train, y_train)
        mse = test_model(model, X_test, y_test)
        save_trained_model(model, args.trained_model_path)
        logger.info("Pipeline finished successfully.")
    except Exception as e:
        logger.critical("An unexpected error occurred during pipeline execution.", exc_info=True)
        print(f"Critical Error: {e}")


if __name__ == "__main__":
    main()