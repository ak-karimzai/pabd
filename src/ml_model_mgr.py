import argparse
import os.path
import cianparser
import joblib
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
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


def load_data(room_numbers):
    logger.info("Starting data loading from cianparser...")
    n_rooms = 1
    moscow_parser = cianparser.CianParser(location="Москва")
    main_df = pd.DataFrame()

    while n_rooms <= room_numbers:
        logger.info(f"Fetching data for {n_rooms}-room apartments...")
        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": 2,
                "object_type": "secondary"
            })
        df = pd.DataFrame(data)
        main_df = pd.concat([main_df, df], axis=0, ignore_index=True)
        n_rooms += 1

    logger.info(f"Finished loading data. Total rows: {len(main_df)}")
    return main_df


def process_cian(main_df):
    logger.info("Starting data processing...")
    max_price, max_meters = 100_000_000, 100

    try:
        main_df['url_id'] = main_df['url'].apply(lambda x: x.split('/')[-2])
    except Exception as e:
        logger.error("Error extracting 'url_id': %s", str(e))
        raise

    new_df = main_df[['url_id', 'total_meters', 'price']].set_index('url_id')
    filtered_df = new_df[(new_df['price'] < max_price) & (new_df['total_meters'] < max_meters)]

    logger.info(f"Data processed and filtered. Remaining rows: {len(filtered_df)}")
    return filtered_df


def preprocess_data(data, test_size, random_state):
    logger.info("Starting data preprocessing...")
    X = data[['total_meters']]
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

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
    parser = argparse.ArgumentParser(description='Automate the machine learning model lifecycle.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for data splitting (default: 42)')
    parser.add_argument('--max_room_numb', type=int, default=1, help='Maximum number of rooms to fetch (default: 1)')
    parser.add_argument('--trained_model_path', type=str, default=os.path.join("..", "models", "linear_reg_model.pkl"),
                        help='Path to save the trained model (default: ../models/linear_reg_model.pkl)')
    args = parser.parse_args()

    try:
        logger.info("Pipeline started.")
        data = load_data(args.max_room_numb)
        data = process_cian(data)
        X_train, X_test, y_train, y_test = preprocess_data(data, args.test_size, args.random_state)
        model = train_model(X_train, y_train)
        mse = test_model(model, X_test, y_test)
        save_trained_model(model, args.trained_model_path)
        logger.info("Pipeline finished successfully.")
    except Exception as e:
        logger.exception("An error occurred during pipeline execution: %s", str(e))
        print(f"Error: {e}")