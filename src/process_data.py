import os
import glob
import logging
import pandas as pd
import argparse

file_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(file_dir, "logs", "data_processing.log")
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


def save_processed_data(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=True)
    logger.info(f"Processed data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Process raw apartment data.')
    parser.add_argument('--input_path', type=str,
                        default=os.path.join(file_dir, "..", "data", "raw"),
                        help='Path to raw data (CSV files)')
    parser.add_argument('--output_path', type=str,
                        default=os.path.join(file_dir, "..", "data", "processed", "processed_data.csv"),
                        help='Path to save processed data')

    args = parser.parse_args()

    try:
        logger.info("Data processing started.")
        raw_data = load_data(args.input_path)
        processed_data = process_cian(raw_data)
        save_processed_data(processed_data, args.output_path)
        logger.info("Data processing completed successfully.")
    except Exception as e:
        logger.critical(f"Critical error during data processing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()