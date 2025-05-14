import os

file_dir = os.path.dirname(os.path.abspath(__file__))
server_log_file_path = os.path.join(file_dir, "logs", "server.log")

log_config = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": server_log_file_path,
            "formatter": "default",
        }
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
}

LRM_TRAINED_PATH = os.path.join("models", "apartment_price_model.pkl")
PREPROCESSOR_PATH = os.path.join("models", "apartment_price_model_preprocessor.pkl")
