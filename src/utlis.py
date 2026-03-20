from src.logger import logger
from src.exceptions import CustomException
import os
import joblib

def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path, compress=0)
    except Exception as e:
        logger.error(f"Error occurred while saving object: {e}")
        raise CustomException(e)

def load_object(file_path: str) -> object | None:
    try:
        return joblib.load(file_path)
    except Exception as e:
        logger.error(f"Error occurred while loading object: {e}")
        raise CustomException(e)