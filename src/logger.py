import logging
import os
import datetime

current_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
log_file_name = f"Log_{current_time}.log"
log_file_path = os.path.join("logs", log_file_name)

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename = log_file_path,
    level = logging.INFO,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

#logger.info("Logger innitialized successfully.")
