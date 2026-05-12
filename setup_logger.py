import logging
import os
import sys
from datetime import datetime

def setup_logger(log_dir="logs"):
    """Set up a logger and redirect print() output to it"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(log_dir, timestamp)
    os.makedirs(folder_path, exist_ok=True)

    log_file = os.path.join(folder_path, "output.log")
    
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect print() to log automatically
    class PrintLogger:
        def write(self, message):
            if message.strip():
                logger.info(message.strip())

        def flush(self):
            pass  # Required for compatibility

    sys.stdout = PrintLogger()  # Redirect print() to logger

    return logger
