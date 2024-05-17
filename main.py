from vald import Vald
from dotenv import load_dotenv
import os
import time
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

populate_folder_path = os.getenv("VALD_FOLDER_PATH")

def main():

    logging.info('Starting...')
    
    vald = Vald()
    logging.info("Vald initialized")
    vald.populate_folders(populate_folder_path, None)
    logging.info("Vald first populate")

    while True:
        interval = 300
        countdown = interval
        while countdown > 0:
            minutes, seconds = divmod(countdown, 60)
            logging.info(f"{minutes}m{seconds}s remaining")
            time.sleep(30)
            countdown -= 30
        vald.populate_folders(populate_folder_path, None)

if __name__ == "__main__":
    main()
