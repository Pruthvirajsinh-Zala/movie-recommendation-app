import joblib
import logging

#setup logging
logging.basicConfig(
    level=logging.INFO,
    format = '[%(asctime)s] %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler("reccomend.log",encodeing='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info("Loading Data...")
try:
    df = joblib.load('df_cleaned.pkl')
    cosine_sim = joblib.load('cosine_sim.pkl')