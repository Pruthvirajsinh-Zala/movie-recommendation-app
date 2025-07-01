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
    logging.info("Data loaded  successfully.")
except Exception as e:
    logging.error("Failed to load requested files : %s", str(e))
    raise e

def recommend_movie(movie_name, top_n=5):
    logging.info("Reccomending movies for : '%s'", movie_name)
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        logging.warning("Movie Not Found in Database.")
        return None
    idx = idx[0]