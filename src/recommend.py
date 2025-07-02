import joblib
import logging
import streamlit as st

#setup logging
logging.basicConfig(
    level=logging.INFO,
    format = '[%(asctime)s] %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler("reccomend.log",encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info("Loading Data...")
try:
    df = joblib.load('src/df_cleaned.pkl')
    import gzip
    def decompress_two_parts_to_file(input_part1_path, input_part2_path, output_file_path):
        with open(input_part1_path, 'rb') as f_in1:
            compressed_part1 = f_in1.read()

        with open(input_part2_path, 'rb') as f_in2:
            compressed_part2 = f_in2.read()

        decompressed_part1 = gzip.decompress(compressed_part1)
        decompressed_part2 = gzip.decompress(compressed_part2)

        combined_data = decompressed_part1 + decompressed_part2

        with open(output_file_path, 'wb') as f_out:
            f_out.write(combined_data)
    decompress_two_parts_to_file('cosine_sim_part1.pkl.gz', 'cosine_sim_part2.pkl.gz', 'cosine_sim.pkl')
    logging.info("Decompressed cosine similarity matrix.")
    cosine_sim = joblib.load('cosine_sim.pkl')
    logging.info("Data loaded  successfully.")
except Exception as e:
    logging.error("Failed to load requested files : %s", str(e))
    raise e
@st.cache_data
def recommend_movie(movie_name, top_n=5):
    logging.info("Reccomending movies for : '%s'", movie_name)
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        logging.warning("Movie Not Found in Database.")
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    logging.info("Top %d recommendations ready.",top_n)
    # create DataFrame with clean serial numbers starting from 1
    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1  # Start from 1 instead of 0
    result_df.index.name = "S.No."

    return result_df
