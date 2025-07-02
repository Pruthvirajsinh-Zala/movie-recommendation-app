import json
import streamlit as st
from recommend import df, recommend_movie
from omdb_utils import get_movie_details
import preprocess

config = json.load(open('src/config.json'))

exec(open('src/preprocess.py').read())
#OMDB api key
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

st.set_page_config(
    page_title='Movie Recommender',
    page_icon='🎬',
    layout='centered'
)

st.title('Movie Recommender System 🎬')

#using title instade of movie name
movie_list = sorted(df['title'].dropna().unique())
selected_movie = st.selectbox("🎬 Select a movie:",movie_list)

if st.button("🚀 Recommend Similar Movies"):
    with st.spinner("Finding Similar Movies..."):
        recommendations = recommend_movie(selected_movie)
        if recommendations is None or recommendations.empty:
            st.warning("Sorry, no recommendations found.")

        else:
            st.success("Top Similar Movies:")
            for _, row in recommendations.iterrows():
                movie_title = row['title']
                plot, poster = get_movie_details(movie_title, OMDB_API_KEY)

                with st.container():
                    col1, col2 = st.columns([1,3])
                    with col1:
                        if poster != "N/A":
                            st.image(poster,width=100)
                        else:
                            st.write("❌ No Poster Found")
                    with col2:
                        st.markdown(f"###  {movie_title}")
                        st.markdown(f"*{plot}*" if plot != "N/A" else "_Plot not available_")
