import requests

def get_movie_details(movie_title, api_key):
    url = f"http://www.omdbapi.com?t={movie_title}&plot=full&apikey={api_key}"
    res = requests.get(url).json()
    if res.get("Response") == "True":
        result = res.get("Plot", "N/A"), res.get("Poster", "N/A")
        plot = result[0]
        poster = result[1]
        return plot, poster
    return "N/A", "N/A"
