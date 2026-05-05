import json
import os
import pickle
from functools import lru_cache
from pathlib import Path

import bs4 as bs
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from requests.adapters import HTTPAdapter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parent
MAIN_DATA_PATH = BASE_DIR / "main_data.csv"
MOVIES_DATA_PATH = BASE_DIR / "movies.csv"
REVIEWS_PATH = BASE_DIR / "reviews.txt"

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p"
DEFAULT_TMDB_API_KEY = "5ce2ef2d7c461dea5b4e04900d1c561e"
TMDB_API_KEY = (os.environ.get("TMDB_API_KEY") or DEFAULT_TMDB_API_KEY).strip()

MOVIE_NOT_FOUND_MESSAGE = (
    "Sorry! The movie you requested is not in our database. "
    "Please check the spelling or try with some other movies"
)

app = Flask(__name__)

tmdb_session = requests.Session()
tmdb_retry = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=0.6,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",),
)
tmdb_adapter = HTTPAdapter(max_retries=tmdb_retry)
tmdb_session.mount("https://", tmdb_adapter)

_movie_data = None
_catalog_data = None
_similarity_matrix = None
_sentiment_model = None
_sentiment_vectorizer = None

GENRE_NAMES = {
    12: "Adventure",
    14: "Fantasy",
    16: "Animation",
    18: "Drama",
    27: "Horror",
    28: "Action",
    35: "Comedy",
    36: "History",
    37: "Western",
    53: "Thriller",
    80: "Crime",
    99: "Documentary",
    878: "Science Fiction",
    9648: "Mystery",
    10402: "Music",
    10749: "Romance",
    10751: "Family",
    10752: "War",
    10770: "TV Movie",
}


class MovieLookupError(Exception):
    """Raised when a requested movie cannot be found."""


class ExternalServiceError(Exception):
    """Raised when TMDB or IMDb cannot be reached cleanly."""


def artifact_path(filename):
    artifacts_dir = BASE_DIR / "Artifacts"
    nested_path = artifacts_dir / filename
    return nested_path if nested_path.exists() else BASE_DIR / filename


def title_case(value):
    return str(value or "").strip().capitalize()


def normalize_title(value):
    return str(value or "").strip().lower()


def image_url(path, size="w500"):
    if not path:
        return ""
    return f"{TMDB_IMAGE_BASE_URL}/{size}{path}"


def format_runtime(minutes):
    try:
        minutes = int(minutes)
    except (TypeError, ValueError):
        return "Not available"

    if minutes <= 0:
        return "Not available"

    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours} hr {mins} min"
    if hours:
        return f"{hours} hr"
    return f"{mins} min"


def format_release_date(value):
    if not value:
        return "Not available"
    try:
        parsed = pd.to_datetime(value)
    except (TypeError, ValueError):
        return str(value)
    return parsed.strftime("%b %d, %Y")


def format_vote_count(value):
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def convert_to_list(value):
    if isinstance(value, list):
        return value
    if not value:
        return []

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        items = str(value).split('","')
        items[0] = items[0].replace('["', "")
        items[-1] = items[-1].replace('"]', "")
        return items


def load_movie_data():
    global _movie_data

    if _movie_data is not None:
        return _movie_data

    if not MAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing movie data file: {MAIN_DATA_PATH}")

    movie_data = pd.read_csv(MAIN_DATA_PATH)
    required_columns = {"movie_title", "comb"}
    missing_columns = required_columns - set(movie_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"main_data.csv is missing required columns: {missing}")

    movie_data["movie_title"] = movie_data["movie_title"].astype(str).str.strip().str.lower()
    movie_data["comb"] = movie_data["comb"].fillna("")
    _movie_data = movie_data
    return _movie_data


def load_catalog_data():
    global _catalog_data

    if _catalog_data is not None:
        return _catalog_data

    if not MOVIES_DATA_PATH.exists():
        _catalog_data = pd.DataFrame()
        return _catalog_data

    catalog_data = pd.read_csv(MOVIES_DATA_PATH)
    for column in ["title", "original_title", "genres", "overview", "status"]:
        if column in catalog_data.columns:
            catalog_data[column] = catalog_data[column].fillna("")

    catalog_data["_title_key"] = catalog_data.get("title", pd.Series(dtype=str)).map(normalize_title)
    catalog_data["_original_title_key"] = catalog_data.get("original_title", pd.Series(dtype=str)).map(normalize_title)
    _catalog_data = catalog_data
    return _catalog_data


def local_movie_details(title):
    catalog_data = load_catalog_data()
    if catalog_data.empty:
        return None

    title_key = normalize_title(title)
    matches = catalog_data[
        (catalog_data["_title_key"] == title_key)
        | (catalog_data["_original_title_key"] == title_key)
    ]

    if matches.empty:
        matches = catalog_data[catalog_data["_title_key"].str.contains(title_key, regex=False, na=False)]

    if matches.empty:
        return None

    row = matches.iloc[0]
    return {
        "id": int(row["id"]) if "id" in row and pd.notna(row["id"]) else None,
        "title": row.get("title") or row.get("original_title") or title_case(title),
        "original_title": row.get("original_title") or row.get("title") or title_case(title),
        "poster_path": "",
        "backdrop_path": "",
        "overview": row.get("overview") or "Overview not available.",
        "vote_average": row.get("vote_average") if pd.notna(row.get("vote_average")) else 0,
        "vote_count": row.get("vote_count") if pd.notna(row.get("vote_count")) else 0,
        "release_date": row.get("release_date") or "",
        "runtime": row.get("runtime") if pd.notna(row.get("runtime")) else 0,
        "status": row.get("status") or "Not available",
        "genres": [{"name": genre} for genre in str(row.get("genres") or "").split() if genre],
        "imdb_id": "",
        "source_note": "Showing local catalog details because TMDB is temporarily unavailable.",
    }


def create_similarity():
    global _similarity_matrix

    movie_data = load_movie_data()
    if _similarity_matrix is not None:
        return movie_data, _similarity_matrix

    count_matrix = CountVectorizer().fit_transform(movie_data["comb"])
    _similarity_matrix = cosine_similarity(count_matrix)
    return movie_data, _similarity_matrix


def rcmd(movie_title, limit=10):
    movie_title = str(movie_title or "").strip().lower()
    movie_data, similarity = create_similarity()

    if movie_title not in movie_data["movie_title"].unique():
        return MOVIE_NOT_FOUND_MESSAGE

    movie_index = movie_data.loc[movie_data["movie_title"] == movie_title].index[0]
    scores = sorted(
        enumerate(similarity[movie_index]),
        key=lambda item: item[1],
        reverse=True,
    )
    recommendation_indexes = [index for index, _ in scores[1 : limit + 1]]
    return [movie_data["movie_title"][index] for index in recommendation_indexes]


def get_suggestions():
    movie_data = load_movie_data()
    return [title_case(title) for title in movie_data["movie_title"]]


def train_sentiment_model():
    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(f"Missing sentiment training data: {REVIEWS_PATH}")

    reviews = pd.read_csv(REVIEWS_PATH, sep="\t", names=["label", "review"])
    review_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="ascii",
        stop_words="english",
    )
    features = review_vectorizer.fit_transform(reviews["review"])
    model = MultinomialNB()
    model.fit(features, reviews["label"])
    return model, review_vectorizer


def ensure_sentiment_model():
    global _sentiment_model, _sentiment_vectorizer

    if _sentiment_model is not None and _sentiment_vectorizer is not None:
        return _sentiment_model, _sentiment_vectorizer

    model_path = artifact_path("nlp_model.pkl")
    vectorizer_path = artifact_path("tranform.pkl")

    try:
        with model_path.open("rb") as model_file:
            _sentiment_model = pickle.load(model_file)
        with vectorizer_path.open("rb") as vectorizer_file:
            _sentiment_vectorizer = pickle.load(vectorizer_file)

        sample_vector = _sentiment_vectorizer.transform(["This movie was good."])
        _sentiment_model.predict(sample_vector)
    except Exception as exc:
        print(f"Could not use saved sentiment artifacts ({exc}). Training from reviews.txt.")
        _sentiment_model, _sentiment_vectorizer = train_sentiment_model()

    return _sentiment_model, _sentiment_vectorizer


def get_tmdb(endpoint, **params):
    cleaned_params = tuple(
        sorted((key, str(value)) for key, value in params.items() if value not in (None, ""))
    )
    return _get_tmdb(endpoint.strip("/"), cleaned_params)


@lru_cache(maxsize=512)
def _get_tmdb(endpoint, params):
    if not TMDB_API_KEY:
        raise ExternalServiceError("TMDB_API_KEY is not configured.")

    request_params = dict(params)
    request_params["api_key"] = TMDB_API_KEY
    url = f"{TMDB_BASE_URL}/{endpoint}"

    try:
        response = tmdb_session.get(url, params=request_params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ExternalServiceError(f"TMDB request failed for {endpoint}: {exc}") from exc

    return response.json()


def search_tmdb_movie(title):
    payload = get_tmdb("search/movie", query=title)
    results = payload.get("results") or []
    if not results:
        raise MovieLookupError(f"No TMDB result found for '{title}'.")
    return results[0]


def details_from_search_result(result, title):
    genre_names = [
        GENRE_NAMES[genre_id]
        for genre_id in result.get("genre_ids", [])
        if genre_id in GENRE_NAMES
    ]
    return {
        "id": result.get("id"),
        "title": result.get("title") or result.get("original_title") or title_case(title),
        "original_title": result.get("original_title") or result.get("title") or title_case(title),
        "poster_path": result.get("poster_path") or "",
        "backdrop_path": result.get("backdrop_path") or "",
        "overview": result.get("overview") or "Overview not available.",
        "vote_average": result.get("vote_average") or 0,
        "vote_count": result.get("vote_count") or 0,
        "release_date": result.get("release_date") or "",
        "runtime": 0,
        "status": "Not available",
        "genres": [{"name": name} for name in genre_names],
        "imdb_id": "",
        "source_note": "Showing partial TMDB search details because the full TMDB detail endpoint is temporarily unavailable.",
    }


def recommendation_cards(recommended_titles):
    cards = []
    for recommended_title in recommended_titles:
        try:
            result = search_tmdb_movie(recommended_title)
        except (MovieLookupError, ExternalServiceError):
            result = {}

        cards.append(
            {
                "title": result.get("title") or title_case(recommended_title),
                "query": recommended_title,
                "poster": image_url(result.get("poster_path"), "w500"),
                "rating": result.get("vote_average"),
            }
        )
    return cards


def cast_cards(movie_id, limit=10):
    try:
        credits = get_tmdb(f"movie/{movie_id}/credits")
    except ExternalServiceError:
        return []

    cards = []
    for person in (credits.get("cast") or [])[:limit]:
        person_id = person.get("id")
        details = {}
        if person_id:
            try:
                details = get_tmdb(f"person/{person_id}")
            except ExternalServiceError:
                details = {}

        cards.append(
            {
                "id": person_id or len(cards),
                "name": person.get("name") or "Unknown",
                "character": person.get("character") or "Unknown",
                "profile": image_url(person.get("profile_path"), "w500"),
                "birthday": format_release_date(details.get("birthday")),
                "place": details.get("place_of_birth") or "Not available",
                "bio": details.get("biography") or "Biography not available.",
            }
        )
    return cards


@lru_cache(maxsize=128)
def fetch_imdb_reviews(imdb_id):
    if not imdb_id:
        return []

    url = f"https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Could not fetch IMDb reviews: {exc}")
        return []

    soup = bs.BeautifulSoup(response.text, "lxml")
    review_nodes = soup.find_all("div", {"class": "text show-more__control"})

    review_texts = []
    seen = set()
    for node in review_nodes:
        text = node.get_text(" ", strip=True)
        if text and text not in seen:
            seen.add(text)
            review_texts.append(text)

    if not review_texts:
        return []

    try:
        model, vectorizer = ensure_sentiment_model()
        predictions = model.predict(vectorizer.transform(review_texts))
    except Exception as exc:
        print(f"Could not classify IMDb reviews: {exc}")
        return [{"text": text, "sentiment": "Unscored"} for text in review_texts[:8]]

    return [
        {"text": text, "sentiment": "Good" if int(prediction) else "Bad"}
        for text, prediction in zip(review_texts[:8], predictions[:8])
    ]


def build_movie_context(title):
    source_note = ""

    try:
        search_result = search_tmdb_movie(title)
    except ExternalServiceError as exc:
        details = local_movie_details(title)
        if details is None:
            raise exc
        search_result = {}
        source_note = details.get("source_note", "")
    except MovieLookupError:
        details = local_movie_details(title)
        if details is None:
            raise
        search_result = {}
        source_note = details.get("source_note", "")
    else:
        movie_id = search_result.get("id")
        try:
            details = get_tmdb(f"movie/{movie_id}")
        except ExternalServiceError as exc:
            print(f"{exc}. Falling back to local or partial movie details.")
            details = local_movie_details(search_result.get("title") or title)
            if details is None:
                details = details_from_search_result(search_result, title)
            source_note = details.get("source_note", "")

    display_title = (
        details.get("title")
        or details.get("original_title")
        or search_result.get("title")
        or title
    )
    movie_id = search_result.get("id") or details.get("id")

    recommendations = rcmd(display_title)
    recommendation_note = ""
    if isinstance(recommendations, str):
        recommendation_note = "No local recommendations are available for this title yet."
        recommendations = []

    return {
        "title": display_title,
        "poster": image_url(details.get("poster_path"), "w500"),
        "backdrop": image_url(details.get("backdrop_path"), "original"),
        "overview": details.get("overview") or "Overview not available.",
        "vote_average": details.get("vote_average") or "0",
        "vote_count": format_vote_count(details.get("vote_count")),
        "release_date": format_release_date(details.get("release_date")),
        "runtime": format_runtime(details.get("runtime")),
        "status": details.get("status") or "Not available",
        "genres": ", ".join(genre.get("name", "") for genre in details.get("genres", [])).strip(", "),
        "casts": cast_cards(movie_id) if search_result.get("id") else [],
        "reviews": fetch_imdb_reviews(details.get("imdb_id")),
        "movie_cards": recommendation_cards(recommendations),
        "recommendation_note": recommendation_note,
        "source_note": source_note,
        "error": "",
    }


def build_local_fallback_context(title, reason=""):
    details = local_movie_details(title) or {
        "title": title_case(title),
        "original_title": title_case(title),
        "poster_path": "",
        "backdrop_path": "",
        "overview": "Movie details are temporarily unavailable.",
        "vote_average": 0,
        "vote_count": 0,
        "release_date": "",
        "runtime": 0,
        "status": "Temporarily unavailable",
        "genres": [],
        "imdb_id": "",
    }

    display_title = details.get("title") or title_case(title)
    recommendations = rcmd(display_title)
    if isinstance(recommendations, str):
        recommendations = []

    note = "TMDB is temporarily unavailable, so this page is using local catalog data."
    if reason:
        note = f"{note} The app will use live TMDB details again automatically when the connection recovers."

    return {
        "title": display_title,
        "poster": "",
        "backdrop": "",
        "overview": details.get("overview") or "Movie details are temporarily unavailable.",
        "vote_average": details.get("vote_average") or "0",
        "vote_count": format_vote_count(details.get("vote_count")),
        "release_date": format_release_date(details.get("release_date")),
        "runtime": format_runtime(details.get("runtime")),
        "status": details.get("status") or "Temporarily unavailable",
        "genres": ", ".join(genre.get("name", "") for genre in details.get("genres", [])).strip(", "),
        "casts": [],
        "reviews": [],
        "movie_cards": [
            {
                "title": title_case(recommended_title),
                "query": recommended_title,
                "poster": "",
                "rating": None,
            }
            for recommended_title in recommendations
        ],
        "recommendation_note": "",
        "source_note": note,
        "error": "",
    }


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", suggestions=get_suggestions())


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "movies": len(load_movie_data()),
            "tmdb_configured": bool(TMDB_API_KEY),
            "pid": os.getpid(),
        }
    )


@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form.get("name", "")
    recommendations = rcmd(movie)
    if isinstance(recommendations, str):
        return recommendations
    return "---".join(recommendations)


@app.route("/movie", methods=["POST"])
def movie():
    title = request.form.get("title", "").strip()
    if not title:
        return render_template("recommend.html", error="Enter a movie title."), 400

    try:
        return render_template("recommend.html", **build_movie_context(title))
    except MovieLookupError as exc:
        return render_template("recommend.html", error=str(exc)), 404
    except ExternalServiceError as exc:
        print(f"{exc}. Rendering local fallback response.")
        return render_template("recommend.html", **build_local_fallback_context(title, str(exc))), 200
    except Exception as exc:
        print(f"Unexpected movie lookup failure: {exc}")
        return render_template("recommend.html", error="Something went wrong while building this movie page."), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    return movie()


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        host="0.0.0.0",
        port=5000,
        use_reloader=False,
    )
