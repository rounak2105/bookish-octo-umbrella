import pandas as pd
import ast
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify

# ---------- Step 0: Set Up CPU Device ----------
device = torch.device("cpu")

# ---------- Step 1: Load the Filtered Movies JSON ----------
print("Loading filtered movies data from 'filtered_movies.json.gz' ...")
movies_df = pd.read_json('filtered_movies.json.gz', orient='records', lines=True, compression='gzip')
print(f"Loaded {len(movies_df)} movies.")

# ---------- Step 2: Define Helper Functions ----------

def clean_text(text):
    """Clean text by lowering case and removing extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_genres(genres_field):
    """Extract genre names from a genres field (string or list)."""
    try:
        if isinstance(genres_field, str):
            genres_list = ast.literal_eval(genres_field)
        else:
            genres_list = genres_field
    except Exception:
        return ""
    if isinstance(genres_list, list):
        return ", ".join([genre.get("name", "") for genre in genres_list])
    return ""

def extract_release_year(release_date):
    """Extract the year from a release date string (format 'YYYY-MM-DD')."""
    if isinstance(release_date, str) and len(release_date) >= 4:
        return release_date[:4]
    return "Unknown"

def create_combined_text(row):
    """
    Create a combined text field using title, overview, genres, special_attribruite,
    and release year. This text is used to generate embeddings.
    """
    title = clean_text(row.get('title', ""))
    overview = clean_text(row.get('overview', ""))
    genres = parse_genres(row.get('genres', ""))
    special = clean_text(row.get('special_attribruite', ""))
    release_year = extract_release_year(row.get('release_date', ""))
    # Upweight genres by repeating them.
    weighted_genres = f"[GENRE]{genres}[/GENRE] " * 2
    combined = (
        f"[TITLE]{title}[/TITLE] "
        f"[OVERVIEW]{overview}[/OVERVIEW] "
        f"{weighted_genres}"
        f"[SPECIAL]{special}[/SPECIAL] "
        f"[RELEASE_YEAR]{release_year}[/RELEASE_YEAR]"
    )
    return combined.strip()

# Create the 'combined_text' field for generating embeddings.
movies_df['combined_text'] = movies_df.apply(create_combined_text, axis=1)

# ---------- Step 3: Load Embeddings ----------
embedding_file = "full_movie_embeddings.npy"
# Always load embeddings since file is assumed to exist.
embeddings = np.load(embedding_file, allow_pickle=True)
embeddings = torch.tensor(embeddings).to(device)
print("Loaded embeddings from file.")

# Load the model (not used for embedding computation in this script)
model = SentenceTransformer("all-MiniLM-L6-v2")
model = model.to(device)

# ---------- Step 4: Define the Recommendation Function ----------
def get_recommendations(query, top_k=5):
    """
    Given a query, compute its embedding, find the top_k most similar movies,
    and return results as a dictionary.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    # Move tensor to CPU and convert to numpy array
    cos_scores = cos_scores.cpu().numpy()

    top_indices = np.argpartition(-cos_scores, range(top_k))[:top_k]
    top_indices = top_indices[np.argsort(-cos_scores[top_indices])]

    results = []
    for idx in top_indices:
        row = movies_df.iloc[idx]
        result = {
            "id": int(row.get('id', 0)),
            "title": row.get('title', ''),
            "vote_average": float(row.get('vote_average', 0)),
            "poster_url": row.get('poster_path', ''),
            "backdrop_path": row.get('backdrop_path', ''),
            "special_attribruite": row.get('special_attribruite', ''),
            "genres": parse_genres(row.get('genres', '')),
            "overview": row.get('overview', ''),
            "release_year": extract_release_year(row.get('release_date', ''))
        }
        results.append(result)
    return results

# ---------- Step 5: Create Flask API ----------
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend_api():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Please provide a query in JSON payload with key "query".'}), 400
    query = data['query']
    recommendations = get_recommendations(query)
    return jsonify(recommendations)

if __name__ == "__main__":
    # For testing locally; in production, use a proper WSGI server.
    app.run(host='0.0.0.0', port=5000, debug=True)
