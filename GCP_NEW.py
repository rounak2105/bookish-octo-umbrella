import json
import gzip
import pandas as pd
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
import gc

# ---------- Step 0: Set Up CPU Device ----------
device = torch.device("cpu")

# ---------- Step 1: Load the Filtered Movies JSON Efficiently ----------
print("Loading filtered movies data from 'filtered_movies.json.gz' ...")
# If each line in the gzipped file is a valid JSON record:
rows = []
with gzip.open("filtered_movies.json.gz", "rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        rows.append(row)
movies_df = pd.DataFrame(rows)
print(f"Loaded {len(movies_df)} movies.")

# Optionally, downcast numeric columns:
for col in movies_df.select_dtypes(include=["float64"]).columns:
    movies_df[col] = pd.to_numeric(movies_df[col], downcast="float")

# ---------- Step 2: Define Helper Functions (Consider Vectorization if possible) ----------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text.lower()).strip()

def parse_genres(genres_field):
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
    if isinstance(release_date, str) and len(release_date) >= 4:
        return release_date[:4]
    return "Unknown"

def create_combined_text(row):
    title = clean_text(row.get('title', ""))
    overview = clean_text(row.get('overview', ""))
    genres = parse_genres(row.get('genres', ""))
    special = clean_text(row.get('special_attribruite', ""))
    release_year = extract_release_year(row.get('release_date', ""))
    weighted_genres = f"[GENRE]{genres}[/GENRE] " * 2
    return f"[TITLE]{title}[/TITLE] [OVERVIEW]{overview}[/OVERVIEW] {weighted_genres}[SPECIAL]{special}[/SPECIAL] [RELEASE_YEAR]{release_year}[/RELEASE_YEAR]".strip()

# Create combined_text; if DataFrame is very large, consider processing in chunks or using vectorized operations.
movies_df['combined_text'] = movies_df.apply(create_combined_text, axis=1)

# Optionally drop columns not needed further:
# movies_df.drop(columns=['title', 'overview', 'genres', 'special_attribruite', 'release_date'], inplace=True)

# Run garbage collection to free memory:
gc.collect()

# ---------- Step 3: Load Embeddings Using Memory Mapping ----------
embedding_file = "full_movie_embeddings.npy"
embeddings_np = np.load(embedding_file, allow_pickle=True, mmap_mode='r')
embeddings = torch.tensor(embeddings_np).to(device)
print("Loaded embeddings from file.")

# ---------- Step 4: Load the Model ----------
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# ---------- Step 5: Define the Recommendation Function ----------
def get_recommendations(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    cos_scores = cos_scores.cpu().numpy()

    top_indices = np.argpartition(-cos_scores, range(top_k))[:top_k]
    top_indices = top_indices[np.argsort(-cos_scores[top_indices])]

    results = []
    for idx in top_indices:
        row = movies_df.iloc[idx]
        results.append({
            "id": int(row.get('id', 0)),
            "title": row.get('title', ''),
            "vote_average": float(row.get('vote_average', 0)),
            "poster_url": row.get('poster_path', ''),
            "backdrop_path": row.get('backdrop_path', ''),
            "special_attribruite": row.get('special_attribruite', ''),
            "genres": parse_genres(row.get('genres', '')),
            "overview": row.get('overview', ''),
            "release_year": extract_release_year(row.get('release_date', ''))
        })
    return results

# ---------- Step 6: Create Flask API ----------
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
    app.run(host='0.0.0.0', port=5000, debug=True)
