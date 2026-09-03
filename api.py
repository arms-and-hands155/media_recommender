import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import load_npz, csr_matrix
from rapidfuzz import process, fuzz
from fastapi import FastAPI
from pydantic import BaseModel

DATA_DIR = "data"

# ---- Item-CF artifacts ----
print("loading similarity matrix...")
item_similarity = load_npz(f"{DATA_DIR}/item_similarity.npz")

animes = pd.read_parquet(f"{DATA_DIR}/animes.parquet")

with open(f"{DATA_DIR}/anime_id_map.json") as f:
    # idx (matrix row) -> animeID
    anime_id_map = {int(k): int(v) for k, v in json.load(f).items()}
anime_id_map_reverse = {v: k for k, v in anime_id_map.items()}  # animeID -> idx

# Title lookup tables
title_by_animeid = animes.set_index("animeID")["title"].to_dict()
titles_list = animes["title"].tolist()
animeids_list = animes["animeID"].tolist()

# ---- NN artifacts ----
print("loading ALS model...")
with open(f"{DATA_DIR}/als_model.pkl", "rb") as f:
    als_model = pickle.load(f)

all_item_embeddings = np.load(f"{DATA_DIR}/all_item_embeddings.npy")

with open(f"{DATA_DIR}/aligned_animeid_list.json") as f:
    aligned_animeid_list = json.load(f)

N_ITEMS = item_similarity.shape[0]  # full catalog size, matches ALS item dimension


class UserTower(nn.Module):
    def __init__(self, input_dim=64, output_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


checkpoint = torch.load(f"{DATA_DIR}/two_tower_checkpoint.pt", map_location="cpu")
user_tower = UserTower()
user_tower.load_state_dict(checkpoint["user_tower_state"])
user_tower.eval()


# ---- Shared helpers ----

def match_titles(user_titles):
    """Fuzzy-match user-typed titles to animeIDs. Returns (matched_ids, unmatched_titles)."""
    matched, unmatched = [], []
    for t in user_titles:
        result = process.extractOne(t, titles_list, scorer=fuzz.WRatio, score_cutoff=70)
        if result is None:
            unmatched.append(t)
        else:
            _, score, pos = result
            matched.append(animeids_list[pos])
    return matched, unmatched


def is_same_franchise(title_a, title_b, threshold=80):
    """Fuzzy check for whether two titles belong to the same franchise (sequels, seasons, remakes)."""
    return fuzz.partial_ratio(title_a.lower(), title_b.lower()) >= threshold


def _filter_and_collect(candidate_animeids_scores, matched_titles, liked_set, k, exclude_franchise):
    """Shared rec-collection loop: skip likes, filter franchise-mates of inputs and of accepted recs."""
    recs = []
    for animeid, score in candidate_animeids_scores:
        if animeid in liked_set:
            continue
        cand_title = title_by_animeid.get(animeid, "?")
        if exclude_franchise:
            if any(is_same_franchise(cand_title, lt) for lt in matched_titles):
                continue
            if any(is_same_franchise(cand_title, r["title"]) for r in recs):
                continue
        recs.append({"animeID": animeid, "title": cand_title})
        if len(recs) == k:
            break
    return recs


# ---- Item-CF engine ----

def recommend(liked_titles, k=10, exclude_franchise=True):
    matched_ids, unmatched = match_titles(liked_titles)
    if not matched_ids:
        return {"error": "no titles matched", "unmatched": unmatched}

    liked_idx = [anime_id_map_reverse[a] for a in matched_ids if a in anime_id_map_reverse]
    if not liked_idx:
        return {"error": "matched titles not in ratings data", "unmatched": unmatched}

    matched_titles = [title_by_animeid[a] for a in matched_ids]

    scores = np.asarray(item_similarity[liked_idx].sum(axis=0)).flatten()
    scores[liked_idx] = -1

    candidate_idx = scores.argsort()[::-1][: k * 10]
    candidates = [
        (anime_id_map[i], scores[i]) for i in candidate_idx if scores[i] > 0
    ]

    recs = _filter_and_collect(
        candidates, matched_titles, set(matched_ids), k, exclude_franchise
    )

    return {
        "matched": matched_titles,
        "unmatched": unmatched,
        "recommendations": recs,
        "model": "item-cf",
    }


# ---- Two-Tower engine ----

def recommend_neural(liked_titles, k=10, exclude_franchise=True):
    matched_ids, unmatched = match_titles(liked_titles)
    if not matched_ids:
        return {"error": "no titles matched", "unmatched": unmatched}

    liked_idx = [anime_id_map_reverse[a] for a in matched_ids if a in anime_id_map_reverse]
    if not liked_idx:
        return {"error": "matched titles not in ratings data", "unmatched": unmatched}

    matched_titles = [title_by_animeid[a] for a in matched_ids]

    # ALS fold-in: sparse row of the user's likes -> synthetic user factor vector
    user_row = csr_matrix(
        (np.ones(len(liked_idx)), ([0] * len(liked_idx), liked_idx)),
        shape=(1, N_ITEMS),
    )
    user_factors = als_model.recalculate_user(0, user_row)

    with torch.no_grad():
        user_vec = torch.tensor(np.asarray(user_factors), dtype=torch.float32).reshape(1, -1)
        user_emb = user_tower(user_vec).numpy().flatten()

    scores = all_item_embeddings @ user_emb  # over the aligned anime subset

    candidate_order = scores.argsort()[::-1]
    candidates = [(aligned_animeid_list[i], scores[i]) for i in candidate_order]

    recs = _filter_and_collect(
        candidates, matched_titles, set(matched_ids), k, exclude_franchise
    )

    return {
        "matched": matched_titles,
        "unmatched": unmatched,
        "recommendations": recs,
        "model": "two-tower",
    }


# ---- FastAPI app ----

app = FastAPI(
    title="Anime Recommender",
    description="Enter anime you like, get recommendations. "
                "/recommend uses item-based collaborative filtering over the full ~20K catalog; "
                "/recommend/neural uses a two-tower neural net (ALS fold-in + content tags) "
                "over a ~4.8K aligned subset.",
)


class RecRequest(BaseModel):
    liked_anime: list[str]
    k: int = 10
    exclude_franchise: bool = True


@app.post("/recommend")
def recommend_endpoint(req: RecRequest):
    return recommend(req.liked_anime, k=req.k, exclude_franchise=req.exclude_franchise)


@app.post("/recommend/neural")
def recommend_neural_endpoint(req: RecRequest):
    return recommend_neural(req.liked_anime, k=req.k, exclude_franchise=req.exclude_franchise)


# ---- Interactive CLI (python api.py) ----

if __name__ == "__main__":
    raw = input("Enter anime you like (comma-separated): ")
    liked = [t.strip() for t in raw.split(",") if t.strip()]

    if not liked:
        print("No titles entered.")
    else:
        for engine, label in [(recommend_neural, "TWO-TOWER")]:
            result = engine(liked, k=10)
            print(f"\n===== {label} =====")
            if "error" in result:
                print(result["error"])
                if result["unmatched"]:
                    print("Couldn't match:", ", ".join(result["unmatched"]))
                continue
            print("Interpreted your list as:", ", ".join(result["matched"]))
            if result["unmatched"]:
                print("Couldn't match:", ", ".join(result["unmatched"]))
            for i, rec in enumerate(result["recommendations"], 1):
                print(f"  {i}. {rec['title']}")