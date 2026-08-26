import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from rapidfuzz import process, fuzz
from fastapi import FastAPI
from pydantic import BaseModel

DATA_DIR = "data"

# ---- Load artifacts once at startup ----
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


# ---- Core engine ----

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


def recommend(liked_titles, k=10, exclude_franchise=True):
    matched_ids, unmatched = match_titles(liked_titles)
    if not matched_ids:
        return {"error": "no titles matched", "unmatched": unmatched}

    liked_idx = [anime_id_map_reverse[a] for a in matched_ids if a in anime_id_map_reverse]
    if not liked_idx:
        return {"error": "matched titles not in ratings data", "unmatched": unmatched}

    matched_titles = [title_by_animeid[a] for a in matched_ids]

    # Sum similarity rows for everything the user likes, mask the inputs themselves
    scores = np.asarray(item_similarity[liked_idx].sum(axis=0)).flatten()
    scores[liked_idx] = -1

    # Overfetch candidates, then filter out franchise-mates of the inputs
    candidate_idx = scores.argsort()[::-1][: k * 5]

    recs = []
    for i in candidate_idx:
        if scores[i] <= 0:
            break
        cand_title = title_by_animeid.get(anime_id_map[i], "?")
        if exclude_franchise:
            # vs the user's inputs
            if any(is_same_franchise(cand_title, lt) for lt in matched_titles):
                continue
            # vs recommendations already accepted — keep only the first entry per franchise
            if any(is_same_franchise(cand_title, r["title"]) for r in recs):
                continue
        recs.append({"animeID": anime_id_map[i], "title": cand_title})
        if len(recs) == k:
            break

    return {
        "matched": matched_titles,
        "unmatched": unmatched,
        "recommendations": recs,
    }


# ---- FastAPI app ----

app = FastAPI(
    title="Anime Recommender",
    description="Item-based collaborative filtering over 118M ratings. "
                "Enter anime you like, get recommendations.",
)


class RecRequest(BaseModel):
    liked_anime: list[str]
    k: int = 10
    exclude_franchise: bool = True


@app.post("/recommend")
def recommend_endpoint(req: RecRequest):
    return recommend(req.liked_anime, k=req.k, exclude_franchise=req.exclude_franchise)


# ---- Interactive CLI (python api.py) ----

if __name__ == "__main__":
    raw = input("Enter anime you like (comma-separated): ")
    liked = [t.strip() for t in raw.split(",") if t.strip()]

    if not liked:
        print("No titles entered.")
    else:
        result = recommend(liked, k=10)

        if "error" in result:
            print(f"\n{result['error']}")
            if result["unmatched"]:
                print("Couldn't match:", ", ".join(result["unmatched"]))
        else:
            print("\nInterpreted your list as:", ", ".join(result["matched"]))
            if result["unmatched"]:
                print("Couldn't match:", ", ".join(result["unmatched"]))
            print("\nRecommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                print(f"  {i}. {rec['title']}")