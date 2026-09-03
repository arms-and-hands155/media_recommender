# media_recommender

I decided to do this project after I finished Hunter x Hunter and couldnt find any ideas for what anime to watch next. I built an anime recommender built on 148M real user ratings. The main feature is cold-start recommendations: you type in a few shows you like and get recommendations back, without needing an account or any rating history in the system.

```
$ curl -X POST localhost:8000/recommend/neural \
    -H "Content-Type: application/json" \
    -d '{"liked_anime": ["Hunter x Hunter", "Attack on Titan", "One Piece"], "k": 10}'

→ Fullmetal Alchemist: Brotherhood, Death Note, Naruto Shippuden, One Punch Man,
  Bleach, Your Name., Code Geass, Steins;Gate, My Hero Academia, Parasyte: The Maxim
```

## Results

I built four models in increasing order of sophistication. Each one is evaluated with Precision@10 and Recall@10 on a held-out 20% test split.

| Model | Precision@10 | Recall@10 | Notes |
|---|---|---|---|
| Popularity baseline | 0.068 | 0.067 | Same top-10 for everyone, sets the floor |
| Item-item CF | 0.178 | 0.172 | Cosine similarity over audience overlap |
| ALS | 0.181 | 0.163 | 64-dim latent factors via the `implicit` library |
| Two-tower neural net | 0.210 | 0.170 | ALS user factors + tag-based content features |

Two things to keep in mind when reading the table:

1. The two-tower model ranks over the ~4.8K anime that survive a three-way ID alignment (AniList / MyAnimeList / ratings dataset), while the other models rank over the full ~20K catalog. A smaller candidate pool makes hits easier, so some of the two-tower's precision advantage comes from that.
2. Each model's evaluation samples 2,000 test users independently, so there's some sampling noise between runs (roughly ±0.005 in my experience).

## Why rarity-weighted tags instead of sentence embeddings

The content features for the neural net originally came from sentence-transformer embeddings of plot synopses. It didn't hold up. It rated Mob Psycho 100 as more similar to Attack on Titan than Fullmetal Alchemist: Brotherhood was, which anyone familiar with these shows can tell you is wrong.

The problem is that synopsis text describes plot events, not tone or theme. And when I added genre/tag text to the embeddings, common low-signal tags like "Male Protagonist" and "Shounen" dominated, while rare meaningful tags like "Cannibalism" or "Steampunk" got diluted on titles with long tag lists.

So I replaced it with explicit rarity-weighted tag vectors. Every title is a vector over the tag vocabulary, and each tag is weighted by how rare it is: `log(total_items / (tag_count + 1))`. Sharing "Steampunk" now counts for a lot more than sharing "Male Protagonist", and titles with long tag lists no longer drown their meaningful tags in noise.

## How it fits together

```
Kaggle ratings (148M rows)         AniList GraphQL API (top 5K anime)
        │                                   │
        ▼                                   ▼
  01_data_collection ── implicit labels, train/test split, 3-way ID crosswalk
        │                                   │
        ▼                                   ▼
  02_baseline ──────── popularity + item-item CF (item_similarity.npz)
        │
  03_ALS ───────────── 64-dim matrix factorization (als_model.pkl)
        │                                   │
  04_content-embedding ── rarity-weighted tag vectors ──┐
        │                                               │
  05_NN ───────────── two-tower net: [ALS user factors] × [ALS item factors ⊕ tag vectors]
        │              BPR loss, epoch checkpointing
        ▼
  api.py ──────────── FastAPI serving both engines to cold-start users
```

The data sources use different ID systems (AniList IDs, MyAnimeList IDs, and the ratings dataset's own internal IDs). Notebook 01 builds the bridge using AniList's `idMal` field plus some URL parsing on the ratings metadata. About 4.8K of the 5K AniList titles make it through the full chain.

The trained models only know users from the training set, so the API has to manufacture a user on the fly:

- `POST /recommend` uses item-CF. It sums the similarity rows for the shows you entered and ranks the full ~20K catalog. Basically free at inference time.
- `POST /recommend/neural` uses the two-tower model. It builds a sparse interaction row from your shows, folds it into the ALS space with `recalculate_user` (a fresh factor vector, no retraining), runs that through the trained user tower, and scores it against precomputed item embeddings.

Both endpoints share the same serving layer: fuzzy title matching via rapidfuzz (so "atack on titan" still works), a response that echoes back which titles it matched so you can catch misinterpretations, and franchise filtering. The filtering runs at two levels — recommendations are checked against your inputs and against each other — so you get one entry per franchise instead of four seasons of Attack on Titan.

## Repo layout

```
notebooks/
  01_data_collection.ipynb    # AniList pull, Kaggle ingest, labels, split, ID crosswalk
  02_baseline.ipynb           # popularity + item-item CF, evaluation harness
  03_ALS.ipynb                # matrix factorization
  04_content-embedding.ipynb  # rarity-weighted tag vectors
  05_NN.ipynb                 # two-tower model, training, evaluation
api.py                        # FastAPI app + interactive CLI
data/                         # gitignored, see below
```

## Setup

The data isn't in the repo (the ratings dataset alone is 148M rows), but everything regenerates from public sources: ratings come from Kaggle via `kagglehub` (`ramazanturann/user-animelist-dataset`) and metadata comes from the AniList GraphQL API, which needs no auth. The fetching code checkpoints as it goes and respects AniList's rate limits.

I developed the notebooks on Google Colab because the pipeline needs around 12GB of RAM at peak (the train/test split and the 20K×20K similarity matrix are the expensive parts), which is more than my 8GB laptop could handle. The notebooks have memory instrumentation and explicit cleanup between stages to stay inside Colab's free tier.

Running the notebooks in order (01 through 05) produces every artifact the API needs. Drop these into `data/` next to `api.py`: `item_similarity.npz`, `animes.parquet`, `anime_id_map.json`, `als_model.pkl`, `two_tower_checkpoint.pt`, `all_item_embeddings.npy`, `aligned_animeid_list.json`.

Then install the dependencies:

```bash
pip install fastapi uvicorn scipy pandas pyarrow rapidfuzz torch implicit
```

## Try it yourself

The quickest way is the interactive CLI. It loads the models (takes 30–60 seconds the first time), prompts you for shows, and prints recommendations from both engines side by side so you can compare them:

```
$ python api.py
Enter anime you like (comma-separated): One Piece, Hunter x Hunter, Attack on Titan

===== ITEM-CF =====
Interpreted your list as: One Piece, Hunter x Hunter, Attack on Titan
  1. Death Note
  2. One Punch Man
  ...

===== TWO-TOWER =====
Interpreted your list as: One Piece, Hunter x Hunter, Attack on Titan
  1. Fullmetal Alchemist: Brotherhood
  2. Death Note
  ...
```

Typos and partial titles are fine — the fuzzy matcher resolves them, and the "Interpreted your list as" line shows you what it matched, so you can tell if it guessed wrong.

To test the actual API, start the server:

```bash
uvicorn api:app --reload
```

and open **http://localhost:8000/docs** in your browser. FastAPI generates an interactive page there: expand either endpoint, click "Try it out", edit the request body, and hit Execute to see the JSON response. Something like:

```json
{
  "liked_anime": ["Steins;Gate", "Death Note", "Code Geass"],
  "k": 10,
  "exclude_franchise": true
}
```

Setting `exclude_franchise` to `false` turns off the franchise filtering, if you'd rather see sequels ranked normally. Or hit it from the terminal:

```bash
curl -X POST localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"liked_anime": ["Steins;Gate", "Death Note"], "k": 10}'
```

Try the same list against both `/recommend` and `/recommend/neural` — the overlap and the differences between the two engines are half the fun.

## Limitations and future work

- The ratings data has no timestamps, so the train/test split is random rather than temporal. A time-based split (train on the past, predict the future) would be the stricter evaluation, but it isn't possible with this dataset.
- The two-tower model's ~4.8K catalog is smaller than the other models'. Evaluating everything on the shared subset (or growing the aligned set) would make the comparison fully apples-to-apples.
- Item embeddings are precomputed, so new titles need an embedding refresh. Since the item tower takes tag vectors as input, it could in principle serve brand-new titles from tags alone.