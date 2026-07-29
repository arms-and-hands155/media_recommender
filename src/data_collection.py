import requests
import time
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANIME_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "anime_data.jsonl")
MANGA_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "manga_data.jsonl")

ANILIST_URL = "https://graphql.anilist.co" #Getting data from AniList API

QUERY = """
query ($page: Int, $type: MediaType) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    media(type: $type, sort: POPULARITY_DESC) {
      id
      title { romaji english }
      genres
      tags { name }
      description
      coverImage { large }
      averageScore
      popularity
      format
    }
  }
}
"""

def fetch_page(page, media_type="ANIME"):
    variables = {"page": page, "type": media_type}
    response = requests.post(ANILIST_URL, json={"query": QUERY, "variables": variables})
    response.raise_for_status()
    return response.json()["data"]["Page"]
i=100
data = []
#Get Anime data
while True:
    x=fetch_page(i)
    data.extend(x['media'])
    
    if i == 100: #API caps at 5000 requests 
        with open(ANIME_DATA_PATH, 'a') as file:
            for item in data:
                file.write(f'{json.dumps(item)}\n')
        break
    i+=1
    
    if i % 10 == 0:
        with open(ANIME_DATA_PATH, 'a') as file:
            for item in data:
                file.write(f'{json.dumps(item)}\n')
        data.clear()
        
    time.sleep(2) #Ensures we wont have too many requests too quick

data.clear()
i=1
#Get Manga data
while True:
    x=fetch_page(i, media_type="MANGA")
    data.extend(x['media'])
    
    if i == 100: 
        with open(MANGA_DATA_PATH, 'a') as file:
            for item in data:
                file.write(f'{json.dumps(item)}\n')
        break
    i+=1
    
    if i % 10 == 0:
        with open(MANGA_DATA_PATH, 'a') as file:
            for item in data:
                file.write(f'{json.dumps(item)}\n')
        data.clear()
        
    time.sleep(2) 
