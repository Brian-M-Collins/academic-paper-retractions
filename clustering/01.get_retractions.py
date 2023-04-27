# %%
import requests
import os
import math

import pandas as pd

from more_itertools import chunked

# %%
CHUNKSIZE = 500

# %%
key = os.getenv("ATYPON_PKG_KEY")
ENDPOINT = "https://pkg-graphql-api.atypon.com/v2/"
headers = {"Authorization": f"Basic {key}"}

# %%
# get total num of retractions to establish the number of pages required
query = """query{
IPublicationCount(retracted: true)
}
"""

r = requests.post(ENDPOINT, json={"query": query}, headers=headers)
retracted_counts = r.json()["data"]["IPublicationCount"]

secondary_pages = math.ceil((retracted_counts / CHUNKSIZE) - 1)

# %%
doi_out_list = []

first_query = """ 
    query{publist: IPublications(retracted: true, scrollId: "cf7b9d17a3a911f24664", size: 500) {
        doi
    }
    }
"""

r = requests.post(ENDPOINT, json={"query": first_query}, headers=headers)

for x in r.json()["data"]["publist"]:
    doi_out_list.append(x["doi"])

# %%
for num in range(0, secondary_pages):
    subsequent_query = """
        IPublications(scrollId: "cf7b9d17a3a911f24664") {
            doi
        }
    """

    r = requests.post(ENDPOINT, json={"query": subsequent_query}, headers=headers)
    for x in r.json()["data"]["publist"]:
        doi_out_list.append(x["doi"])

# %%
final_list = pd.DataFrame({"doi": [x for x in doi_out_list if x != None]})

# %%
final_list.to_csv("/workspaces/academic-paper-retractions/data/retracted_dois.csv", index=False)
