# %%
import awswrangler as wr
import pandas as pd

from sklearn.metrics import silhouette_score
from data_functions import get_clusterer, clustered_embeddings

# %%
retracted_articles = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet"
)
negative_class = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/negative_class.parquet"
)

# %%
cols = ["doi", "tsne_1", "tsne_2"]

combined = pd.concat([retracted_articles[cols], negative_class[cols]], axis=0)

# %%
scores = []
distances = []
sizes = []
num_clusters = []
pct_clusters = []

# %%
distance_list = [0.1, 0.2, 0.3, 1.5]

for distance in distance_list:
    for size in range(50, 1500, 50):
        clusterer = get_clusterer(combined, size, distance)
        clustering_data, num, pct = clustered_embeddings(clusterer, combined)
        score = silhouette_score(
            clustering_data[["tsne_1", "tsne_2"]],
            clustering_data["cluster_label"].to_numpy(),
        )
        scores.append(score)
        distances.append(distance)
        sizes.append(size)
        num_clusters.append(num)
        pct_clusters.append(pct)

size_df = pd.DataFrame({"size": sizes})
distance_df = pd.DataFrame({"distance": distances})
score_df = pd.DataFrame({"score": scores})
num_clusters_df = pd.DataFrame({"num clusters": num_clusters})
pct_df = pd.DataFrame({"pct clusters": pct_clusters})

out_table = pd.concat(
    [size_df, distance_df, num_clusters_df, pct_df, score_df], axis=1
).sort_values(by="score", ascending=False)
out_table

# %%
clusterer = get_clusterer(combined, 300, 0.3)
clustering_data, num, pct = clustered_embeddings(clusterer, combined)

# %%
retracted_articles = retracted_articles.merge(
    clustering_data[["doi", "cluster_label", "exemplar"]], on="doi", how="left"
).drop_duplicates(subset=["doi"])
negative_class = negative_class.merge(
    clustering_data[["doi", "cluster_label", "exemplar"]], on="doi", how="left"
).drop_duplicates(subset=["doi"])

# %%
retracted_articles.to_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet",
    index=False,
)
negative_class.to_parquet(
    "/workspaces/academic-paper-retractions/data/negative_class.parquet", index=False
)


# %%
