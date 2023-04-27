# %%
import numpy as np
import pandas as pd

from openTSNE.affinity import PerplexityBasedNN
from openTSNE import TSNEEmbedding, initialization

# %%
retracted_articles = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet"
)
negative_class = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/negative_class.parquet"
)

cols = ["doi", "embeddings"]

pos = retracted_articles[cols]
neg = negative_class[cols]

# %%
combined = pd.concat([pos, neg], axis=0)
combined = combined.drop_duplicates(subset=["doi"]).reset_index(drop=True)

# %%
n_jobs = -1  # use all available cores
METHOD = "pynndescent"  # NN calculation method
METRIC = "cosine"  # cosine performs well for high dimensional data, specter embeddings are 768 dimensions
RANDOM_STATE = 42
PERPLEXITY = 500  # based on openTSNE documentation, 500 for large datasets

# %%
x_train = np.vstack(combined["embeddings"])
y_train = combined["doi"]

# %%
affinities_train = PerplexityBasedNN(
    x_train,
    perplexity=PERPLEXITY,
    method=METHOD,
    metric=METRIC,
    n_jobs=n_jobs,
    random_state=RANDOM_STATE,
)

# %%
init_train = initialization.pca(x_train, random_state=RANDOM_STATE)

# %%
embedding_train = TSNEEmbedding(
    init_train,
    affinities_train,
    negative_gradient_method="fft",
    n_jobs=n_jobs,
)

# %%
# Optimize embedding: 1 Early exaggeration phase
embedding_train_1 = embedding_train.optimize(
    n_iter=250,
    exaggeration=12,
    momentum=0.5,
)

# %%
embedding_train_2 = embedding_train_1.optimize(n_iter=750, momentum=0.8)

# %%
tsne_df_train = pd.DataFrame(embedding_train_2, columns=["tsne_1", "tsne_2"]).assign(doi=y_train)

# %%
retracted_articles = retracted_articles.merge(tsne_df_train, on="doi", how="left")
negative_class = negative_class.merge(tsne_df_train, on="doi", how="left")

# %%
retracted_articles.to_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet",
    index=False,
)
negative_class.to_parquet(
    "/workspaces/academic-paper-retractions/data/negative_class.parquet", index=False
)
