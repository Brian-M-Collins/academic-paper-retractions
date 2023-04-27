# %%
import pandas as pd

from embedding_functions import prepare_data, embed_df_abstracts

# %%
retracted_articles = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet"
)
neg_class = pd.read_parquet("/workspaces/academic-paper-retractions/data/negative_class.parquet")

# %%
cols = ["doi", "article_title", "concat_abstract"]
positive = retracted_articles[cols]
negative = neg_class[cols]

# %%
combined = pd.concat([positive, negative], axis=0)

# %%
prepared = prepare_data(combined)

# %%
embeddings = embed_df_abstracts(prepared)

# %%
retracted_articles = retracted_articles.merge(
    embeddings[["doi", "embeddings"]], on="doi", how="left"
).drop_duplicates(subset=["doi"])
neg_class = neg_class.merge(
    embeddings[["doi", "embeddings"]], on="doi", how="left"
).drop_duplicates(subset=["doi"])

# %%
retracted_articles.to_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet",
    index=False,
)
neg_class.to_parquet(
    "/workspaces/academic-paper-retractions/data/negative_class.parquet", index=False
)
# %%
