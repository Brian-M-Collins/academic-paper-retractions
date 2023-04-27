# %%
from keybert import KeyBERT
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm

tqdm.pandas()

# %%
retracted_articles = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet"
)
negative_class = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/negative_class.parquet"
)

# %%
cols = ["doi", "concat_abstract"]

combined = pd.concat([retracted_articles[cols], negative_class[cols]], axis=0)

# %%
kb_model = KeyBERT()

# %%
combined["keybert_keywords"] = combined["concat_abstract"].progress_apply(
    lambda x: kb_model.extract_keywords(x, keyphrase_ngram_range=(1, 3))
)

# %%
combined["keywords"] = combined["keybert_keywords"].apply(
    lambda x: ", ".join([a_tuple[0] for a_tuple in x])
)

# %%
retracted_articles = retracted_articles.merge(
    combined[["doi", "keywords"]], on="doi", how="left"
).drop_duplicates(subset=["doi"])
negative_class = negative_class.merge(
    combined[["doi", "keywords"]], on="doi", how="left"
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
