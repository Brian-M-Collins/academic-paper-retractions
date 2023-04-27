# %%
import pandas as pd

# from intsol_toolkit import snowflake_utils removed from project repo (proprietary)

# %%
retracted_dois = pd.read_csv(
    "/workspaces/academic-paper-retractions/data/retracted_dois.csv"
).dropna()

# %%
n = 1000
chunked_dois = [retracted_dois[i : i + n] for i in range(0, retracted_dois.shape[0], n)]

# %%
article_data = pd.DataFrame()

for chunk in chunked_dois:
    STMT = f"""
        SELECT      article.doi,
                    article.article_id,
                    article.article_title,
                    article.full_source_title,
                    subject.subject_cat_desc,
                    article.publisher_group,
                    article.year_published,
                    article.ARTICLE_OPEN_ACCESS_STATUS,
                    article.FWCI,
                    metrics.citations
        FROM "PROD_EDW"."EBAC"."DW_ARTICLE_EXTN" article
        JOIN "PROD_EDW"."EBAC"."DW_ABSTRACT" abstract on article.article_id = abstract.article_id
        JOIN "PROD_EDW"."EBAC"."ARTICLE_METRICS_AGG" metrics on article.article_id = metrics.article_id
        JOIN "PROD_EDW"."EBAC"."DW_SUBJECT_CATEGORY_EXTN" subject on article.article_id = subject.article_id
        WHERE article.doi in {tuple(chunk["doi"].to_list())} 
    """

    conn = snowflake_utils.connect_to_snowflake()
    df = pd.read_sql(STMT, conn)
    article_data = pd.concat([article_data, df], axis=0).drop_duplicates("doi")

# %%
abstract_data = pd.DataFrame()

abstract_dois = [article_data[i : i + n] for i in range(0, article_data.shape[0], n)]

for chunk in abstract_dois:
    STMT = f"""
        SELECT      article.doi,
                    abstract.abstract_text
        FROM "PROD_EDW"."EBAC"."DW_ARTICLE_EXTN" article
        JOIN "PROD_EDW"."EBAC"."DW_ABSTRACT" abstract on article.article_id = abstract.article_id
        WHERE article.doi in {tuple(chunk["doi"].to_list())}
    """

    conn = snowflake_utils.connect_to_snowflake()
    df = pd.read_sql(STMT, conn)
    abstract_data = pd.concat([abstract_data, df], axis=0).dropna(subset=["abstract_text"])

# %%

papers_grouped = (
    abstract_data.drop_duplicates(["doi", "abstract_text"], keep="first")
    .groupby(["doi"])["abstract_text"]
    .apply(",".join)
    .reset_index()
)

papers_grouped.columns = ["doi", "concat_abstract"]

# %%
combined_data = article_data.merge(papers_grouped, on="doi", how="left")

# %%
combined_data = combined_data.dropna(subset=["concat_abstract"])

# %%
combined_data.to_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet",
    index=False,
)

# %%
