# %%
import pandas as pd

from tqdm import tqdm

# from intsol_toolkit import snowflake_utils

# %%
article_data = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet"
)

# %%
subject_cat_dist = (
    pd.DataFrame(article_data["subject_cat_desc"].value_counts())
    .reset_index()
    .rename(columns={"subject_cat_desc": "count", "index": "subject_cat_desc"})
)
article_data["retracted"] = True

# %%
out_df = pd.DataFrame(
    columns=[
        "doi",
        "article_id",
        "article_title",
        "full_source_title",
        "subject_cat_desc",
        "publisher_group",
        "year_published",
        "ARTICLE_OPEN_ACCESS_STATUS",
        "FWCI",
        "citations",
    ]
)

for i, row in tqdm(subject_cat_dist.iterrows(), total=subject_cat_dist.shape[0]):
    if row["subject_cat_desc"] == "Women's Studies":
        STMT = f"""
            SELECT  article.doi,
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
            WHERE lower(subject.subject_cat_desc) like 'women%'
            AND article.doi IS NOT NULL
            ORDER BY RANDOM()
            LIMIT {row["count"]}
        """
        conn = snowflake_utils.connect_to_snowflake()
        df = pd.read_sql(STMT, conn)
        out_df = pd.concat([out_df, df], axis=0)
    else:
        STMT = f"""
            SELECT  article.doi,
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
            WHERE lower(subject.subject_cat_desc) like '{row["subject_cat_desc"].lower()}'
            AND article.doi IS NOT NULL
            ORDER BY RANDOM()
            LIMIT {row["count"]}
        """
        conn = snowflake_utils.connect_to_snowflake()
        df = pd.read_sql(STMT, conn)
        out_df = pd.concat([out_df, df], axis=0)

# %%
out_df["retracted"] = False

# %%
n = 1000
chunked_dois = [out_df[i : i + n] for i in range(0, out_df.shape[0], n)]

abstract_data = pd.DataFrame()

for chunk in tqdm(chunked_dois, total=len(chunked_dois)):
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
out_df = out_df.merge(papers_grouped, on="doi", how="left")
out_df = out_df[~out_df["concat_abstract"].isna()]

# %%
out_df.to_parquet(
    "/workspaces/academic-paper-retractions/data/negative_class.parquet", index=False
)
