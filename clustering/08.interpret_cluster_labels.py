# %%
import plotly.express as px
import pandas as pd

# %%
retracted_articles = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet"
)
negative_class = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/negative_class.parquet"
)

# %%
retracted_articles["Reason(s)"] = (
    retracted_articles["Reason(s)"].str.strip("+").str.replace("+", ", ")
)

# %%
cols = ["doi", "article_title", "keywords", "subject_cat_desc", "exemplar", "cluster_label"]


# %%
combined = pd.concat([retracted_articles[cols], negative_class[cols]], axis=0)

# %%
cluster_counts = (
    pd.DataFrame(combined.groupby("cluster_name").size())
    .reset_index()
    .rename(columns={0: "# Articles"})
)
cluster_counts = cluster_counts[cluster_counts["cluster_name"] != "Unclustered"].sort_values(
    "# Articles", ascending=False
)

# %%
counts_bar = px.bar(
    cluster_counts,
    x="cluster_name",
    y="# Articles",
    labels={"cluster_name": "Topic"},
    title="Plotting number of articles by identified article cluster",
)


# %%
cluster_keywords = pd.DataFrame()
for num in range(0, 12):
    cluster_keywords["Cluster_" + str(num)] = (
        combined["keywords"][combined["cluster_label"] == num]
        .str.split(",/s+", expand=True)
        .stack()
        .value_counts()
        .nlargest(10)
        .index
    )

cluster_subjects = pd.DataFrame()
for num in range(0, 12):
    cluster_subjects["Cluster_" + str(num)] = (
        combined["subject_cat_desc"][combined["cluster_label"] == num]
        .value_counts()
        .nlargest(10)
        .index
    )

# %%
exemplars = combined[combined["exemplar"] == True].to_csv("exemplars.csv", index=False)

# %%
cluster_labels = pd.read_csv(
    "/workspaces/academic-paper-retractions/data/cluster_labels.csv",
)

# %%
combined = combined.merge(cluster_labels, left_on="cluster_label", right_on="Cluster", how="left")

# %%
combined = combined.rename(columns={"Preliminary Label": "cluster_name"})

# %%
cluster_counts = (
    pd.DataFrame(combined.groupby("cluster_name").size())
    .reset_index()
    .rename(columns={0: "# Articles"})
)
cluster_counts = cluster_counts[cluster_counts["cluster_name"] != "Unclustered"].sort_values(
    "# Articles", ascending=False
)

# %%
import plotly_express as px

counts_bar = px.bar(
    cluster_counts,
    x="cluster_name",
    y="# Articles",
    labels={"cluster_name": "Topic"},
    title="Plotting number of articles by identified article cluster",
)


# %%
cluster_retractions = pd.DataFrame()
for num in range(0, 12):
    cluster_retractions["Cluster_" + str(num)] = (
        retracted_articles["Reason(s)"][retracted_articles["cluster_label"] == num]
        .str.split(",/s+", expand=True)
        .stack()
        .value_counts()
        .nlargest(10)
        .index
    )

cluster_retractions_values = pd.DataFrame()
for num in range(0, 12):
    cluster_retractions_values["Cluster_" + str(num)] = (
        retracted_articles["Reason(s)"][retracted_articles["cluster_label"] == num]
        .str.split(",/s+", expand=True)
        .stack()
        .value_counts()
    )


# %%
clusters = cluster_retractions.columns
out_df = pd.DataFrame()

for col in clusters:
    retraction_total = pd.DataFrame(cluster_retractions_values[col]).sum()
    retraction_values = (
        pd.DataFrame(cluster_retractions_values[col])
        .sort_values(col, ascending=False)
        .head(10)
        .reset_index(drop=False)
        .rename(columns={"index": f"{col}_reason", f"{col}": f"{col}_value"})
    )
    retraction_values["proportion"] = round(
        retraction_values[f"{col}_value"].apply(lambda x: (x / retraction_total) * 100), 2
    )
    retraction_values[col] = (
        retraction_values[f"{col}_reason"]
        + " ("
        + retraction_values["proportion"].astype(str)
        + "%)"
    )
    out_df = pd.concat([out_df, retraction_values[col]], axis=1)

out_df.columns = cluster_labels["Preliminary Label"].drop(index=0).tolist()

# %%
out_df.to_csv(
    "/workspaces/academic-paper-retractions/data/cluster_retraction_reasons.csv", index=False
)

# %%
