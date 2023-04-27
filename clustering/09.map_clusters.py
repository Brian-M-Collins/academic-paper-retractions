# %%
import pandas as pd

from src.data_functions import *
from src.plotting_functions import *

# %%
retracted_articles = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/retracted_article_details.parquet"
)
negative_class = pd.read_parquet(
    "/workspaces/academic-paper-retractions/data/negative_class.parquet"
)

cols = [
    "doi",
    "article_title",
    "full_source_title",
    "year_published",
    "subject_cat_desc",
    "cluster_label",
    "article_open_access_status",
    "tsne_1",
    "tsne_2",
    "citations",
]

combined = pd.concat([retracted_articles[cols], negative_class[cols]], axis=0)

# %%
cluster_labels = pd.read_csv(
    "/workspaces/academic-paper-retractions/data/cluster_labels.csv"
).rename(columns={"Preliminary Label": "Topic", "Cluster": "Label"})

# %%
colors = get_color_sequence()
colors1 = get_color_sequence_topics(cluster_labels)

# %%
cluster_map = plot_clusters(retracted_articles, colors)

# %%
retracted_avg_cites = get_average_cites(retracted_articles)
retracted_avg_change = get_change(
    retracted_articles,
    min(retracted_articles["year_published"]),
    max(retracted_articles["year_published"]),
)
retracted_stats = generate_stats(
    retracted_articles.rename(columns={"full_source_title": "publication_name"})
)

# %%
retracted_biblio_fig = plot_fig_label(
    retracted_stats,
    cluster_labels,
    colors1,
    retracted_avg_cites,
    retracted_avg_change,
    min(retracted_articles["year_published"]),
    max(retracted_articles["year_published"]),
    "Retracted Articles",
)

# %%
retracted_avg_oa = get_average_OA(combined)

# %%
retracted_oa_plot = plot_OA_papers(
    combined.merge(cluster_labels, left_on="cluster_label", right_on="Label", how="left")
    .drop_duplicates("doi")
    .rename(columns={"article_open_access_status": "art_oa_status"}),
    colors1,
    retracted_avg_oa,
    "retracted",
)

# %%
negative_avg_cites = get_average_cites(negative_class)
negative_avg_change = get_change(
    negative_class, min(negative_class["year_published"]), max(negative_class["year_published"])
)
negative_stats = generate_stats(
    negative_class.rename(columns={"full_source_title": "publication_name"})
)

# %%
negative_biblio_fig = plot_fig_label(
    negative_stats,
    cluster_labels,
    colors1,
    negative_avg_cites,
    negative_avg_change,
    min(negative_class["year_published"]),
    max(negative_class["year_published"]),
    "Published Class",
)  #
