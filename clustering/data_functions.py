import awswrangler as wr
import colorcet as cc
import hdbscan
import numpy as np
import pandas as pd


# Get category average citations for whole timeframe:
def get_average_cites(loaded_data):
    """
    Determine the category average citations
     Parameters
    ----------
    loaded_data: Pandas dataframe
        The merged tsne and bibliometric data.
    Returns
    -------
    category_average_citations_per_paper: int
        The average citations received per paper for the dataset.
    """
    category_data = pd.DataFrame(loaded_data)
    # first we need to drop the duplicates - one row per article (arbitrary publisher_group in cases with 2 p_gs?)
    category_data = category_data.drop_duplicates(subset=["tsne_1", "tsne_2"]).reset_index(
        drop=True
    )

    category_citations = category_data["citations"].sum()
    category_papers = len(category_data)

    category_average_citations_per_paper = (category_citations / category_papers).astype(int)

    return category_average_citations_per_paper


# Get category average OA percentage for whole timeframe:
def get_average_OA(loaded_data):
    """
    Determine the category OA percentage
    Parameters
    ----------
    loaded_data: Pandas dataframe
        The merged tsne and bibliometric data.
    Returns
    -------
    category_average_OA: int
        The percentage of OA publications for the dataset.
    """

    category_data = pd.DataFrame(loaded_data)
    # first we need to drop the duplicates -one row per article (arbitrary publisher_group in cases with 2 p_gs?)
    category_data = category_data.drop_duplicates(subset=["tsne_1", "tsne_2"]).reset_index(
        drop=True
    )

    # convert online open tag to open access
    category_data["open_access_binary"] = "No"
    category_data.loc[
        category_data["article_open_access_status"].str.contains("Open"),
        "open_access_binary",
    ] = "Open Access"
    category_data.loc[
        category_data["article_open_access_status"].str.startswith("Online"),
        "open_access_binary",
    ] = "Open Access"

    category_data["Open_Access_Boolean"] = np.where(
        (category_data["open_access_binary"] == "Open Access"),
        True,
        False,
    )

    category_OA = category_data["Open_Access_Boolean"].sum()
    category_papers = len(category_data)

    category_average_OA = ((category_OA / category_papers) * 100).astype(int)

    return category_average_OA


# Get category average change for whole cluster
def get_change(loaded_data, start_year, end_year):
    """
    Determine the growth of the category over time.
    Parameters
    ----------
    loaded_data: Pandas dataframe
        The merged tsne and bibliometric data.
    start_year: int
        The earliest year with which to calculate the growth stats.
    end_year: int
        The latest year with which to calculate the growth stats.
    Returns
    -------
    category_change: int
        The percentage of change (growth or decline) in the category over time.
    """
    category_data = pd.DataFrame(loaded_data)
    # first we need to drop the duplicates - one row per article (arbitrary publisher_group in cases with 2 p_gs?)
    category_data = category_data.drop_duplicates(subset=["tsne_1", "tsne_2"]).reset_index(
        drop=True
    )

    category_data["Total_Year_Count"] = (
        category_data.groupby(["year_published"])["doi"].transform("count").astype(int)
    )
    category_data = category_data.drop_duplicates(
        ["Total_Year_Count", "year_published"], keep="first"
    ).sort_values(by=["Total_Year_Count", "year_published"], ascending=[True, True])
    values = range(start_year, end_year - 1, 1)

    growth_data = pd.DataFrame(
        category_data[category_data.year_published.isin(values)]
    ).reset_index()
    category_change = (
        (
            (growth_data.at[1, "Total_Year_Count"] - growth_data.at[0, "Total_Year_Count"])
            / growth_data.at[0, "Total_Year_Count"]
        )
        * 100
    ).astype(int)

    return category_change


# Build clustering object
def get_clusterer(loaded_data, min_cluster_size, cluster_selection_epsilon=1):
    """
    Cluster embeddings with HDBSCAN and create a "clusterer" object to call for predictions on new embedding tsnes.
    Parameters
    ----------
    loaded_data: Pandas dataframe
        The merged tsne and bibliometric data.
    min_cluster_size: int
        The minimum number of points a cluster should include
    cluster_selection_epsilon: float
        The maximum distance the clustering algorithm should use to infer relationships (bigger distance = lower similarity required).
    Returns
    -------
    clusterer: object
        The output of the HDBScan data fitting.
    """

    data = pd.DataFrame(loaded_data)
    # first we need to drop the duplicates - one row per article (arbitrary publisher_group in cases with 2 p_gs?)
    clustering_data = data.drop_duplicates(subset=["tsne_1", "tsne_2"]).reset_index(drop=True)

    if len(clustering_data) < 2:
        print(
            "Not enough papers to cluster. Please select a different subject/country combination."
        )

    else:

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            gen_min_span_tree=True,
            cluster_selection_method="leaf",
            cluster_selection_epsilon=cluster_selection_epsilon,
            prediction_data=True,
        ).fit(clustering_data[["tsne_1", "tsne_2"]])

        return clusterer

    # Build cluster dataframe:


def clustered_embeddings(clusterer, loaded_data):
    """
    Convert the clusterer object to a dataframe for analytics and identifies exemplar papers.
    Parameters
    ----------
    clusterer: object
        The output of the HDBScan data fitting.
    loaded_data: Pandas dataframe
        The merged tsne and bibliometric data.
    Returns
    -------
    labeled_clustering_data_exemplars: Pandas dataframe
        A dataframe of all the articles in the data set with cluster and exemplar labels.
    """

    clusterer = clusterer
    # join the labels from the clusterer to the dataframe by index.
    data = pd.DataFrame(loaded_data)
    # first we need to drop the duplicates
    clustering_data = data.drop_duplicates(subset=["tsne_1", "tsne_2"]).reset_index(drop=True)

    pct_clustered = 1 - np.count_nonzero(clusterer.labels_ == -1) / len(clusterer.labels_)

    number_clusters = clusterer.labels_.max() + 1

    if number_clusters == 0:
        print("Cluster Report")
        print("No clusters could be formed. Please reduce the minimum cluster size.")

    else:
        print("Cluster Report")
        print(f"{len(clusterer.labels_)} articles analyzed.")
        print(f"{number_clusters} clusters found.")
        print(f"{pct_clustered:.0%} articles clustered.")

        # Add the labels to the dataframe
    cluster_labels = pd.DataFrame(clusterer.labels_, columns=["cluster_label"])

    labeled_clustering_data = pd.concat(
        [clustering_data, cluster_labels], axis=1, join="inner"
    ).sort_values(by="cluster_label")

    labeled_clustering_data["cluster_label"] = labeled_clustering_data["cluster_label"].astype(
        "category"
    )

    # Label individual papers as exemplars
    exemplars = (
        pd.DataFrame(clusterer.exemplars_, columns=["exemplars"])
        .explode("exemplars")
        .rename_axis("exemplar")
        .reset_index()
    )

    exemplars["tsne_1"], exemplars["tsne_2"] = zip(*exemplars.pop("exemplars"))
    exemplars["exemplar"] = True

    labeled_clustering_data_exemplars = pd.merge(
        labeled_clustering_data, exemplars, how="outer", on=["tsne_1", "tsne_2"]
    )
    labeled_clustering_data_exemplars["exemplar"] = labeled_clustering_data_exemplars[
        "exemplar"
    ].fillna(False)

    return labeled_clustering_data_exemplars, number_clusters, pct_clustered


def generate_stats(clustering_data):
    """
    Calculate all the bibliometric stats for the whole dataset.
     Parameters
    ----------
    clustering data: Pandas dataframe
        The cluster-labelled data for each article.
    Returns
    -------
    labeled_clustering_data: Pandas dataframe
        A dataframe of all the articles in the dataset with grouped statistics on clusters (article counts, citations, etc).
    """

    labeled_clustering_data = pd.DataFrame(clustering_data)

    labeled_clustering_data["year_published"] = labeled_clustering_data["year_published"].astype(
        int
    )

    labeled_clustering_data["citations"] = labeled_clustering_data["citations"].astype(int)

    labeled_clustering_data["Papers per Cluster"] = (
        labeled_clustering_data.groupby(["cluster_label"])["doi"].transform("count").astype(int)
    )
    labeled_clustering_data["Cites per Cluster"] = (
        labeled_clustering_data.groupby(["cluster_label"])["citations"]
        .transform("sum")
        .astype(int)
    )
    labeled_clustering_data["Average Cites in Cluster"] = (
        labeled_clustering_data["Cites per Cluster"]
        / labeled_clustering_data["Papers per Cluster"]
    )
    labeled_clustering_data["Max Cites in Cluster"] = labeled_clustering_data.groupby(
        ["cluster_label"]
    )["citations"].transform("max")
    labeled_clustering_data["Min Cites in Cluster"] = labeled_clustering_data.groupby(
        ["cluster_label"]
    )["citations"].transform("min")
    labeled_clustering_data["Median Cites in Cluster"] = labeled_clustering_data.groupby(
        ["cluster_label"]
    )["citations"].transform("median")

    labeled_clustering_data["Cluster_Count"] = (
        labeled_clustering_data.groupby(["year_published", "cluster_label"])["doi"]
        .transform("count")
        .astype(int)
    )
    labeled_clustering_data["Cluster_Cites"] = (
        labeled_clustering_data.groupby(["year_published", "cluster_label"])["citations"]
        .transform("sum")
        .astype(int)
    )
    labeled_clustering_data["Total_Year_Count"] = (
        labeled_clustering_data.groupby(["year_published"])["doi"].transform("count").astype(int)
    )
    labeled_clustering_data["Journal Papers per Cluster"] = (
        labeled_clustering_data.groupby(["cluster_label", "publication_name"])["doi"]
        .transform("count")
        .astype(int)
    )
    labeled_clustering_data["Journal Cites per Cluster"] = (
        labeled_clustering_data.groupby(["cluster_label", "publication_name"])["citations"]
        .transform("sum")
        .astype(int)
    )
    labeled_clustering_data["Journal Average Cites in Cluster"] = (
        labeled_clustering_data["Journal Cites per Cluster"]
        / labeled_clustering_data["Journal Papers per Cluster"]
    )
    labeled_clustering_data["Journal Max Cites in Cluster"] = labeled_clustering_data.groupby(
        ["cluster_label", "publication_name"]
    )["citations"].transform("max")

    labeled_clustering_data["Journal Min Cites in Cluster"] = labeled_clustering_data.groupby(
        ["cluster_label", "publication_name"]
    )["citations"].transform("min")
    labeled_clustering_data["Journal Median Cites in Cluster"] = labeled_clustering_data.groupby(
        ["cluster_label", "publication_name"]
    )["citations"].transform("median")
    labeled_clustering_data["Journal Median Cites in Cluster"] = labeled_clustering_data.groupby(
        ["cluster_label", "publication_name"]
    )["citations"].transform("median")

    return labeled_clustering_data
