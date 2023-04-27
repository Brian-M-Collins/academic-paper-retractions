import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import confusion_matrix
from textwrap3 import wrap


def get_color_sequence():
    """
    Generates a large categorical palette of colors with colorcet to use in the plotly plots.
    Returns
    -------
    colors: dict
        A dictionary of cluster labels and hex codes to be used for consistent topic labeling throughout analyses.
    """
    num_categories = 12
    color_swatches = cc.glasbey_light[:num_categories]

    # generate a dictionary for plotly
    col_df = pd.DataFrame(color_swatches)
    col_df["label"] = col_df.index - 1
    col_df.columns = ["color", "label"]
    col_df["label"] = col_df["label"]
    col_df.loc[0, "label"] = "unclustered"
    colors = dict(zip(col_df.label, col_df.color))

    return colors


def get_color_sequence_topics(label_df):
    """
    Generates a large categorical palette of colors and topic names with colorcet to use in the plotly plots.
    Returns
    -------
    colors: dict
        A dictionary of cluster labels and hex codes to be used for consistent topic labeling throughout analyses.
    """
    num_categories = 12
    color_swatches = cc.glasbey_light[:num_categories]

    # generate a dictionary for plotly
    col_df = pd.DataFrame(color_swatches)
    col_df["Label"] = col_df.index - 1
    col_df.columns = ["color", "Label"]
    col_df["Label"] = col_df["Label"].astype(int)
    label_df = label_df
    label_df["Label"] = label_df["Label"].astype(int)
    col_labels = pd.merge(col_df, label_df, how="inner", on="Label")
    colors = dict(zip(col_labels.Topic, col_df.color))

    return colors


def plot_clusters(labeled_clustering_data, colors):
    """
    Make an interactive plot to visualize clusters.
    Parameters
    ----------
    labeled_clustering_data: Pandas dataframe
        a dataframe containing the article data and cluster labels
    colors: dict
        a dictionary of unqiue colors per cluster
    Returns
    -------
    fig: Plotly figure object
        A scatter plot of the clustered articles.
    """
    if labeled_clustering_data.shape[0] > 100000:
        labeled_clustering_data = labeled_clustering_data.sample(frac=0.33)
    else:
        labeled_clustering_data = labeled_clustering_data

    if labeled_clustering_data is None:
        return None
    else:
        # Wrap the titles
        labeled_clustering_data["article_title"] = labeled_clustering_data["article_title"].apply(
            lambda txt: "<br>".join(wrap(txt, width=50))
        )

        # Change the custom color sequence to a string (for this chart only)
        keys_values = colors.items()

        labeled_clustering_data.sort_values(by="cluster_label", ascending=True, inplace=True)

        # relabel the clusters as strings to allow "unclustered" label
        labeled_clustering_data_strings = labeled_clustering_data.copy()
        labeled_clustering_data_strings["cluster_label"] = labeled_clustering_data_strings[
            "cluster_label"
        ].astype(str)
        labeled_clustering_data_strings = labeled_clustering_data_strings.replace(
            "-1", "unclustered"
        )

        # plot the figure
        fig = px.scatter(
            labeled_clustering_data_strings,
            x="tsne_1",
            y="tsne_2",
            color="cluster_label",
            title=f"Plotting the clusters for each identified sub-topic for retracted articles",
            hover_name="article_title",
            hover_data={
                "full_source_title": True,
                "tsne_1": False,
                "tsne_2": False,
                "citations": True,
            },
            width=800,
            height=800,
            color_discrete_map=colors,
        )

        # Set the size of the plot and make symmetric
        x_range = (
            max(
                max(labeled_clustering_data["tsne_1"]),
                -min(labeled_clustering_data["tsne_1"]),
            )
            + 10
        )
        y_range = (
            max(
                max(labeled_clustering_data["tsne_2"]),
                -min(labeled_clustering_data["tsne_2"]),
            )
            + 10
        )

        range = max(x_range, y_range)

        fig.update_xaxes(range=(-range, range), autorange=False, showticklabels=False, title="")
        fig.update_yaxes(range=(-range, range), autorange=False, showticklabels=False, title="")

        # Set unclustered papers to lower opacity
        fig.for_each_trace(
            lambda trace: trace.update(opacity=0.1)
            if trace.name == "unclustered"
            else trace.update(opacity=1)
        )

        return fig


def plot_fig_label(
    labeled_clustering_data,
    label_df,
    colors,
    average_cites,
    average_change,
    start_year,
    end_year,
    type,
):
    """
    Plot a scatter graph of the cluster bibliometrics for a given set of articles.
    Parameters
    ----------
    labeled_clustering_data: Pandas dataframe
        a dataframe containing the article data and cluster labels
     colors: dict
        a dictionary of unqiue colors per cluster
    average_cites: int
        the calculated average cites value for a given set of articles
    average_change: int
        the calculated change in output for a given set of articles
    start_year: int
        the year from which the output change calculation begins
    end_year: int
     the year at which the output change calculation ends
    type: str
        the type of bibliometric data being plotted
    Returns
    -------
    fig: Plotly figure object
        A scatter plot of the clusters in a given set of articles, showing relative growth, output, and citation information.
    """

    orig_chart_data = labeled_clustering_data[labeled_clustering_data["cluster_label"] != -1]
    orig_chart_data_reduced = orig_chart_data[["cluster_label", "year_published", "Cluster_Count"]]
    # get labels and sizes for growth
    values = range(start_year, end_year - 1, 1)  # take full years only
    change = average_change.astype(int)  # load average growth
    growth_data = pd.DataFrame(
        orig_chart_data_reduced[orig_chart_data_reduced.year_published.isin(values)]
    ).sort_values(by=["cluster_label", "year_published"], ascending=[True, True])
    grouped = growth_data.groupby(["cluster_label"])  # group the data by cluster label
    growth_data["Change"] = grouped["Cluster_Count"].diff(
        1
    )  # count the difference between end and start years
    growth_data = growth_data.fillna(method="bfill")  # fill the data to the start year rows
    growth_data = growth_data.drop_duplicates(
        ["cluster_label", "year_published"], keep="first"
    )  # clear the duplicates
    growth_data = growth_data.drop_duplicates(
        ["cluster_label"], keep="last"
    )  # keep the end year rows with calcualtions only
    growth_data["% Change"] = (
        growth_data["Change"] / growth_data["Cluster_Count"]
    ) * 100  # calculate growth over the period
    growth_data["% Change"] = growth_data["% Change"].astype(int)
    growth_data["Cluster Change"] = "Growing"  # generic label growth
    growth_data["Size"] = 20
    growth_data.loc[
        growth_data["% Change"] > change, "Cluster Change"
    ] = "High Growth"  # tag fast growing clusters
    growth_data.loc[
        growth_data["% Change"] < 0, "Cluster Change"
    ] = "Declining"  # tag declining clusters

    chart_data = pd.merge(
        orig_chart_data, growth_data, how="left", on="cluster_label"
    )  # merge growth data back to main df

    chart_data_topic = pd.merge(
        chart_data,
        label_df,
        how="inner",
        left_on="cluster_label",
        right_on="Label",
    )
    # plot the figure
    fig = px.scatter(
        chart_data_topic,
        x="Papers per Cluster",
        y="Average Cites in Cluster",
        color="Topic",
        color_discrete_map=colors,
        size="Size",
        symbol="Cluster Change",
        title=f"Plotting of bibliometric data for each identified sub-topic: {type}",
        symbol_map={
            "Growing": "triangle-up",
            "High Growth": "cross",
            "Declining": "triangle-down",
        },
        height=800,
        hover_data={
            "Average Cites in Cluster": ":.2f",
            "Median Cites in Cluster": True,
            "Min Cites in Cluster": True,
            "Max Cites in Cluster": True,
        }
        # "Size": True}
    )

    # Add a line to guide the eye
    average_citations_per_paper = average_cites

    fig.update_layout(
        shapes=[
            dict(
                type="line",
                xref="x",
                x0=-5,
                x1=(max(chart_data["Papers per Cluster"]) * 1.5),
                yref="y",
                y0=average_citations_per_paper,
                y1=average_citations_per_paper,
                line=dict(width=4, dash="dash", color="Gray"),
            )
        ],
        annotations=[
            dict(
                x=max(chart_data["Papers per Cluster"]),
                y=average_citations_per_paper,
                xref="x",
                yref="y",
                text=f"Average citations per {type} paper: {average_citations_per_paper:0.2f}",
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-20,
            )
        ],
    )

    ## style scatter points to equal size
    fig.update_traces(
        marker=dict(line=dict(width=1, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    return fig


def plot_OA_papers(labeled_clustering_data, colors, average_OA, header):
    """
    Plot the % OA and #papers per cluster from the selected subject
    Parameters
    ----------
    labeled_clustering_data: Pandas dataframe
        a dataframe containing the article data and cluster labels
     colors: dict
        a dictionary of unqiue colors per cluster
    average_OA: int
        the calculated average percentage of OA publications for the subject category
    header: str
        noting article type used in plots
    Returns
    -------
    fig: Plotly figure object
        A scatter plot of the clusters in the subject category, showing relative growth, output, and citation information.
    """

    if labeled_clustering_data is None:
        return None
    else:
        # Get unique papers for average cites per paper
        labeled_clustering_data = labeled_clustering_data.copy().drop_duplicates(subset=["doi"])
        labeled_clustering_data_strings = labeled_clustering_data.copy()
        labeled_clustering_data_strings["cluster_label"] = labeled_clustering_data_strings[
            "cluster_label"
        ].astype(str)
        labeled_clustering_data_strings = labeled_clustering_data_strings.replace(
            "-1", "unclustered"
        )
        plot_data = labeled_clustering_data_strings[
            labeled_clustering_data_strings["cluster_label"] != "unclustered"
        ]

        # generate the required stats
        plot_data["Papers per Cluster"] = (
            plot_data.groupby(["cluster_label"])["doi"].transform("count").astype(int)
        )

        # convert online open tag to open access
        plot_data["open_access_binary"] = "No"
        plot_data.loc[
            plot_data["art_oa_status"].str.startswith("Open"),
            "open_access_binary",
        ] = "Open Access"
        plot_data.loc[
            plot_data["art_oa_status"].str.startswith("Online"),
            "open_access_binary",
        ] = "Open Access"

        plot_data["Open_Access_Boolean"] = np.where(
            (plot_data["open_access_binary"] == "Open Access"),
            True,
            False,
        )
        plot_data["Open_Access_Count"] = (
            plot_data.groupby(["cluster_label"])["Open_Access_Boolean"]
            .transform("sum")
            .astype(int)
        )
        plot_data["Percent Open Access"] = (
            (plot_data["Open_Access_Count"] / plot_data["Papers per Cluster"]) * 100
        ).astype(int)

        # cut the unassigned papers and create custom label order that matches clustering plot
        chart_data = plot_data[plot_data.cluster_label != -1]

        # colors = get_color_sequence()#(labeled_clustering_data)
        # colors.pop(0) # Remove the first color swatch in the list so it matches

        # plot the figure
        fig = px.scatter(
            chart_data,
            x="Papers per Cluster",
            y="Percent Open Access",
            color="Topic",
            title=f"Percentage of OA publications for {header} papers"
            + "<br>"
            + "as a proportion of all publications",
            color_discrete_map=colors,
            height=800,
            # error_y="Avg Cites std per Cluster",
            # error_y_minus=None,
            hover_data={"Papers per Cluster": True, "Percent Open Access": True},
        )

        # Add a line to guide the eye

        average_OA = average_OA

        fig.update_layout(
            shapes=[
                dict(
                    type="line",
                    xref="x",
                    x0=-5,
                    x1=(max(chart_data["Papers per Cluster"]) * 1.5),
                    yref="y",
                    y0=average_OA,
                    y1=average_OA,
                    line=dict(width=4, dash="dash", color="Gray"),
                )
            ],
            annotations=[
                dict(
                    x=max(chart_data["Papers per Cluster"]),
                    y=average_OA,
                    xref="x",
                    yref="y",
                    text=f"Category average percent OA: {average_OA}",
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-20,
                )
            ],
        )

        ## style scatter points to equal size
        fig.update_traces(
            marker=dict(size=20, line=dict(width=1, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )

        # lock y-axis to show 100%
        fig.update_yaxes(range=[-5, 100])

        return fig


def plot_cm(y_true, y_pred, title):
    """'
    input y_true-Ground Truth Labels
          y_pred-Predicted Value of Model
          title-What Title to give to the confusion matrix

    Draws a Confusion Matrix for better understanding of how the model is working

    return None

    """

    figsize = (10, 10)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%.1f%%\n%d" % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt="", ax=ax)


def roc_curve_plot(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic for Retraction Prediction Model")
    plt.legend(loc="lower right")
    plt.show()
