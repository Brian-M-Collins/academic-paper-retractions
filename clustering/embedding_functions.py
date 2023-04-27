# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel

# %%
# globals
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("allenai/specter").to(device)
tqdm.pandas()
# %%


def prepare_data(dataframe: str):
    """
    A function to check for the presence of correct columns and strip excess columns from a source dataframe.
    Parameters
    ---------
    dataframe (str): the source dataframe
    Returns
    ---------
    output(dataframe): a dataframe with columns doi, title, abstract
    """
    df = dataframe
    df = df.rename(columns={"article_title": "title", "concat_abstract": "abstract"})
    # check the data has the necessary columns to read and stop if it doesn't
    assert (
        "doi" in df.columns
    ), "doi column not found, please check your dataframe and column name case."
    assert (
        "title" in df.columns
    ), "title column not found, please check your dataframe and column name case."
    assert (
        "abstract" in df.columns
    ), "abstract column not found, please check your dataframe and column name case."

    # cut down to the three required columns.
    output = pd.DataFrame(df[["doi", "title", "abstract"]])
    output["title"] = output["title"].str.strip()
    output["abstract"] = output["abstract"].str.strip()

    return output


# %%
def embed_df_abstracts(abstracts, batch_size=4):
    """
    Embed a dataframe of abstracts in batches.
    Parameters
    ----------
    abstracts: pandas.DataFrame
        A dataframe of abstracts to be embedded.
    batch_size: int (default: 4)
        How many abstracts to load onto the GPU at once, reduce liklihood of CUDA memory error

    Returns
    -------
    embedded_df: pandas.DataFrame
        A copy of the abstracts dataframe with a new column with embeddings.
    """
    # set up embedding process
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    embedded_df = pd.DataFrame()

    # pass chunks of df to the GPU for processing
    for chunk in tqdm(np.split(abstracts, np.arange(batch_size, len(abstracts), batch_size))):

        chunk_dict = chunk.to_dict("records")
        title_abs = [
            (d.get("title") or "") + tokenizer.sep_token + (d.get("abstract") or "")
            for d in chunk_dict
        ]

        inputs = tokenizer(
            title_abs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)

        results = model(**inputs)

        embedding_list = results.last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()
        embedding = [np.asarray(i) for i in embedding_list]

        chunk["embeddings"] = embedding

        embedded_df = pd.concat([embedded_df, chunk], axis=0)

    torch.cuda.empty_cache()

    df2 = pd.DataFrame(embedded_df).reset_index(drop=True)

    output_dataframe = pd.DataFrame(df2[["doi", "title", "embeddings"]])

    return output_dataframe


def embed_df(abstracts, batch_size=4):
    """
    Embed a dataframe of abstracts in batches.
    Parameters
    ----------
    abstracts: pandas.DataFrame
        A dataframe of abstracts to be embedded.
    batch_size: int (default: 4)
        How many abstracts to load onto the GPU at once, reduce liklihood of CUDA memory error
    Returns
    -------
    embedded_df: pandas.DataFrame
        A copy of the abstracts dataframe with a new column with embeddings.
    """
    # set up embedding process
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    embedded_df = pd.DataFrame()

    # pass chunks of df to the GPU for processing
    for chunk in tqdm(np.split(abstracts, np.arange(batch_size, len(abstracts), batch_size))):

        chunk_dict = chunk.to_dict("records")
        titles = [(d.get("title") or "") for d in chunk_dict]

        inputs = tokenizer(
            titles,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)

        results = model(**inputs)

        embedding_list = results.last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()
        embedding = [np.asarray(i) for i in embedding_list]

        chunk["embeddings"] = embedding

        embedded_df = pd.concat([embedded_df, chunk], axis=0)

    torch.cuda.empty_cache()

    df2 = pd.DataFrame(embedded_df).reset_index(drop=True)

    output_dataframe = pd.DataFrame(df2[["doi", "title", "embeddings"]])

    return output_dataframe
