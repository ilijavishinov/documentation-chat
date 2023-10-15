import time

import pandas as pd
import os
from openai.embeddings_utils import get_embedding
import helper as h
import collections
import tqdm
import openai
import pinecone

DIM_ADA = 1536

def setup_openai_api_key(api_key):
    openai.api_key = api_key


def console_print(string):
    """
    Printing recognizable format in console to keep track of script progress
    """
    print(f'\n===== {string} =====')


def pinecone_init(idx_name,
                  idx_dimension,
                  api_key):
    """
    Init a pinecone vector database, or connect to a current one if it exists
    """

    pinecone.init(
        api_key = api_key,
        # environment = 'us-east1-gcp'  # find in console next to api key
        environment = 'gcp-starter'  # find in console next to api key
        
    )

    # if index not created on server, create it
    if idx_name not in pinecone.list_indexes():
        pinecone.create_index(name = idx_name, dimension = idx_dimension)
        console_print('Initialized Pinecone index')
    else:
        console_print('Connected to a existing Pinecone index')

    return pinecone.Index(index_name = idx_name)


def pinecone_connect(idx_name,
                     api_key):
    """
    Connect to a Pinecone vector database
    """

    pinecone.init(
        api_key = api_key,
        # environment = 'us-east1-gcp'  # find in console next to api key
        environment = 'gcp-starter'  # find in console next to api key
    )

    # if index not created on server, create it
    if idx_name not in pinecone.list_indexes():
        raise NameError(f"The pinecone index: {idx_name} does not exist")

    console_print(f'Successfuly connected to {idx_name} Pinecone index')

    return pinecone.Index(index_name=idx_name)


def df_from_documentation(docs_dir):
    """
    Parse the documentation to a dataframe that will work with Pinecone
    """

    # init columns
    df = pd.DataFrame(columns = ['title', 'content'])

    for docs_subdir in os.listdir(docs_dir):
        # read data into df
        for file in os.listdir(os.path.join(docs_dir, docs_subdir)):
            with open(os.path.join(docs_dir, docs_subdir, file), 'r', encoding = 'utf-8') as f:
                content = f.read()
            
            # write df row
            df.loc[len(df)] = dict(title = f"{docs_subdir}/{file.replace('.md','')}",
                                   content = content)
    
        # define text that will be encoded
        df["text_to_encode"] = ("Title: " + df['title'].str.strip() + "; Content: " + df['content'].str.strip())
        
    # preprocess df so it is suitable for pinecone upsert
    df.index = df.index.map(str)
    df.index.name = 'vector_id'
    
    return df


def get_embedding_dimensionality(model_name):
    """
    Returns the dimensionality of the embedding model depending on its name
    """
    if model_name == 'text-embedding-ada-002':
        return DIM_ADA
    # elif model_name == 'openai_davinci':
    #     return DIM_DAVINCI
    # elif model_name == 'sbert':
    #     return DIM_SBERT
    else:
        raise NameError(f'The model_name: {model_name} you passed is not supported')


def get_vector_metadata_from_dataframe_row(df_row):
    """
    Return vector metadata
    """
    vector_metadata = {
        'source': df_row['title']
    }
    return vector_metadata

    
def get_vectors_to_upload_to_pinecone(df_chunk,
                                      model_name):
    """
    Return list of tuples (vector_id, vector_values, vector_metadata).
    """
    
    if model_name in ['text-embedding-ada-002']:
        vector_values = df_chunk['text_to_encode'].apply(
            lambda x: get_embedding(x, engine=model_name)
        ).to_list()
    else:
        raise NameError(f'The model_name: {model_name} you passed is not supported')
    
    # create vector ids and metadata
    vector_ids = df_chunk.index.tolist()
    vector_metadata = df_chunk.apply(get_vector_metadata_from_dataframe_row, axis=1).tolist()
    
    return list(zip(vector_ids, vector_values, vector_metadata))


def upload_df_to_pinecone_in_chunks(df,
                                    pinecone_index,
                                    model_name,
                                    chunk_size=100,
                                    upsert_size=100):
    """
    Encode dataframe column `text_to_encode` to dense vector and upsert to Pinecone
    """
    
    tqdm_kwargs = h.get_tqdm_kwargs(df, chunk_size)
    async_results = collections.defaultdict(list)
    
    for df_chunk in tqdm.tqdm(h.chunks(df, chunk_size), **tqdm_kwargs):
        time.sleep(15)
        vectors = get_vectors_to_upload_to_pinecone(df_chunk = df_chunk,
                                                    model_name = model_name)
    
        # upload to Pinecone in batches of `upsert_size`
        for vectors_chunk in h.chunks(vectors, upsert_size):
            print('uploading vector chunk')
            start_index_chunk = df_chunk.index[0]
            async_result = pinecone_index.upsert(vectors_chunk, async_req=True)
            async_results[start_index_chunk].append(async_result)

        # wait for results
        _ = [async_result.get() for async_result in async_results[start_index_chunk]]
        is_all_successful = all(map(lambda x: x.successful(), async_results[start_index_chunk]))

        # report chunk upload status
        print(f'All upserts in chunk successful with index starting with {start_index_chunk:>7}: '
              f'{is_all_successful}. Vectors uploaded: {len(vectors):>3}.')
    
    return async_results
    
    
def query_response(index,
                   query,
                   model_name,
                   top_n = 5):
    """
    Get the most similar matches from a Pinecone database
    """
    
    query_vector_emb = get_embedding(query, engine = model_name)
    response = index.query([query_vector_emb], top_k = top_n, include_metadata = True)
    
    # print a well-formated top-n matches response
    print('\n************************')
    print(f'Question: {query}\n\n')
    print(f'The best {top_n} matches to the question are the following:\n')
    for i in range(top_n):
        match = response._data_store['matches'][i]
        print(f"{i + 1}. Score = {round(match['score'] * 100, 2)}% \tDocumentation file: {match['metadata']['source']}")
    print('************************\n')
    
    return response
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    