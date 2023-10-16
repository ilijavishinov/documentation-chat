import pandas as pd
import os
from transformers import DistilBertTokenizer, DistilBertModel


def df_from_documentation_pinecone(docs_dir):
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
            df.loc[len(df)] = dict(title = f"{docs_subdir}/{file.replace('.md', '')}",
                                   content = content)
        
        # define text that will be encoded
        df["text_to_encode"] = ("Title: " + df['title'].str.strip() + "; Content: " + df['content'].str.strip())
    
    # preprocess df so it is suitable for pinecone upsert
    df.index = df.index.map(str)
    df.index.name = 'vector_id'
    
    return df


def df_from_documentation_distilbert(docs_dir,
                                     model_name):
    """
    """
    
    # init columns
    df = pd.DataFrame(columns = ['title', 'content'])
    
    for docs_subdir in os.listdir(docs_dir):
        # read data into df
        for file in tqdm.tqdm(os.listdir(os.path.join(docs_dir, docs_subdir))[:1], desc = 'Loading chunks of docs into df'):
            with open(os.path.join(docs_dir, docs_subdir, file), 'r', encoding = 'utf-8') as f:
                content = f.read()
                content = markdown_to_text(content).lower()
                
                max_tokens = 128
                curr_pos = 0
                
                while curr_pos < len(content) - max_tokens:
                    curr_len = 128
                    final_len = None
                    num_tokens = 0
                    
                    while num_tokens < max_tokens - 2:
                        # print(curr_pos, curr_len)
                        cut_content = content[curr_pos:curr_pos + curr_len]
                        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
                        num_tokens = tokenizer(cut_content, padding = True, return_tensors = "pt")['input_ids'].shape[1]
                        if num_tokens < max_tokens - 2:
                            final_len = curr_len
                        curr_len += int(max_tokens * 2)
                        
                        if curr_pos + curr_len > len(content):
                            break
                    
                    df.loc[len(df)] = dict(title = f"{docs_subdir}/{file.replace('.md', '')}",
                                           content = content[curr_pos:curr_pos + final_len])
                    # print(content[curr_pos:curr_pos+final_len])
                    
                    if curr_pos + curr_len > len(content):
                        # print(curr_pos, curr_len, len(content))
                        break
                    
                    curr_pos += int(final_len * 2)
    
    df['embedding'] = df['content'].apply(lambda x: sentence_to_vector(x, model_name = model_name))
    df.index = df.index.map(str)
    df.index.name = 'vector_id'
    df['id'] = df.index
    
    return df









