import chromadb
import torch
import pandas as pd
import os
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import tqdm
from bs4 import BeautifulSoup
from markdown import markdown
import re
import random
import numpy as np

def seed_everything(random_seed):
    """
    Set random seeds for reproducibility
    """
    # set random seeds
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text


def df_from_documentation(docs_dir,
                          model_name):
    """
    Parse the documentation to a dataframe that will work with Pinecone
    """
    
    # init columns
    df = pd.DataFrame(columns = ['title', 'content'])
    
    for docs_subdir in os.listdir(docs_dir):
        # read data into df
        for file in tqdm.tqdm(os.listdir(os.path.join(docs_dir, docs_subdir))[:5], desc = 'Loading chunks of docs into df'):
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


def mean_pooling(model_output,
                 attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def sentence_to_vector(raw_inputs,
                       model_name):
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)
    inputs_tokens = tokenizer(raw_inputs, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs_tokens)

    sentence_embeddings = mean_pooling(outputs, inputs_tokens['attention_mask'])
    return sentence_embeddings


def create_chromadb_collection(path,
                               name):
    chroma_client = chromadb.Client()
    # chroma_client = chromadb.PersistentClient(path = path)
    collection = chroma_client.create_collection(name = name)
    return collection


def add_df_rows_to_collection(collection,
                              df):
    
    for row_idx, row in df.iterrows():
        collection.add(
            embeddings = row['embedding'].tolist()[0],
            metadatas = {"source": row['title']},
            documents = row['content'],
            ids = row['id'],
        )


def get_top_docs(question,
                 collection,
                 model_name,
                 n_results):
    
    question_emb = sentence_to_vector(question, model_name).tolist()[0]
    
    results = collection.query(
        query_embeddings = question_emb,
        n_results = n_results,
    )
    return results


def remove_bert_tokens(text: str) -> str:
    for separator_token in ["[CLS]", "[SEP]", "[UNK]", "<s>", "</s>", "[]"]:
        text = text.replace(separator_token, "")
    
    return text


def answer_q_from_docs(question,
                       documents):
    
    BERT_MODEL_NAME = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(BERT_MODEL_NAME)
    
    all_answers_string = ''
    i = 1
    
    first_answer = None
    
    for doc in documents:
        content = doc

        inputs = tokenizer(question.lower(), content, add_special_tokens = True, return_tensors = "pt")
        input_ids = inputs["input_ids"].tolist()[0]

        inputs_dict = dict(**inputs)
        model_output = model(**inputs_dict)

        answer_start_scores, answer_end_scores = model_output['start_logits'], model_output['end_logits']
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )
        answer = remove_bert_tokens(answer)
        
        # TODO: will it select the right doc if less than five
        if not first_answer and len(answer) > 5:
            first_answer ='. '.join(list(map(lambda x: x.strip().capitalize(), answer.split('.'))))
            first_answer = first_answer.strip()
            if first_answer[-1] !='.':
                first_answer += '.'

        all_answers_string += f'{i}. {answer}\n'
        i += 1
        
    return first_answer
    
    
        








