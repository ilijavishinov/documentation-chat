import torch
import chromadb
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils_dir.ingest_data import sentence_to_vector

def create_chromadb_collection(path,
                               name):
    if not path:
        chroma_client = chromadb.Client()
    else:
        chroma_client = chromadb.PersistentClient(path = path)
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
                       documents,
                       model_name):
    BERT_MODEL_NAME = model_name
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
            first_answer = '. '.join(list(map(lambda x: x.strip().capitalize(), answer.split('.'))))
            first_answer = first_answer.strip()
            if first_answer[-1] != '.':
                first_answer += '.'
        
        all_answers_string += f'{i}. {answer}\n'
        i += 1
    
    return first_answer
