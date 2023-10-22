from langchain.docstore.document import Document
import torch
import os
import utils_dir.text_processing as text_processing
from utils_dir_backup.ingest_data import sentence_to_vector
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class QaAgent:
    
    qa_tokenizer = None
    qa_model = None
    
    def __init__(self,
                 qa_model_name = 'roberta'):
        
        self.qa_model_name = qa_model_name
        self.get_qa_object()
    
    def get_qa_object(self):
        """
        """
        
        if self.qa_model_name.startswith('qa_albert'):
            self.qa_tokenizer = AutoTokenizer.from_pretrained('Akari/albert-base-v2-finetuned-squad')
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained('Akari/albert-base-v2-finetuned-squad')
        elif self.qa_model_name.startswith('qa_bert'):
            self.qa_tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
        elif self.qa_model_name.startswith('qa_roberta'):
            self.qa_tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
        
        if not self.qa_model:
            raise NameError("The model_name for llm that you entered is not supported")
        
    @staticmethod
    def remove_bert_tokens(text: str) -> str:
        for separator_token in ["[CLS]", "[SEP]", "[UNK]", "<s>", "</s>", "[]"]:
            text = text.replace(separator_token, "")
        return text
    
    def qa_model_answer(self,
                        query,
                        context):
        inputs = self.qa_tokenizer(query.lower(), context, add_special_tokens = True, return_tensors = "pt")
        input_ids = inputs["input_ids"].tolist()[0]
        
        inputs_dict = dict(**inputs)
        model_output = self.qa_model(**inputs_dict)
        
        answer_start_scores, answer_end_scores = model_output['start_logits'], model_output['end_logits']
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        
        answer = self.qa_tokenizer.convert_tokens_to_string(
            self.qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )
        answer = self.remove_bert_tokens(answer)
        return answer
    
    @staticmethod
    def relevant_docs_ordered_by_similarity(query,
                                            db,
                                            k,
                                            threshold = 0.5):
        """

        """
        relevant_docs_tuples = db.similarity_search_with_relevance_scores(query, k = k)
        
        # sort by relevance score
        relevant_docs_tuples.sort(key = lambda a: a[1], reverse = True)
        
        # take only relevant docs with cosine similarity > 0.5
        relevant_docs = [pair[0] for pair in relevant_docs_tuples if pair[1] >= threshold]
        similarity_scores = [pair[1] for pair in relevant_docs_tuples if pair[1] >= threshold]
        
        return relevant_docs, similarity_scores
    
    def qa_response(self,
                    query,
                    db):
        """

        """
        
        query = query.lower()
        
        self.get_qa_object()
        
        result = None
        relevant_docs = None
        
        current_k = 5
        k_increase = 30
    
        
        while not result:
            # if result not found in 50 retrieved docs, do not provide one
            if current_k > 65:
                result = dict(query = query,
                              result = 'Could not answer question',
                              source_documents = [])
                break
            
            relevant_docs, similarity_scores = self.relevant_docs_ordered_by_similarity(query, db, current_k)
            
            # take last retrieved documents
            for doc in relevant_docs[current_k - k_increase:]:
                context = doc.page_content
                
                try:
                    answer = self.qa_model_answer(query = query,
                                                  context = context)
                except Exception as e:
                    print(e)
                    continue
                
                # iterate retrieved docs while sufficient answer
                if not result and len(answer) > 5:
                    result = dict(query = query,
                                  result = text_processing.format_answer(answer),
                                  source_documents = [doc])
                    break
            
            current_k += k_increase
            
            print('while', not result)
        
        return result, relevant_docs
    