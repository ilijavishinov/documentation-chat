import torch
import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModel, RobertaForCausalLM, AutoModelForQuestionAnswering
from langchain import HuggingFacePipeline
from langchain.llms import HuggingFacePipeline
from langchain.llms import LlamaCpp, GPT4All
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings, GPT4AllEmbeddings
from pathlib import Path
from langchain.document_loaders import (
    TextLoader, UnstructuredMarkdownLoader
)
import os
import utils_dir.text_processing as text_processing
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, MarkdownHeaderTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"


def console_print(arg, desc = None):
    print()
    print('_________________________ ********************** ______________________')
    print(desc)
    print(arg)
    print('_________________________ ********************** ______________________')
    print()


class DocumentationAgent:
    
    documents = []
    texts = []
    docs_dir = None
    db = None
    embedding_tokenizer = None
    embedding_model = None
    qa_tokenizer = None
    qa_model = None
    llm = None
    
    def __init__(self,
                 db_dir = None,
                 embedding_model_name = 'distilbert',
                 llm_model_name = None,
                 qa_model_name = 'roberta'):
        
        self.db_dir = db_dir
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.qa_model_name = qa_model_name
        
        self.rag_prompt_template = """Use only the following pieces of context to answer the question at the end. \
        If the context does not contain the answer, say that the documentation does not contain the answer.

        {context}

        Question: {question}
        Answer:"""
        self.llm_rag_prompt = PromptTemplate(
            template = self.rag_prompt_template, input_variables = ["context", "question"]
        )
    
    def read_documents(self,
                       docs_dir):
        """
        """
        
        glob = Path(f"{docs_dir}").glob
        ps = list(glob("**/*.md"))
        documents = list()
        for p in tqdm.tqdm(ps, "Loading documents"):
            file_extension = os.path.splitext(p)[1]
            if file_extension != '.md': continue
            document = UnstructuredMarkdownLoader(p, encoding = "utf-8").load()[0]
            document.page_content = text_processing.markdown_to_lower_text(document.page_content)
            document.metadata["source"] = document.metadata['source'].__str__()
            documents.append(document)
        self.documents = documents
    
    def split_documents(self,
                        chunk_size,
                        text_splitter_name = 'recursive'):
        """

        """

        text_splitter = None
        
        if type(chunk_size) is int:
            chunk_size = [chunk_size]
        
        chunk_sizes = chunk_size
        
        for chunk_size in chunk_sizes:
            chunk_overlap = chunk_size // 3
            
            if text_splitter_name == 'recursive':
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = chunk_size,
                    chunk_overlap = chunk_overlap,
                    separator = ' '
                )
            elif text_splitter_name == 'character':
                text_splitter = CharacterTextSplitter(
                    chunk_size = chunk_size,
                    chunk_overlap = chunk_overlap,
                    separator = ' '
                )
            elif text_splitter_name == 'markdown':
                text_splitter = MarkdownTextSplitter()
            elif text_splitter_name == 'tokens':
                text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
                    self.embedding_tokenizer,
                    chunk_size = chunk_size,
                    chunk_overlap = chunk_overlap,
                    separator = ' '
                )
                # text_splitter = SentenceTransformersTokenTextSplitter(
                #     model_name = qa_model_name
                # )
        
            if not text_splitter:
                raise NameError("The text_splitter_name that you entered is not supported")
            
            texts = text_splitter.split_documents(self.documents)
            self.texts.extend(texts)
    
    def get_embeddings_object(self):
        """

        """
        
        if self.embedding_model_name == 'openai':
            self.embedding_model = OpenAIEmbeddings(
                model = 'text-embedding-ada-002'
            )
            
        elif self.embedding_model_name == 'llamacpp':
            self.embedding_model = LlamaCppEmbeddings(
                model_path = r'C:\Users\ilija\llama.cpp\models\7B\ggml-model-q4_0.gguf',
                verbose = True,
                n_ctx = 1024,
                n_gpu_layers = 40,
                n_batch = 512
            )
            
        elif self.embedding_model_name == 'llamacpppython':
            self.embedding_model = LlamaCppEmbeddings(
                model_path = r'C:\Users\ilija\llama.cpp\models\7B\ggml-model-q4_0.gguf',
                verbose = True,
                n_ctx = 1024,
                n_gpu_layers = 40,
                n_batch = 512
            )
            
        elif self.embedding_model_name == 'sbert':
            self.embedding_model = GPT4AllEmbeddings(
                model_path = r"ggml-all-MiniLM-L6-v2-f16.bin"
            )
            
        elif self.embedding_model_name == 'ggml-falcon':
            print("Using falcon model")
            self.embedding_model = GPT4AllEmbeddings(
                model = r"D:\python_projects\loka_final\models\ggml-model-gpt4all-falcon-q4_0.bin"
            )
            
        elif self.embedding_model_name.startswith('flan'):
            self.embedding_model = GPT4AllEmbeddings(
                model_path = r"ggml-all-MiniLM-L6-v2-f16.bin"
            )
            
        elif self.embedding_model_name.startswith('distilbert'):
            model_name = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
            self.embedding_model = HuggingFaceEmbeddings(
                model_name = model_name,
                model_kwargs = {'device': 'cuda:0'},
                # encode_kwargs = {'normalize_embeddings': False}`
            )
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        elif self.embedding_model_name.startswith('bert'):
            model_name = "sentence-transformers/bert-base-nli-stsb-mean-tokens",
            self.embedding_model = HuggingFaceEmbeddings(
                model_name = model_name,
                model_kwargs = {'device': 'cuda:0'},
                # encode_kwargs = {'normalize_embeddings': False}`
            )
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)

            
        elif self.embedding_model_name.startswith('roberta'):
            # model_name = "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
            model_name = "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"
            self.embedding_model = HuggingFaceEmbeddings(
                model_name = model_name,
                model_kwargs = {'device': 'cuda:0'},
                # encode_kwargs = {'normalize_embeddings': False}`
            )
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)

        
        if not self.embedding_model:
            raise NameError("The model_name for embeddings that you entered is not supported")
    
    
    def get_llm_object(self):
        """
        """
        
        console_print(f'Getting {self.llm_model_name}')
        if self.llm_model_name == 'openai':
            self.llm = ChatOpenAI(model_name = "gpt-3.5-turbo")
        elif self.llm_model_name == 'llamacpp':
            self.llm = LlamaCpp(
                model_path = r'C:\Users\ilija\llama.cpp\models\7B\ggml-model-q4_0.gguf',
                verbose = True,
                n_ctx = 1024,
                n_threads = 8,
                n_gpu_layers = 40,
                n_batch = 512)
        elif self.llm_model_name == 'gpt4all':
            self.llm = GPT4All(
                model = './models/ggml-gpt4all-j-v1.3-groovy.bin',
            )
            # verbose = True, n_ctx = 1024, n_gpu_layers = 1, n_batch = 4)
        elif self.llm_model_name == 'ggml-falcon':
            self.llm = GPT4All(model = r"D:\Downloads\ggml-model-gpt4all-falcon-q4_0.bin")
            # verbose = True, n_ctx = 1024, n_gpu_layers = 1, n_batch = 4)
        elif self.llm_model_name.startswith('flan'):
            tokenizer = AutoTokenizer.from_pretrained(f"google/{self.llm_model_name}")
            model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{self.llm_model_name}")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            self.llm = HuggingFacePipeline(
                pipeline = pipe,
                model_kwargs = {"temperature": 0, "max_length": 512},
            )
        elif self.llm_model_name.startswith('distilbert'):
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
            model = AutoModelForSeq2SeqLM.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            self.llm = HuggingFacePipeline(
                pipeline = pipe,
            )
        elif self.llm_model_name.startswith('bert'):
            tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/bert-base-nli-stsb-mean-tokens")
            model = AutoModelForSeq2SeqLM.from_pretrained("sentence-transformers/bert-base-nli-stsb-mean-tokens")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            self.llm = HuggingFacePipeline(
                pipeline = pipe,
            )
        elif self.llm_model_name.startswith('roberta'):
            tokenizer = AutoTokenizer.from_pretrained(f"deepset/roberta-base-squad2")
            model = RobertaForCausalLM.from_pretrained("deepset/roberta-base-squad2")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            self.llm = HuggingFacePipeline(
                pipeline = pipe,
            )
        
        if not self.llm:
            raise NameError("The model_name for llm that you entered is not supported")
    
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
    
    def load_documentation_folder(self,
                                  docs_dir,
                                  text_splitter_name,
                                  chunk_size = None,
                                  similarity_metric_name = 'cosine'):
        """
        """
        
        # load the embedding model
        self.get_embeddings_object()
        
        # assertions
        if text_splitter_name == 'tokens' and self.embedding_model_name not in ['distilbert', 'roberta', 'bert']:
            raise NameError(f"tokens chunking implementation for {self.embedding_model_name} is not supported")
        
        # define the db folder name based on chunking and embedding parameters
        dir_suffix = f"{self.embedding_model_name}_{similarity_metric_name}_{text_splitter_name}"
        if chunk_size:
            dir_suffix += f'_{"-".join([str(i) for i in chunk_size])}'
        persist_directory = os.path.join(self.db_dir, f'db_{os.path.basename(os.path.normpath(docs_dir))}_{dir_suffix}')
        
        # load chroma db, or create if it does not exist
        if not os.path.exists(persist_directory):
            self.read_documents(docs_dir)
            self.split_documents(text_splitter_name = text_splitter_name, chunk_size = chunk_size)
            chroma = Chroma.from_documents(self.texts,
                                           self.embedding_model,
                                           persist_directory = persist_directory,
                                           collection_metadata={"hnsw:space": similarity_metric_name})
        else:
            chroma = Chroma(persist_directory = persist_directory,
                            embedding_function = self.embedding_model,
                            collection_metadata={"hnsw:space": similarity_metric_name})
        
        self.db = chroma
    
    def llm_rag(self,
                query):
        """

        """
        query = query.lower()

        self.get_llm_object()
        
        result = None
        answer = 'not contain the answer'
        current_k = 0
        while 'not contain the answer' in answer and current_k <= 1:
            current_k += 1
            qa = RetrievalQA.from_chain_type(llm = self.llm,
                                             chain_type = "stuff",
                                             retriever = self.db.as_retriever(search_kwargs = {'k': current_k}),
                                             chain_type_kwargs = {"prompt": self.llm_rag_prompt},
                                             return_source_documents = True
                                             )
            result = qa({"query": query})
            answer = result['result']
        
        # console_print(result, 'result')
        relevant_docs, similarity_scores = self.relevant_docs_ordered_by_similarity(query, current_k)
        # console_print(relevant_docs, 'relevant_docs')
        return result, relevant_docs
    
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
    
    def relevant_docs_ordered_by_similarity(self,
                                            query,
                                            k,
                                            threshold = 0.5):
        """
        
        """
        relevant_docs_tuples = self.db.similarity_search_with_relevance_scores(query, k = k)
        
        # sort by relevance score
        relevant_docs_tuples.sort(key = lambda a: a[1], reverse = True)
        
        # take only relevant docs with cosine similarity > 0.5
        relevant_docs = [pair[0] for pair in relevant_docs_tuples if pair[1] >= threshold]
        similarity_scores = [pair[1] for pair in relevant_docs_tuples if pair[1] >= threshold]
        
        return relevant_docs, similarity_scores
        
    
    def qa_response(self,
                    query):
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
            
            relevant_docs, similarity_scores = self.relevant_docs_ordered_by_similarity(query, current_k)
            
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
    
    
    @staticmethod
    def check_langchain_gpu_usage():
        """
        """
        import torch
        if torch.cuda.is_available():
            print("GPU is available")
        else:
            print("GPU is not available")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Langchain is using device:", device)
        
        
        
        