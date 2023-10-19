from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModel, RobertaForCausalLM
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
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"


def console_print(arg, desc):
    print()
    print('_________________________ ********************** ______________________')
    print(desc)
    print(arg)
    print('_________________________ ********************** ______________________')
    print()


class DocumentationAgent:
    
    documents = None
    texts = None
    db = None
    docs_dir = None
    
    def __init__(self,
                 # docs_dir = None,
                 db_dir = None,
                 embedding_model_name = 'distilbert',
                 llm_model_name = None,
                 qa_model_name = 'roberta'):
        
        # self.docs_dir = docs_dir
        self.db_dir = db_dir
        # self.persist_dir = os.path.join(db_dir, f'/db_{os.path.basename(os.path.normpath(docs_dir))}_{embedding_model_name}')
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
        for p in ps:
            file_extension = os.path.splitext(p)[1]
            if file_extension != '.md': continue
            document = UnstructuredMarkdownLoader(p, encoding = "utf-8").load()[0]
            document.page_content = text_processing.markdown_to_text(document.page_content)
            document.metadata["source"] = document.metadata['source'].__str__()
            documents.append(document)
        self.documents = documents
    
    def split_documents(self,
                        split_type = 1,
                        chunk_size = 1000,
                        chunk_overlap = 200):
        """

        """
        # type = 'markdown'
        text_splitter = None
        
        if split_type == 1:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                                           chunk_overlap = chunk_overlap)
        elif split_type == 2:
            text_splitter = MarkdownTextSplitter()
        
        texts = text_splitter.split_documents(self.documents)
        self.texts = texts
    
    def get_embeddings_object(self):
        """

        """
        
        model_name = self.embedding_model_name
        embeddings_object = None
        
        if model_name == 'openai':
            embeddings_object = OpenAIEmbeddings(model = 'text-embedding-ada-002')
        elif model_name == 'llamacpp':
            # embeddings_object = LlamaCppEmbeddings(model_path = r'C:\Users\ilija\models\llama.cpp\models\7B_noblas\ggml-model-q4_0.gguf',
            embeddings_object = LlamaCppEmbeddings(model_path = r'C:\Users\ilija\llama.cpp\models\7B\ggml-model-q4_0.gguf',
                                                   # embeddings_object = LlamaCppEmbeddings(model_path = r'D:\python_projects\loka_final\models\llama-2-7b.Q4_0.gguf',
                                                   verbose = True,
                                                   n_ctx = 1024,
                                                   n_gpu_layers = 40,
                                                   n_batch = 512)
        elif model_name == 'llamacpppython':
            # embeddings_object = LlamaCppEmbeddings(model_path = r'C:\Users\ilija\models\llama.cpp\models\7B_noblas\ggml-model-q4_0.gguf',
            embeddings_object = LlamaCppEmbeddings(model_path = r'C:\Users\ilija\llama.cpp\models\7B\ggml-model-q4_0.gguf',
                                                   # embeddings_object = LlamaCppEmbeddings(model_path = r'D:\python_projects\loka_final\models\llama-2-7b.Q4_0.gguf',
                                                   verbose = True,
                                                   n_ctx = 1024,
                                                   n_gpu_layers = 40,
                                                   n_batch = 512)
        elif model_name == 'sbert':
            embeddings_object = GPT4AllEmbeddings(model_path = r"ggml-all-MiniLM-L6-v2-f16.bin")
        elif model_name == 'ggml-falcon':
            print("Using falcon model")
            embeddings_object = GPT4AllEmbeddings(model = r"D:\python_projects\loka_final\models\ggml-model-gpt4all-falcon-q4_0.bin")
            # verbose = True, n_ctx = 1024, n_gpu_layers = 1, n_batch = 4)
        elif model_name.startswith('flan'):
            embeddings_object = GPT4AllEmbeddings(model_path = r"ggml-all-MiniLM-L6-v2-f16.bin")
        elif model_name.startswith('distilbert'):
            embeddings_object = HuggingFaceEmbeddings(
                model_name = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
                model_kwargs = {'device': 'cpu'},
                encode_kwargs = {'normalize_embeddings': False}
            )
        elif model_name.startswith('bert'):
            embeddings_object = HuggingFaceEmbeddings(
                model_name = "sentence-transformers/bert-base-nli-stsb-mean-tokens",
                model_kwargs = {'device': 'cpu'},
                encode_kwargs = {'normalize_embeddings': False}
            )
        elif model_name.startswith('roberta'):
            embeddings_object = HuggingFaceEmbeddings(
                model_name = "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
                model_kwargs = {'device': 'cpu'},
                encode_kwargs = {'normalize_embeddings': False}
            )
        
        if not embeddings_object:
            raise NameError("The model_name for embeddings that you entered is not supported")
        
        self.embedding_object = embeddings_object
        return embeddings_object
    
    
    def get_llm_object(self):
        """
        """
        
        model_name = self.llm_model_name
        llm = None
        
        if model_name == 'openai':
            llm = ChatOpenAI(model_name = "gpt-3.5-turbo")
        elif model_name == 'llamacpp':
            # llm = LlamaCpp(model_path = r'C:\Users\ilija\models\llama.cpp\models\7B_noblas\ggml-model-q4_0.gguf',
            llm = LlamaCpp(model_path = r'C:\Users\ilija\llama.cpp\models\7B\ggml-model-q4_0.gguf',
                           verbose = True,
                           n_ctx = 1024,
                           n_threads = 8,
                           n_gpu_layers = 40,
                           n_batch = 512)
        elif model_name == 'gpt4all':
            # llm = GPT4All(
            #     model = './models/ggml-gpt4all-j-v1.3-groovy.bin',
            #     callbacks = [StreamingStdOutCallbackHandler()]
            # )
            llm = GPT4All(model = r"https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf")
            # verbose = True, n_ctx = 1024, n_gpu_layers = 1, n_batch = 4)
        elif model_name == 'ggml-falcon':
            llm = GPT4All(model = r"D:\Downloads\ggml-model-gpt4all-falcon-q4_0.bin")
            # verbose = True, n_ctx = 1024, n_gpu_layers = 1, n_batch = 4)
        elif model_name.startswith('flan'):
            tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}")
            model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{model_name}")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            llm = HuggingFacePipeline(
                pipeline = pipe,
                model_kwargs = {"temperature": 0, "max_length": 512},
            )
        elif model_name.startswith('distilbert'):
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
            model = AutoModelForSeq2SeqLM.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            llm = HuggingFacePipeline(
                pipeline = pipe,
            )
        elif model_name.startswith('bert'):
            tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/bert-base-nli-stsb-mean-tokens")
            model = AutoModelForSeq2SeqLM.from_pretrained("sentence-transformers/bert-base-nli-stsb-mean-tokens")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            llm = HuggingFacePipeline(
                pipeline = pipe,
            )
        elif model_name.startswith('roberta'):
            tokenizer = AutoTokenizer.from_pretrained(f"deepset/roberta-base-squad2")
            model = RobertaForCausalLM.from_pretrained("deepset/roberta-base-squad2")
            pipe = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
            llm = HuggingFacePipeline(
                pipeline = pipe,
            )
        
        if not llm:
            raise NameError("The model_name for llm that you entered is not supported")
        
        self.llm = llm
    
    
    def load_documentation_folder(self,
                                  docs_dir):
        """
        """
        
        
        self.read_documents(docs_dir)
        self.split_documents()
        self.get_embeddings_object()
        
        persist_directory = os.path.join(self.db_dir, f'db_{os.path.basename(os.path.normpath(docs_dir))}_{self.embedding_model_name}')
        if not os.path.exists(persist_directory):
            chroma = Chroma.from_documents(self.texts, self.embedding_object, persist_directory = persist_directory)
        else:
            chroma = Chroma(persist_directory = persist_directory, embedding_function = self.embedding_object)
        
        self.db = chroma
    
    def get_retrieval_qa(self,
                         query):
        """

        """
        
        self.get_llm_object()
        qa = RetrievalQA.from_chain_type(llm = self.llm,
                                         chain_type = "stuff",
                                         retriever = self.db.as_retriever(search_kwargs = {'k': 1}),
                                         chain_type_kwargs = {"prompt": self.llm_rag_prompt},
                                         return_source_documents = True
                                         )
        result = qa({"query": query})
        console_print(result, 'result')
        relevant_docs = self.db.similarity_search(query, k = 10)
        console_print(relevant_docs, 'relevant_docs')
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
        
        
        
        