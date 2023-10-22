from langchain.docstore.document import Document
import torch
import tqdm
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModel, RobertaForCausalLM, AutoModelForQuestionAnswering
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings, GPT4AllEmbeddings
from pathlib import Path
from langchain.document_loaders import (
    TextLoader, UnstructuredMarkdownLoader
)
import os
import utils_dir.text_processing as text_processing
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, MarkdownHeaderTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter
from utils_dir_backup.ingest_data import sentence_to_vector

from utils_dir.documentation_handler import DocumentationHandler
from utils_dir.llm_agent import LlmAgent
from utils_dir.qa_agent import QaAgent
from utils_dir.documentation_embedder import DocumentationEmbedder

os.environ["OPENAI_API_KEY"] = "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"


class DocumentationAgent:
    
    documents = []
    texts = []
    docs_dir = None
    db = None
    embedding_tokenizer = None
    embedding_model = None

    qa_agent = None
    llm_agent = None
    
    
    def __init__(self,
                 db_dir = None,
                 standalone_chroma_db = False,
                 embedding_model_name = 'distilbert',
                 llm_model_name = None,
                 qa_model_name = None):
        
        self.db_dir = db_dir
        self.standalone_chroma_db = standalone_chroma_db
        
        self.embedding_model_name = embedding_model_name
        self.qa_model_name = qa_model_name
        
        if qa_model_name:
            self.qa_agent = QaAgent(qa_model_name)
        if llm_model_name:
            self.llm_agent = LlmAgent(llm_model_name)
    
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
            if not self.standalone_chroma_db:
                self.embedding_model_standalone = HuggingFaceEmbeddings(
                    model_name = model_name,
                    model_kwargs = {'device': 'cuda:0'},
                    # encode_kwargs = {'normalize_embeddings': False}`
                )
            else:
                self.embedding_model_standalone = AutoModel.from_pretrained(model_name)
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        elif self.embedding_model_name.startswith('bert'):
            model_name = "sentence-transformers/bert-base-nli-stsb-mean-tokens",
            self.embedding_model = HuggingFaceEmbeddings(
                model_name = model_name,
                model_kwargs = {'device': 'cuda:0'},
                # encode_kwargs = {'normalize_embeddings': False}`
            )
            if not self.standalone_chroma_db:
                self.embedding_model_standalone = HuggingFaceEmbeddings(
                    model_name = model_name,
                    model_kwargs = {'device': 'cuda:0'},
                    # encode_kwargs = {'normalize_embeddings': False}`
                )
            else:
                self.embedding_model_standalone = AutoModel.from_pretrained(model_name)
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        elif self.embedding_model_name.startswith('roberta'):
            # model_name = "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
            model_name = "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"
            self.embedding_model = HuggingFaceEmbeddings(
                model_name = model_name,
                model_kwargs = {'device': 'cuda:0'},
                # encode_kwargs = {'normalize_embeddings': False}`
            )
            if not self.standalone_chroma_db:
                self.embedding_model_standalone = HuggingFaceEmbeddings(
                    model_name = model_name,
                    model_kwargs = {'device': 'cuda:0'},
                    # encode_kwargs = {'normalize_embeddings': False}`
                )
            else:
                self.embedding_model_standalone = AutoModel.from_pretrained(model_name)
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)


        if not self.embedding_model:
            raise NameError("The model_name for embeddings that you entered is not supported")
    
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
        return self.llm_agent.llm_rag(query, self.db)
    
    def qa_response(self,
                    query):
        """
        
        """
        
        return self.qa_agent.qa_response(query, self.db)
    
