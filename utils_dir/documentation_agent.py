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
    
    embedding_agent = None
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
        
        self.embedding_agent = DocumentationEmbedder(embedding_model_name)

        if qa_model_name:
            self.qa_agent = QaAgent(qa_model_name)

        if llm_model_name:
            self.llm_agent = LlmAgent(llm_model_name)
    
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
        persist_directory_standalone = os.path.join(self.db_dir, f'db_{os.path.basename(os.path.normpath(docs_dir))}_{dir_suffix}_standalone')
        
        # load chroma db, or create if it does not exist
        if not os.path.exists(persist_directory):
            self.read_documents(docs_dir)
            self.split_documents(text_splitter_name = text_splitter_name, chunk_size = chunk_size)
            chroma = Chroma.from_documents(self.texts,
                                           self.embedding_model,
                                           persist_directory = persist_directory,
                                           collection_metadata = {"hnsw:space": similarity_metric_name})
        else:
            chroma = Chroma(persist_directory = persist_directory,
                            embedding_function = self.embedding_model,
                            collection_metadata = {"hnsw:space": similarity_metric_name})
        
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
    
