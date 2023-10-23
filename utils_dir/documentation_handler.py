from typing import List
from utils_dir.documentation_embedder import DocumentationEmbedder
import tqdm
from langchain.vectorstores import Chroma
from pathlib import Path
from langchain.document_loaders import (
    UnstructuredMarkdownLoader
)
import os
import utils_dir.text_processing as text_processing
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, CharacterTextSplitter


class DocumentationHandler:
    documents: List = []
    texts: List = []
    docs_dir: str = None
    db = None
    
    def __init__(self,
                 db_dir: str = None):
        self.db_dir = db_dir
    
    def read_documents(self,
                       docs_dir: str):
        """
        Reads all markdown files from a folder structure as langchain documents
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
                        embedding_agent: DocumentationEmbedder,
                        chunk_size: List[int],
                        text_splitter_name: str = 'recursive'):
        """
        Splits the document based on a chunking strategy
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
                    embedding_agent.embedding_tokenizer,
                    chunk_size = chunk_size,
                    chunk_overlap = chunk_overlap,
                    separator = ' '
                )
            
            if not text_splitter:
                raise NameError("The text_splitter_name that you entered is not supported")
            
            texts = text_splitter.split_documents(self.documents)
            self.texts.extend(texts)
    
    def load_documentation_folder(self,
                                  embedding_agent: DocumentationEmbedder,
                                  docs_dir: str,
                                  text_splitter_name: str,
                                  chunk_size: List[int] = None,
                                  similarity_metric_name: str = 'cosine'):
        """
        Creates or load a chroma vector database depnding on if it exists for the passed documentation
        """
        
        # assertions
        if text_splitter_name == 'tokens' and embedding_agent.embedding_model_name not in ['distilbert', 'roberta', 'bert']:
            raise NameError(f"tokens chunking implementation for {embedding_agent.embedding_model_name} is not supported")
        
        # define the db folder name based on chunking and embedding parameters
        dir_suffix = f"{embedding_agent.embedding_model_name}_{similarity_metric_name}_{text_splitter_name}"
        if chunk_size:
            dir_suffix += f'_{"-".join([str(i) for i in chunk_size])}'
        persist_directory = os.path.join(self.db_dir, f'db_{os.path.basename(os.path.normpath(docs_dir))}_{dir_suffix}')
        
        # load chroma db, or create if it does not exist
        if not os.path.exists(persist_directory):
            self.read_documents(docs_dir)
            self.split_documents(embedding_agent = embedding_agent,
                                 text_splitter_name = text_splitter_name,
                                 chunk_size = chunk_size)
            print(len(self.texts))
            
            chroma = Chroma.from_documents(self.texts,
                                           embedding_agent.embedding_model,
                                           persist_directory = persist_directory,
                                           collection_metadata = {"hnsw:space": similarity_metric_name})
        else:
            chroma = Chroma(persist_directory = persist_directory,
                            embedding_function = embedding_agent.embedding_model,
                            collection_metadata = {"hnsw:space": similarity_metric_name})
        
        self.db = chroma
