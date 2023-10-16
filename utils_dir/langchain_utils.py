from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from pathlib import Path
from langchain.document_loaders import (
    TextLoader,
)
import os
import text_processing
from langchain.text_splitter import RecursiveCharacterTextSplitter


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


def read_documents(docs_dir):
    """
    
    """
    
    glob = Path(f"{docs_dir}").glob
    ps = list(glob("**/*.md"))
    documents = list()
    for p in ps:
        file_extension = os.path.splitext(p)[1]
        if file_extension != 'md': continue
        document = TextLoader(p, encoding = "utf-8").load()[0]
        document.page_content = text_processing.markdown_to_text(document.page_content)
        document.metadata["source"] = document.metadata['source'].__str__()
        documents.append(document)
    return documents
    
    
def split_documents(documents,
                    chunk_size = 1000,
                    chunk_overlap = 200):
    """
    
    """
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    
    texts = text_splitter.split_documents(documents)
    return texts


def get_embeddings_object(model_name):
    """
    
    """
    
    embeddings_object = None
    
    if model_name == 'llamacpp':
        embeddings_object = LlamaCppEmbeddings(model_path = r'C:\Users\ilija\models\llama.cpp\models\7B_noblas\ggml-model-q4_0.gguf',
                                               verbose = True,
                                               n_ctx = 1024,
                                               n_gpu_layers = 1,
                                               n_batch = 4)
        
    if not embeddings_object:
        raise NameError("The model_name for embeddings that you entered is not supported")
    
    return embeddings_object


def get_llm_object(model_name):
    """

    """
    
    llm = None
    
    if model_name == 'llamacpp':
        llm = LlamaCpp(model_path = r'C:\Users\ilija\models\llama.cpp\models\7B_noblas\ggml-model-q4_0.gguf',
                       verbose = True, n_ctx = 1024, n_gpu_layers = 1, n_batch = 4)
    
    if not llm:
        raise NameError("The model_name for llm that you entered is not supported")
    
    return llm


def load_chroma(texts,
                embeddings,
                persist_directory):
    """
    
    """
    
    if not os.path.exists(persist_directory):
        chroma = Chroma.from_documents(texts, embeddings, persist_directory = persist_directory)
    else:
        chroma = Chroma(persist_directory = persist_directory, embedding_function = embeddings)
        
    return chroma



def get_retrieval_qa(llm,
                     vector_db,
                     prompt = None):
    """
    
    """
    
    qa = RetrievalQA.from_chain_type(llm = llm,
                                     chain_type = "stuff",
                                     retriever = vector_db.as_retriever(),
                                     chain_type_kwargs = {"prompt": prompt} if prompt else None,
                                     return_source_documents = True
                                     )
    
    return qa

    