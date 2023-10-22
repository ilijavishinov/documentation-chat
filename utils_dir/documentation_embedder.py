from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain.embeddings import LlamaCppEmbeddings, GPT4AllEmbeddings
import os


os.environ["OPENAI_API_KEY"] = "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"


class DocumentationEmbedder:
    docs_dir = None
    db = None
    embedding_tokenizer = None
    embedding_model = None
    
    def __init__(self,
                 embedding_model_name = 'distilbert'):
        
        self.embedding_model_name = embedding_model_name
        self.get_embeddings_object()
    
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

