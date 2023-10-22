import os
from utils_dir.documentation_handler import DocumentationHandler
from utils_dir.llm_agent import LlmAgent
from utils_dir.qa_agent import QaAgent
from utils_dir.documentation_embedder import DocumentationEmbedder

os.environ["OPENAI_API_KEY"] = "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"


class DocumentationAgent:
    
    documentation_handler = None
    embedding_agent = None
    qa_agent = None
    llm_agent = None
    
    def __init__(self,
                 db_dir = None,
                 embedding_model_name = None,
                 answering_model_name = None):
        
        self.documentation_handler = DocumentationHandler(db_dir = db_dir)
        self.embedding_agent = DocumentationEmbedder(embedding_model_name = embedding_model_name)
        
        if answering_model_name.startswith("qa_"):
            self.qa_agent = QaAgent(qa_model_name = answering_model_name)
        else:
            self.llm_agent = LlmAgent(llm_model_name = answering_model_name)
    
    def load_documentation_folder(self,
                                  docs_dir,
                                  text_splitter_name,
                                  chunk_size,
                                  similarity_metric_name = 'cosine'):
        """
        """
        
        self.documentation_handler.load_documentation_folder(embedding_agent = self.embedding_agent,
                                                             docs_dir = docs_dir,
                                                             text_splitter_name = text_splitter_name,
                                                             chunk_size = chunk_size,
                                                             similarity_metric_name = similarity_metric_name)
    
    def get_response(self,
                     query):
        """
        """
        
        if self.llm_agent:
            return self.llm_agent.llm_rag(query, self.documentation_handler.db)
        else:
            return self.qa_agent.qa_response(query, self.documentation_handler.db)
            
