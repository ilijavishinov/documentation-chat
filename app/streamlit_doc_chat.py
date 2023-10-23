from utils_dir.documentation_agent import DocumentationAgent
import streamlit as st
from streamlit_chat import message
import utils_dir.streamlit_utils as streamlit_utils
import utils_dir.text_processing as text_processing
import importlib
importlib.reload(streamlit_utils)
from utils_dir.text_processing import console_print


DEVICE = 'cuda'
DOCS_DIR = './docs_loka'
DB_DIR = f'./dbs'


def initialize_streamlit(page_title: str):
    """
    Initialize streamlit state and page config
    """
    
    st.set_page_config(page_title = page_title)
    st.header(page_title)
    st.markdown(streamlit_utils.styl, unsafe_allow_html = True)
    
    if 'generated' not in st.session_state: st.session_state['generated'] = []
    if 'past' not in st.session_state: st.session_state['past'] = []
    if 'answers' not in st.session_state: st.session_state['answers'] = []
    if 'last_source' not in st.session_state: st.session_state['last_source'] = ''
    if 'chat_loaded' not in st.session_state: st.session_state['chat_loaded'] = False


def show_streamlit_elements():
    """
    Show streamlit chat when new input is received
    """
    if st.session_state['generated']:
        with st.sidebar:
            last_source = st.session_state["last_source"]
            if last_source != '/':
                with open(fr"{last_source}", 'r', encoding = 'utf-8') as f:
                    content = f.read()
                    content = '# Source of last answer:\n\n' + content
                    st.markdown(content)
        
        for i, _ in enumerate(st.session_state['generated']):
            message(st.session_state['past'][i], is_user = True,
                    key = str(i) + '_user', logo = streamlit_utils.USER_ICON)
            message(st.session_state["answers"][i],
                    key = str(i), logo = streamlit_utils.AWS_ICON, allow_html = True)


@st.cache_resource
def initialize_documentation_agent(embeddings_model_name: str,
                                   answering_model_name: str,
                                   chunking_type: str,
                                   chunk_size: int):
    """
    Load the DocumentationAgent and keep it in cache until new configuration is chosen
    """
    
    doc_agent_ = DocumentationAgent(db_dir = DB_DIR,
                                    embedding_model_name = embeddings_model_name,
                                    answering_model_name = answering_model_name)
    
    doc_agent_.load_documentation_folder(docs_dir = DOCS_DIR,
                                         text_splitter_name = chunking_type,
                                         chunk_size = [chunk_size])
    return doc_agent_


def query(query_: str,
          doc_agent_: DocumentationAgent):
    """
    Generate answer for query and return html formatted answer
    """
    
    # add in streamlit state
    st.session_state.past.append(query_)
    history = []
    for i, _ in enumerate(st.session_state['generated']):
        history.append([st.session_state['past'][i],
                        st.session_state["generated"][i]])
    
    # get result
    result, relevant_documents = doc_agent_.get_response(query_)
    st.session_state.generated.append(text_processing.format_answer(result['result']))
    
    # create html
    answer_html, first_source = streamlit_utils.answer_html_func(result, relevant_documents)
    
    st.session_state.last_source = first_source
    
    st.session_state.answers.append(answer_html)
    return answer_html


def pass_assertions(embeddings_model_name: str,
                    chunking_type: str):
    """
    Make assertions for unsupported combinations from the frontend choices
    """
    
    if chunking_type == 'tokens' and embeddings_model_name not in ['roberta', 'distilbert']:
        st.write('Tokens chunking is not supported for the chosen embeddings model')
        return False
    
    return True


if __name__ == '__main__':
    initialize_streamlit(page_title = 'AWS Documentation Chat')
    
    with st.sidebar:
        with st.sidebar:
            embeddings_model_name = st.selectbox(
                'Choose the embedding model', (
                    'RoBERTa',
                    'DistilBERT ',
                    'OpenAI ADAv2',
                )
            ).lower()
            
            answering_model_name = st.selectbox(
                'Choose the answering model', (
                    'QA_RoBERTa',
                    'QA_AlBERT',
                    'QA_BERT',
                    'OpenAI GPT3.5',
                )
            ).lower()
        
        chunking_type = st.selectbox(
            'Choose the chunking type', (
                'tokens',
                'character',
            )
        )
        
        chunk_size = st.number_input(
            'Choose the chunks size (write it)',
            min_value = 50, max_value = 1000, value = 504, step = 0
        )
    
    if pass_assertions(embeddings_model_name,
                       chunking_type):
        console_print(f"""\n
            Loading chat with\n
            embeddings_model_name = {embeddings_model_name},\n
            answering_model_name = {answering_model_name},\n
            chunking_type = {chunking_type},\n
            chunk_size = {chunk_size}\n
        """)
        
        doc_agent = initialize_documentation_agent(
            embeddings_model_name = embeddings_model_name,
            answering_model_name = answering_model_name,
            chunking_type = chunking_type,
            chunk_size = chunk_size
        )
        
        # get user input and answer query if assertions are passed
        user_input = streamlit_utils.get_text()
        if user_input:
            query(user_input, doc_agent)
        
        show_streamlit_elements()
