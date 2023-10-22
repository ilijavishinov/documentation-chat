from utils_dir.documentation_agent import DocumentationAgent
import os
import streamlit as st
from streamlit_chat import message
import utils_dir.streamlit_utils as streamlit_utils
import utils_dir.text_processing as text_processing
import importlib
importlib.reload(streamlit_utils)

DEVICE = 'cuda'
DOCS_DIR = './docs_loka'
EMBEDDINGS_MODEL_NAME = 'roberta'
ANSWERING_MODEL_NAME = [
    # 'openai',
    'qa_roberta'
    # 'qa_albert',
    # 'openai',
][0]
DB_DIR = f'./dbs'


def initialize_streamlit(page_title):
    st.set_page_config(page_title = page_title)
    st.header(page_title)
    st.markdown(streamlit_utils.styl, unsafe_allow_html = True)
    
    if 'generated' not in st.session_state: st.session_state['generated'] = []
    if 'past' not in st.session_state: st.session_state['past'] = []
    if 'answers' not in st.session_state: st.session_state['answers'] = []
    if 'last_source' not in st.session_state: st.session_state['last_source'] = ''


def show_streamlit_elements():
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
def initialize_documentation_agent():
    doc_agent_ = DocumentationAgent(db_dir = DB_DIR,
                                    embedding_model_name = EMBEDDINGS_MODEL_NAME,
                                    answering_model_name = ANSWERING_MODEL_NAME)
    
    doc_agent_.load_documentation_folder(docs_dir = DOCS_DIR,
                                         text_splitter_name = 'tokens',
                                         chunk_size = [502])
    return doc_agent_
    
    
def query(query_,
          doc_agent_):
    st.session_state.past.append(query_)
    history = []
    for i, _ in enumerate(st.session_state['generated']):
        history.append([st.session_state['past'][i],
                        st.session_state["generated"][i]])
    
    result, relevant_documents = doc_agent_.get_response(query_)
    
    # Append references
    st.session_state.generated.append(text_processing.format_answer(result['result']))
    answer_html, first_source = streamlit_utils.answer_html_func(result, relevant_documents)
    
    st.session_state.last_source = first_source
    
    st.session_state.answers.append(answer_html)
    return answer_html


if __name__ == '__main__':
    
    initialize_streamlit(page_title = 'AWS Documentation Chat')
    
    doc_agent = initialize_documentation_agent()
    
    user_input = streamlit_utils.get_text()
    if user_input:
        query(user_input, doc_agent)
    
    show_streamlit_elements()
    