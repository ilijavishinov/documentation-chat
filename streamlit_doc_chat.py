from utils_dir.utils_class import DocumentationAgent
import os
import streamlit as st
from streamlit_chat import message
import utils_dir.streamlit_utils as streamlit_utils
import utils_dir.langchain_utils as langchain_utils
import utils_dir.text_processing as text_processing
import importlib
importlib.reload(langchain_utils)
importlib.reload(streamlit_utils)

DEVICE = 'cuda'
st.set_page_config(page_title = 'AWS Documentation Chat')
st.header('AWS Documentation Chat')
st.markdown(streamlit_utils.styl, unsafe_allow_html = True)

if 'generated' not in st.session_state: st.session_state['generated'] = []
if 'past' not in st.session_state: st.session_state['past'] = []
if 'answers' not in st.session_state: st.session_state['answers'] = []
if 'last_source' not in st.session_state: st.session_state['last_source'] = ''

docs_dir = './docs_loka'
embeddings_model_name = 'distilbert'
retrieval_model_name = [
    # 'qa_roberta',
    'qa_albert',
    # 'openai',
][0]
db_dir = f'./dbs'


@st.cache_resource
def initializing():
    if retrieval_model_name.startswith('qa'):
        doc_agent_ = DocumentationAgent(db_dir = db_dir,
                                        embedding_model_name = embeddings_model_name,
                                        qa_model_name = retrieval_model_name)
    else:
        doc_agent_ = DocumentationAgent(db_dir = db_dir,
                                        embedding_model_name = embeddings_model_name,
                                        llm_model_name = retrieval_model_name)
    
    doc_agent_.load_documentation_folder(docs_dir = docs_dir,
                                         text_splitter_name = 'character')
    return doc_agent_


doc_agent = initializing()


def query(query):
    st.session_state.past.append(query)
    history = []
    for i, _ in enumerate(st.session_state['generated']):
        history.append([st.session_state['past'][i],
                        st.session_state["generated"][i]])
    
    if retrieval_model_name.startswith('qa'):
        result, relevant_documents = doc_agent.qa_response(query)
    else:
        result, relevant_documents = doc_agent.llm_rag(query)
    
    # Append references
    st.session_state.generated.append(text_processing.format_answer(result['result']))
    answer_html, first_source = streamlit_utils.answer_html_func(result, relevant_documents)
    
    st.session_state.last_source = first_source

    st.session_state.answers.append(answer_html)
    return answer_html


user_input = streamlit_utils.get_text()
if user_input:
    query(user_input)

if st.session_state['generated']:
    with st.sidebar:
        last_source = st.session_state["last_source"]
        if last_source != '/':
            with open(fr"{last_source}", 'r', encoding = 'utf-8') as f:
                content = f.read()
                content = '# Source of last answer:\n' + content
                st.markdown(content)
    
    for i, _ in enumerate(st.session_state['generated']):
        message(st.session_state['past'][i], is_user = True,
                key = str(i) + '_user', logo = streamlit_utils.USER_ICON)
        message(st.session_state["answers"][i],
                key = str(i), logo = streamlit_utils.AWS_ICON, allow_html = True)
