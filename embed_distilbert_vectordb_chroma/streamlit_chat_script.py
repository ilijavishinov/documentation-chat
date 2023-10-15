from re import split
import random
import torch
import streamlit as st
from streamlit_chat import message

# from llama2gptq.qa import qa, load_model, load_db
# from llama2gptq.ingest import extract_ref

DEVICE = 'cpu'
TITLE = 'AWS Documentation Chat'
HUG = 'https://assets.stickpng.com/images/585e4bcdcb11b227491c3396.png'
ANGRY = "https://static-00.iconduck.com/assets.00/aws-icon-2048x2048-274bm1xi.png"

st.set_page_config(page_title=TITLE)#, layout="wide")
st.header(TITLE)
st.markdown('''
###
''', unsafe_allow_html=True)

import utils_loka as utils
import importlib
importlib.reload(utils)



model_name = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"

@st.cache_resource  # 👈 Add the caching decorator
def initializing():
    docs_dir = 'loka_final/docs_loka'
    df = utils.df_from_documentation(docs_dir = docs_dir, model_name = model_name)
    collection_path = "D:\python_projects\mushon_semantic_search\loka\chromadb_utils_demo"
    collection_name = "aws"
    collection = utils.create_chromadb_collection(path = collection_path,
                                                  name = collection_name)
    
    utils.add_df_rows_to_collection(collection = collection,
                                    df = df)
    return collection

collection = initializing()




# @st.cache_resource
# def load_transformer():
#   return (load_model(DEVICE), load_db(DEVICE))


# transformer, db = load_transformer()

styl = """
<style>
    .stTextInput {
      position: fixed;
      bottom: 3rem;
      z-index: 1;
    }
    .StatusWidget-enter-done{
      position: fixed;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
    }
    .StatusWidget-enter-done button{
      display: none;
    }
</style>
"""

BTN_STYLE = """
color: #aaa;
padding-right: 0.5rem;
"""


st.markdown(styl, unsafe_allow_html=True)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'answers' not in st.session_state:
    st.session_state['answers'] = []
    
if 'buttons' not in st.session_state:
    st.session_state['buttons'] = []
    
if 'last_source' not in st.session_state:
    st.session_state['last_source'] = ''


def click_button(clicked_url):
    with st.sidebar:
        with open(fr"D:\python_projects\mushon_semantic_search\loka_final\docs_loka\{clicked_url}.md", 'r', encoding = 'utf-8') as f:
            content = f.read()
            st.markdown(content)

def query(query):
    st.session_state.past.append(query)
    history = []
    for i, _ in enumerate(st.session_state['generated']):
        history.append([st.session_state['past'][i],
                        st.session_state["generated"][i]])
    
    # answer, refs = qa(query, DEVICE, db, transformer, history)
    
    question = query
    
    results = utils.get_top_docs(question = question,
                                 collection = collection,
                                 model_name = model_name,
                                 n_results = 2)
    
    # all_answers_string = utils.answer_q_from_docs(question = question,
    #                                               documents = results['documents'][0])

    first_answer = utils.answer_q_from_docs(question = question,
                                            documents = results['documents'][0])
    
    first_source = results['metadatas'][0][0]['source']
    other_sources = list()
    for meta in results['metadatas'][0]:
        other_sources.append(meta['source'])
    
    answer, refs = first_answer, other_sources
    # Append references
    st.session_state.generated.append(answer)
    
    all_sources = list()
    my_link_id = 1
    answer += '<hr style="border: 1px solid #424242;"> Source: '
    for ref in [first_source]:
        ref_info = {'link': ref, 'title': ref}  # extract_ref(ref)
        all_sources.append(ref)
        st.session_state.buttons.append(ref)
        st.session_state.last_source = ref+'.md'
        answer += '<br>'
        answer += f"<em href='{ref_info['link']}.md' style='{BTN_STYLE}' id='my-link-{my_link_id}'>{ref_info['title']}</em>"
        answer += '<br>'
    answer += '<hr style="border: 1px solid #424242;"> Other relevant docs: '
    for ref in refs:
        st.session_state.buttons.append(ref)
        if ref in all_sources:
            continue
        all_sources.append(ref)
        ref_info = {'link': ref, 'title': ref}  # extract_ref(ref)
        answer += '<br>'
        answer += f"<em href='{ref_info['link']}.md' style='{BTN_STYLE}' id='my-link-{my_link_id}'>{ref_info['title']}</em>"

    st.session_state.answers.append(answer)
    
    # source_button = st.button(f'{first_source}', on_click = click_button, args = [first_source], key = str(random.randint(0,1000000)))
    
    
    return answer

# Using "with" notation
# with st.sidebar:
#     with open(r"D:\python_projects\mushon_semantic_search\loka_final\docs_loka\sagemaker_documentation\amazon-sagemaker-toolkits.md", 'r', encoding = 'utf-8') as f:
#         content = f.read()
#         st.markdown(content)
#     # add_radio = st.radio(
#     #     "Choose a shipping method",
#     #     ("Standard (5-15 days)", "Express (2-5 days)")
#     # )

# if st.button("my-link"):
#     st.write("Link clicked!")
    
# # import webbrowser
para = st.experimental_get_query_params()
if 'url' in para.keys():
    clicked_url = para.get("url")[0]
# if st.button("my-link"):
    st.write(f"{clicked_url} clicked!")
# # if 'url' in para.keys():
#     st.experimental_set_query_params()
#     clicked_url = para.get("url")[0]
#     st.write(f'you just clicked {para.get("url")[0]}')
#     # webbrowser.open_new_tab(f'https://{para.get("url")[0]}.com')
#     # Using "with" notation
#     with st.sidebar:
#         with open(fr"D:\python_projects\mushon_semantic_search\loka_final\docs_loka\{clicked_url}.md", 'r', encoding = 'utf-8') as f:
#             content = f.read()
#             st.markdown(content)

def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text


user_input = get_text()


if user_input:
    query(user_input)

col1, col2 = st.columns([0.7, 0.3])
if st.session_state['generated']:
    # with col2:
    #     for i, _ in enumerate(st.session_state['generated']):
    #         st.text('Sources:')
    #         for button_text in st.session_state["buttons"]:
    #             st.button(f'{button_text}', on_click = click_button, args = [button_text], key = str(random.randint(0, 1000000)))
    #         # print('__________DEBUG____________')
    #         # print(st.session_state["buttons"][i])
    #         # print('__________DEBUG____________')
    #         # message(st.session_state["buttons"][i],
    #         #         key=str(i), logo = ANGRY, allow_html = True)
    #         #
    # with col1:
    with st.sidebar:
        last_source = st.session_state["last_source"]
        with open(fr"D:\python_projects\mushon_semantic_search\loka_final\docs_loka\{last_source}", 'r', encoding = 'utf-8') as f:
            content = f.read()
            content = '# Source of last answer:\n' + content
            st.markdown(content)
            
    for i, _ in enumerate(st.session_state['generated']):
        message(st.session_state['past'][i], is_user=True,
                key=str(i) + '_user', logo=HUG)
        message(st.session_state["answers"][i],
                key=str(i), logo=ANGRY, allow_html=True)
        # print('__________DEBUG____________')
        # print(st.session_state["buttons"][i])
        # print('__________DEBUG____________')
        # message(st.session_state["buttons"][i],
        #         key=str(i), logo = ANGRY, allow_html = True)
        #
        


        
#
# if st.button("my-link"):
#     st.write("Link clicked!")