import os
import streamlit as st
import utils_dir.text_processing as text_processing
# USER_ICON = 'https://assets.stickpng.com/images/585e4bcdcb11b227491c3396.png'
USER_ICON = 'https://creazilla-store.fra1.digitaloceanspaces.com/icons/7914927/man-icon-md.png'
AWS_ICON = "https://static-00.iconduck.com/assets.00/aws-icon-2048x2048-274bm1xi.png"


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


def answer_html_func(result, relevant_documents):
    answer = text_processing.format_answer(result['result'])
    
    # take last document
    # for qa model, the 0th and only result will contain the relevant answer
    # for rag with llm, the last document will contain the answer (see llm_rag and increaese of k)
    try:
        first_source = result['source_documents'][-1].metadata['source']
    except:
        first_source = '/'
        
    other_sources = list()
    for doc in relevant_documents:
        other_sources.append(doc.metadata['source'])
    
    all_sources = list()
    answer += '<hr style="border: 1px solid #424242;"> Source: '
    
    if not first_source.endswith('.md') and first_source != '/':
        first_source = first_source + '.md'
    all_sources.append(first_source)
    answer += f"<br><em href='{first_source}' style='{BTN_STYLE}'>{os.path.basename(os.path.normpath(first_source))}</em><br>"
    
    answer += '<hr style="border: 1px solid #424242;"> Other relevant docs: '
    for ref in other_sources:
        if not ref.endswith('.md'):
            ref = ref + '.md'
        if ref in all_sources:
            continue
        all_sources.append(ref)
        answer += f"<br><em href='{ref}' style='{BTN_STYLE}'>{os.path.basename(os.path.normpath(ref))}</em>"
        if len(all_sources) > 5:
            break
    
    return answer, first_source


def get_text():
    input_text = st.text_input("You: ", key = "input")
    return input_text
