from bs4 import BeautifulSoup
from markdown import markdown
import re


def markdown_to_lower_text(markdown_string: str):
    """
    Converts a markdown string to lower text
    """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))
    
    text = re.sub(r'<[^<]+?>', '', text)

    return text.lower()


def format_answer(answer: str):
    """
    Format answer for frontend display
    """
    
    answer = '. '.join(list(map(lambda x: x.strip().capitalize(), answer.split('.'))))
    answer = answer.strip()
    if answer[-1] != '.':
        answer += '.'
    
    return answer


def console_print_debug(arg, desc = None):
    print()
    print('_________________________ ********************** ______________________')
    print(desc)
    print(arg)
    print('_________________________ ********************** ______________________')
    print()


def console_print(str_):
    print('\n***** LOKA DOCS LOG *****\t\t', str_, '\n')
