import re
import requests
import pdfplumber


def extract_relatedwork_and_references(input_content):
    references_start = re.search(r"References|REFERENCES", input_content)
    if not references_start:
        return [], []
    references_content = input_content[references_start.start():]
    next_section_start = re.search(r"## +", references_content)
    if next_section_start:
        references_content = references_content[: next_section_start.start()]
    related_work_start = re.search(r"Related Work|Related work|Related works|Related Works", input_content) 
    if not related_work_start: 
        return [], []
    related_work_content = input_content[related_work_start.end():]
    next_section_start = re.search(r"## \d+", related_work_content)
    if next_section_start:
        related_work_content = related_work_content[: next_section_start.start()]
    return related_work_content, references_content


def download_pdf(pdf_url, save_path):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(pdf_url, headers=headers)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        return False


def get_abstract(input_content):
    abstract_start = re.search(r"Abstract|abstract|ABSTRACT", input_content)
    if not abstract_start:
        return False
    abstract_content = input_content[abstract_start.start():]
    next_section_start = re.search(r"## +", abstract_content)
    if next_section_start:
        abstract_content = abstract_content[: next_section_start.start()]
    if abstract_content.startswith('Abstract'):
        abstract_content = abstract_content[len('Abstract'):] 
    return abstract_content.strip()

def get_title(input_content):
    raw_title = input_content.split("\n")[0]
    title = raw_title.replace("#", "")
    return title.strip()


def check_pdf_valid(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) > 0:
                return True
            else:
                return False
    except Exception as e:
        return False