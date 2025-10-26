import re
import os
from urllib.parse import quote
import requests
from bs4 import BeautifulSoup
import time
from fake_useragent import UserAgent
import logging
from pdf_parse import parse_pdf
from utils import *
from tqdm import tqdm
from hbg_utils import *


def extract_related_paper_by_index(related_work_content, references_content):
    reference_indices = re.findall(r"\[(\d+(?:[;,]\s*\d+)*)\]", related_work_content) 

    all_reference_indices = []
    for indices in reference_indices:
        all_reference_indices.extend(re.findall(r"\d+", indices))
    
    reference_dict = {} 
    for match in re.finditer(r"\[(\d+)\]\s+(.+)", references_content):
        index = int(match.group(1))
        paper_name = match.group(2).strip()
        reference_dict[index] = paper_name
    index = 1
    split_references = references_content.split("\n")
    for line in split_references:
        line = line.strip()
        if line.startswith("*") and "[" in line and "]":
            if index in reference_dict.keys(): 
                index += 1
                continue
            else:
                reference_dict[index] = " ".join(str(item) for item in line.split("]")[1:]).strip()
                index += 1
    related_work_references = []
    for index in all_reference_indices:
        index = int(index)
        if index in reference_dict:
            related_work_references.append(reference_dict[index])
    return related_work_references

def extract_related_paper_by_author_year(related_work_content, references_content):
    author_year_references = []
    matches = re.finditer(r'\(([^)]+?)\)', related_work_content)  
    for match in matches:  
        citation_string = match.group(1)  
        citations = citation_string.split(';')  
        for citation in citations:  
            parts = citation.rsplit(',', 1)  
            if len(parts) == 2:  
                authors, year = parts  
                authors = authors.strip().strip(')').strip('(')
                year = year.strip().strip(')').strip('(')  
                if len(year) >= 4 and len(year) <= 5:
                    try:
                        int_year = int(year[:4])
                        if int_year >= 1900 and int_year <= 2100:
                            author_year_references.append((authors, year))
                    except:
                        continue
    all_author_year_references = []
    for match in author_year_references:
        if isinstance(match, tuple):
            all_author_year_references.append((match[0].strip(), match[1]))
            if len(match) > 2:
                all_author_year_references.append((match[2].strip(), match[3]))
        else:
            all_author_year_references.append((match[0].strip(), match[1]))
    
    reference_dict = {} 
    for match in re.finditer(r"([A-Za-z\s,\.]+)\s*\((\d{4})\)\s*(.+)", references_content):
        author_year = (match.group(1).strip(), match.group(2))
        paper_name = match.group(3).strip()
        reference_dict[author_year] = paper_name

    related_work_references = []
    for author, year in all_author_year_references:
        author_year = (author, year)
        if author_year in reference_dict:
            related_work_references.append(reference_dict[author_year])
    
    return related_work_references

def construct_google_scholar_url(paper_name):
    encoded_title = quote(paper_name)
    search_url = "https://scholar.google.com/scholar?q={}".format(encoded_title)
    return search_url

def get_webpage_source(url):
    ua = UserAgent() 
    headers = {
        'User-Agent': ua.random  
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.status_code, response.text
        elif response.status_code == 429:
            logging.info("ERROR!!! NETWORK ERROR!")
            exit()
        else:
            return response.status_code, None
    except:
        logging.info("ERROR CODE: %d, NETWORK ERROR!", response.status_code)
        return 500, None

def find_first_paper_pdf_link(webpage_source):
    soup = BeautifulSoup(webpage_source, 'html.parser')
    max_try = 0
    while max_try < 3:
        first_paper_div = soup.find('div', {'data-rp': '0'})
        if first_paper_div is None:
            max_try += 1
            time.sleep(5)
            continue
        else:
            break
    if first_paper_div is None:
        return None
    first_paper_div = first_paper_div.find("div", {"class": "gs_or_ggsm"})
    if first_paper_div:
        pdf_link = first_paper_div.find('a', href=True)
        
        if pdf_link:
            return pdf_link['href']
        else:
            return None
    else:
        return None

def get_related_papers(paper_path, output_path):
    assert os.path.exists(paper_path), "{:s} Paper file path not exists".format(paper_path)
    input_content = read_txt_file(paper_path)
    related_work_content, references_content = extract_relatedwork_and_references(input_content)
    related_papers_by_index = extract_related_paper_by_index(related_work_content, references_content)
    related_papers_by_author_year = extract_related_paper_by_author_year(related_work_content, references_content)
    related_set = set(related_papers_by_index + related_papers_by_author_year)
    logging.info("RELATED PAPERS: %s", related_set)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.info("Step4: All found %d related papers in related work section and references.", len(related_set))
    logging.info("Step4: Start to find the pdf link of related papers...")
    papers_found = []
    for one_related_paper in related_set:
        if os.path.exists("{}/{}pdf".format(output_path, one_related_paper)):
            papers_found.append(one_related_paper)
            continue
        search_url = construct_google_scholar_url(one_related_paper)
        max_try = 0
        while max_try < 3:
            status_code, text = get_webpage_source(search_url)
            if status_code != 200:
                logging.info("Step4: Try %d times, Paper %s search failed, Error code: %d", max_try, one_related_paper, status_code)
                max_try += 1
                time.sleep(5)
                continue
            else:
                break
        if text is None:
            logging.info("Step4: Paper %s search failed", one_related_paper)
            continue
        pdf_url = find_first_paper_pdf_link(text)
        if pdf_url is None:
            logging.info("{} Download failed, no pdf link in the webpage".format(one_related_paper))
            continue
        max_try = 0
        while max_try < 3:
            download_result = False
            try:
                download_result = download_pdf(pdf_url, "{}/{}pdf".format(output_path, one_related_paper))
            except:
                time.sleep(5)
            max_try += 1
            if download_result:
                break
        if download_result:
            logging.info("{} Download success".format(one_related_paper))
            papers_found.append(one_related_paper)
            time.sleep(5)
    logging.info("Step4: Successfully found %d papers from %d, failed %d", len(papers_found), len(related_set), len(related_set)-len(papers_found))


if __name__ == '__main__':
    args = init()

    if args.if_related_search:
        logging.info("Step4: Find related papers...")
        paper_mmd_path = '../result/' + args.filename + '/paper/' + args.filename + '.mmd'
        bkg_pdf_path = '../result/' + args.filename + '/paper/background/pdf/'
        if not os.path.exists(bkg_pdf_path):
            os.makedirs(bkg_pdf_path)
        get_related_papers(paper_mmd_path, bkg_pdf_path)
        logging.info("Step4: Find related papers success!")
    else:
        bkg_pdf_path = '../result/' + args.filename + '/paper/background/pdf/'
        if not os.path.exists(bkg_pdf_path):
            raise ValueError("Not running search related papers, ifsearch=False, it's wrong. See bkg_related_paper_search.py Line215")
    
    papers_found = os.listdir(bkg_pdf_path)
    for p in papers_found:
        if not check_pdf_valid(os.path.join(bkg_pdf_path, p)):
            os.remove(os.path.join(bkg_pdf_path, p))
            continue
        if len(p) > 150:
            os.rename(os.path.join(bkg_pdf_path, p), os.path.join(bkg_pdf_path, p.split('.pdf')[0][:150] +'.pdf'))
    papers_found = os.listdir(bkg_pdf_path)
    papers_found = [p.split('.pdf')[0] for p in papers_found]


    bkg_mmd_path = '../result/' + args.filename + '/paper/background/mmd/'
    if not os.path.exists(bkg_mmd_path):
        os.makedirs(bkg_mmd_path)
    assert len(papers_found) > 0, "No papers found in {:s}".format(bkg_pdf_path)
    
    logging.info("Step4: Parse related papers...")
    print(papers_found)
    for p in tqdm(papers_found):
        if not os.path.exists(os.path.join(bkg_mmd_path, p+'.mmd')):
            parse_pdf(os.path.join(bkg_pdf_path, p + '.pdf'), bkg_mmd_path, args.gpu, verbose=True)
    mmd_files = os.listdir(bkg_mmd_path)
    assert len(mmd_files) == len(papers_found), "Parse related papers error! MMD files number: {:d}, papers number: {:d}".format(len(mmd_files), len(papers_found))
    logging.info("Step4: Parse related papers success!")