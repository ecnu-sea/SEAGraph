import os
from urllib.parse import quote
import requests
from bs4 import BeautifulSoup
import time
from fake_useragent import UserAgent
import logging
from pdf_parse import parse_pdf
from utils import *
from hbg_utils import *
from tqdm import tqdm

def get_related_papers_by_search(graph_path, filename, save_path, device, start_year=2023):
    themes_path = graph_path + filename + '_themesv1.pt'
    themes, _, referenced_papers_info = torch.load(themes_path)
    for theme in tqdm(themes.keys()):
        encoded_title = quote(theme) 
        ua = UserAgent()
        search_url = "https://scholar.google.com/scholar?as_ylo={}&q={}".format(start_year, encoded_title)
        headers = {
            'User-Agent': ua.random  
        }
        max_try = 0
        while max_try < 3:
            response = requests.get(search_url, headers=headers)
            if response.status_code != 200:
                max_try += 1
                time.sleep(5)
                continue
            else:
                break
        if max_try == 3:
            logging.info("Step6: Failed to get the webpage source of theme %s", theme)
            continue
        soup = BeautifulSoup(response.text, 'html.parser')
        sucess_num = 0
        for index in range(3):
            first_paper_div = soup.find('div', {'data-rp': '{}'.format(index)})
            first_paper_div = first_paper_div.find("div", {"class": "gs_or_ggsm"})
            if first_paper_div:
                pdf_link = first_paper_div.find('a', href=True)
                if pdf_link:
                    max_try = 0
                    while max_try < 3:
                        try:
                            download_result = download_pdf(pdf_link['href'], os.path.join(save_path, 'pdf',f"{theme}_{index}.pdf"))
                            max_try += 1
                            if download_result:
                                if not check_pdf_valid(os.path.join(save_path, 'pdf',f"{theme}_{index}.pdf")):
                                    os.remove(os.path.join(save_path, 'pdf',f"{theme}_{index}.pdf"))
                                    break
                                paper_name = f"{theme}_{index}"
                                parse_pdf(os.path.join(save_path, 'pdf', f'{paper_name}.pdf'), os.path.join(save_path, 'mmd'), device, verbose=False)
                                paper_mmd = read_txt_file(os.path.join(save_path, 'mmd', f'{paper_name}.mmd'))
                                paper_title = get_title(paper_mmd)
                                paper_abstract = get_abstract(paper_mmd)
                                if paper_abstract == False:
                                    os.remove(os.path.join(save_path, 'pdf',f"{theme}_{index}.pdf"))
                                    os.remove(os.path.join(save_path, 'mmd', f'{theme}_{index}.mmd'))
                                    break
                                os.rename(os.path.join(save_path, 'mmd', f'{paper_name}.mmd'), os.path.join(save_path, 'mmd', f'{paper_title}.mmd'))
                                os.rename(os.path.join(save_path, 'pdf', f'{paper_name}.pdf'), os.path.join(save_path, 'pdf', f'{paper_title}.pdf'))
                                
                                themes[theme].append(paper_title)
                                referenced_papers_info[paper_title] = {}
                                referenced_papers_info[paper_title]['ab'] = paper_abstract
                                referenced_papers_info[paper_title]['path'] = os.path.basename(os.path.join(save_path, 'mmd', f'{paper_title}.mmd')).split('.mmd')[0]
                                sucess_num += 1
                                break
                        except: 
                            max_try += 1
                            time.sleep(5)
                time.sleep(8)
        logging.info("Step6: Successfully found %d papers for theme %s", sucess_num, theme)
    torch.save((themes, referenced_papers_info), os.path.join(graph_path, filename + '_related_papers_info.pt'))


if __name__ == '__main__':
    args = init()
    graph_path = '../result/' + args.filename + '/graph/'
    save_path = '../result/' + args.filename + '/paper/background/'
    
    if args.if_hot_search:
        get_related_papers_by_search(graph_path, args.filename, save_path, args.gpu)