import os
from utils import *
import torch
from llm_model import GenerativeModel
from hbg_utils import *

def extract_themes_and_papers(input_string, referenced_papers_info):
    theme_block_pattern = r'(Theme\s*\d+:\s*.*?(?=Theme\s*\d+:|$))'

    theme_blocks = re.findall(theme_block_pattern, input_string, re.DOTALL)

    theme_pattern = r'Theme\s*\d+:\s*(.*)'
    title_pattern = r'-\s*(?:Title:|Paper\s*Title:)?\s*(.*)'

    title_pattern1 = 'Paper title:'
    title_pattern2 = 'Title:'
    title_pattern3 = 'Title of Paper:'
    
    themes = {}
    for block in theme_blocks:
        theme_match = re.search(theme_pattern, block)
        if theme_match:
            theme = theme_match.group(1).strip()
            themes[theme] = []
        title_text = block.strip().split(theme)[1].strip().split('\n')
        for t in title_text:
            t = t.strip()
            if t.startswith('- '):
                t = t[2:]
                t = t.strip()
                t = t.strip('"')
                if t.startswith(title_pattern1):
                    t = t[len(title_pattern1):]
                elif t.startswith(title_pattern2):
                    t = t[len(title_pattern2):]
                elif t.startswith(title_pattern3):
                    t = t[len(title_pattern3):]
                t = t.strip()
                if t not in referenced_papers_info:
                    continue
                themes[theme].append(t)
    for k in list(themes.keys()):
        if len(themes[k]) == 0:
            del themes[k]
    return themes


def background_graph_build(target_md_path, related_md_root_path, llm, device):
    with open(target_md_path, 'r') as file:
        center_paper_md = file.read()
    center_paper_title = get_title(center_paper_md)
    center_paper_abstract = get_abstract(center_paper_md)
    center_paper_related_work, _ = extract_relatedwork_and_references(center_paper_md)
    related_md_paths = []
    for root, dirs, files in os.walk(related_md_root_path):
        for file in files:
            if file.endswith(".mmd"):
                related_md_paths.append(os.path.join(root, file))
    referenced_papers_info = {}
    for related_md_path in related_md_paths:
        with open(related_md_path, 'r') as file:
            related_paper_md = file.read()
        related_paper_title = get_title(related_paper_md)
        related_paper_abstract = get_abstract(related_paper_md)
        if related_paper_abstract == False:
            continue
        related_paper_abstract_list = related_paper_abstract.split('\n\n')
        related_paper_abstract = ""
        for r in related_paper_abstract_list:
            related_paper_abstract += r

        if len(related_paper_abstract) >= 0 and len (related_paper_title) >= 0:
            referenced_papers_info[related_paper_title] = {}
            referenced_papers_info[related_paper_title]['ab'] = related_paper_abstract
            referenced_papers_info[related_paper_title]['path'] = os.path.basename(related_md_path).split('.mmd')[0]
        else:
            continue
    str_referenced_papers_info = str(referenced_papers_info)
    str_input_prompt = {"title": center_paper_title, "Abstract": center_paper_abstract, 
                            "Related work": center_paper_related_work, "Related work detail": str_referenced_papers_info}
    
    instruction = read_json_file('./prompt.json')['themes_infer_2']
    instruction += "Title of target paper: " + str_input_prompt["title"] + '\n'
    instruction += "Abstract of target paper: " + str_input_prompt["Abstract"] + '\n\n'
    instruction += "Related work: " + '\n\n'
    for i, (k,v) in enumerate(referenced_papers_info.items()):
        instruction += "- Title: " + str(k) + "\nAbstract: " + str(v['ab']) + '\n\n'
        
    input = instruction + '''\nPlease output the different themes and their corresponding titles of papers. Here is the template for a theme generation format, you must follow this format to output your theme generation result: \n
    Output:
        Theme 1: <Theme name>
            - Paper title under Theme 1
            - Paper title under Theme 1
            - ......
        Theme 2: <Theme name>
            - Paper title under Theme 2
            - Paper title under Theme 2
            - ......
        ......
        Theme m: <Theme name>
            - Paper title under Theme m
            - Paper title under Theme m
            - ......
    '''
    generate_model = GenerativeModel(device=device)
    messages_en = [{"role": "user", "content": input}]
    str_result = generate_model.encode(messages=messages_en)


    max_try = 0
    while max_try < 3:
        themes = extract_themes_and_papers(str_result, referenced_papers_info)
        if len(themes.keys()) > 0:
            break
        max_try += 1
    if len(themes.keys()) == 0:
        raise ValueError('Step5: Generate three times, cannot generate valid Theme!')

    theme_description = {}
    for theme in themes.keys():
        messages_en = [{"role": "user", "content": "Please generate a concise description for the theme \"" + theme + "\"" + "."}]
        theme_description[theme] = generate_model.encode(messages=messages_en)

    paper_file_name = os.path.basename(target_md_path).split(".mmd")[0]
    print(referenced_papers_info)
    
    torch.save((themes, theme_description, referenced_papers_info), os.path.join(graph_path, f"{paper_file_name}_themesv1.pt"))


if __name__ == '__main__':
    args = init()

    graph_path = '../result/' + args.filename + '/graph/'
    target_md_path = '../result/' + args.filename + '/paper/' + args.filename + '.mmd'
    related_md_root_path = '../result/' + args.filename + '/paper/background/mmd/'

    logging.info('Step5: Infer themes')
    if args.if_related_search:
        background_graph_build(target_md_path, related_md_root_path, args.llm, args.gpu)
    logging.info('Step5: Infer themes done')