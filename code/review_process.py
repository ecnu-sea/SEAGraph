import os
from utils import *
import re
from llm_model import GenerativeModel
import logging

def extract_one_review(c):
    one_review = {}
    keywords_pair = [
        ('Strengths','Weaknesses'),
        ('Weaknesses', 'Questions'),
        ('Questions', 'Flag For Ethics Review'),
    ]
    
    for (s1,s2) in keywords_pair:
        match = re.search(r'\*\*' + s1 +':\*\*(.*?)\*\*' + s2, c, re.DOTALL)
        one_content = match.group(1).strip()
        if s1 in ['Strengths', 'Weaknesses', 'Questions']:
            one_content = replace_bullets_with_numbers(one_content)
        else:
            one_content = re.sub(r'\n(?!\n)', ' ', one_content)
        one_review[s1] = one_content
    return one_review

def replace_bullets_with_numbers(text):
    lines = text.split('\n')
    numbered_lines = []
    count = 1
    for line in lines:
        if line.strip().startswith('- '):
            numbered_line = f"{count}. {line.strip()[2:]}"
            numbered_lines.append(numbered_line)
            count += 1
        else:
            numbered_lines.append(line)
    return '\n'.join(numbered_lines)

def check_review_content(one_review_dict):
    for v in one_review_dict.values():
        if len(v)==0:
            return False
    return True

def txttojson(read_path, save_path):
    # For ICLR 2024 Format
    text = read_txt_file(read_path)
    review_all = {}
    reviews_list = []
    reviewer_num = 1
    for c in text.split('−＝≡'):
        ifexist_add = c.find('Add:Public Comment')
        if ifexist_add!=-1:
            c = c[:ifexist_add]
        if c.startswith("\n\n#### Official Review of Submission"):
            flag = True
            keywords = ['Strengths', 'Weaknesses', 'Questions']
            for key in keywords:
                if '**'+key+':**' not in c:
                    flag = False
                    break
            if flag==False:
                continue
            one_review = extract_one_review(c)
            if check_review_content(one_review):
                reviews_list.append(one_review)
                reviewer_num = reviewer_num + 1
    review_all['reviews'] = reviews_list
    save_json_file(review_all, save_path)

def extract_abstract_line(file_path, num_lines=10):
    with open(file_path, 'r', encoding='utf-8') as file:
        for _ in range(num_lines):
            line = file.readline()
            match = re.search(r'\(([^()]*)\)$', line)
            if match:
                content_in_parentheses = match.group(1)
                if 'Abstract' in content_in_parentheses:
                    return line.strip()
    return None


def review_augment(review_json_path, review_aug_json_path, abstract, device):
    generate_model = GenerativeModel(device)
    review_json = read_json_file(review_json_path)
    prompt_json = read_json_file('./prompt.json')
    prompt_id = "augment_review"
    prompt = prompt_json[prompt_id]
    review_json_update = {}
    review_json_update['reviews'] = []
    for i,r in enumerate(review_json['reviews']):
        review_one = {}
        for k in ['Strengths', 'Weaknesses', 'Questions']:
            review_one[k] = []
            for arg in r[k]:
                argument = '\n<ABSTRACT>\n'+ abstract + '\n<\ABSTRACT>\n'
                # print(argument)
                argument = argument + '\n<REVIEW>\n'+ k + ':\n' + arg + '\n<\REVIEW>\n'
                argument = argument + '\nPlease directly give me the augmented output for the review comment.'
                logging.info('--------AUG INPUT---------------')
                logging.info(prompt+argument)
                message = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": argument}
                ]
                augment_content = generate_model.encode(message)
                logging.info('--------AUG OUTPUT---------------')
                logging.info(augment_content)
                if augment_content.strip():
                    review_one[k].append(augment_content)
        review_json_update['reviews'].append(review_one)
    save_json_file(review_json_update, review_aug_json_path)


def process_split_response(string):
    lines = string.split("\n")

    result = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            result.append(line)
        else:
            if result:
                result[-1] += " " + line
    for i in range(len(result)):
        if result[i].endswith("</REVIEW>") or result[i].endswith("<//REVIEW>") or result[i].endswith("</\\REVIEW>") or result[i].endswith("<\\REVIEW>"):
            result[i] = result[i].rsplit("</REVIEW>", 1)[0].rsplit("<//REVIEW>", 1)[0].rsplit("</\\REVIEW>", 1)[0].rsplit("<\\REVIEW>", 1)[0].strip()
    return result

def review_split(review_json_path, review_split_json_path, device):
    generate_model = GenerativeModel(device)
    review_json = read_json_file(review_json_path)
    prompt_json = read_json_file('./prompt.json')
    prompt_id = "split_review"
    prompt_template = prompt_json[prompt_id]

    review_json_update = {}
    review_json_update['reviews'] = []
    for i,r in enumerate(review_json['reviews']):
        review_one = {}
        for k in ['Strengths', 'Weaknesses', 'Questions']:
            prompt = prompt_template.format(section_name=k)
            argument = '\n<REVIEW>\n'+ r[k] + '\n<\REVIEW>\n'
            if k == 'Questions':
                if r[k].strip() == 'Please see the weakness section for details.' or r[k].strip() == "See above" or r[k].strip() == "See above." or r[k].strip() == "See weaknesses.":
                    review_one[k] = []
                    continue
                argument = argument + '\nIf the review comments are references to weaknesses sections and have no substance, only the string <OVER> is output and nothing else.'

            argument = argument + "\nPlease give me the standardized output format for the review comments."
            logging.info(f'---------SPLIT INPUT------------')
            logging.info(prompt + argument)
            logging.info('---------SPLIT OUTPUT---------')
            message = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": argument}
            ]
            standard_content = generate_model.encode(message)
            logging.info(standard_content)
            logging.info('-----------------------')
            argument_list = process_split_response(standard_content)
            review_one[k] = argument_list
        review_json_update['reviews'].append(review_one)
    save_json_file(review_json_update, review_split_json_path)

if __name__ == '__main__':
    args = init()
    review_txt_path = '../data/raw_review/' + args.filename + '.txt'
    review_json_path = '../result/' + args.filename + '/review/' + args.filename + '.json'
    
    if not os.path.exists(review_json_path):
        logging.info("Step3: Convert review to json...")
        txttojson(review_txt_path, review_json_path)
        logging.info("Step3: Convert review to json success!")


    # split review
    review_spilit_file_name = 'raw_review_split_' + args.filename 
    review_split_json_path = '../result/' + args.filename + '/review/' + review_spilit_file_name + '.json'
    if not os.path.exists(review_split_json_path):
        logging.info("Step3: Split review...")
        review_split(review_json_path, review_split_json_path, args.gpu)
        logging.info("Step3: Split review success!")

    # augment review
    paper_path = '../result/' + args.filename + '/paper/' + args.filename + '_' + str(args.similarity_threshold) + '.txt'
    abstract = extract_abstract_line(paper_path)
    review_aug_file_name = 'raw_review_aug_' + args.filename
    review_aug_json_path = '../result/' + args.filename + '/review/' + review_aug_file_name + '.json'
    if not os.path.exists(review_aug_json_path):
        review_augment(review_split_json_path, review_aug_json_path, abstract, args.gpu)