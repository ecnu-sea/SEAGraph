from llm_model import *
from utils import *
import torch
import os
import warnings
import logging
from utils import *
from torch_geometric.data import Data



if __name__ == '__main__':
    args = init()

    prompt = read_json_file('./prompt.json')['rag_seagraph']

    review_json = read_json_file('../result/' + args.filename + '/review/raw_review_aug_' + args.filename + '.json')
    smg_retrieved = torch.load('../result/' + args.filename + '/' + args.filename + '_retrieved_smg.pt')
    hbg_retrieved = read_json_file('../result/' + args.filename + '/' + args.filename + '_retrieved_hbg.json')

    logging.info("Step10: seagraph generation...")
    response_list = []
    generative_model = GenerativeModel(args.gpu)
    for i, review in tqdm(enumerate(review_json['reviews']), total=len(review_json['reviews'])):
        response_one_review = {}
        for k in ['Strengths', 'Weaknesses', 'Questions']:
            response_one_key = {}
            for arg in review[k]:
                # print(f"Processing: {arg}")
                # arg_embeds = sentence_model.encode([arg])
                highlights = smg_retrieved[i][k][arg]['highlights']
                related_work = hbg_retrieved[i][k][arg]

                input = prompt + "\n<PAPER HIGHLIGHTS>\n"
                input += highlights + "\n</PAPER HIGHLIGHTS>\n"
                input = input + "\n<RELATED WORK>\n" + related_work + "\n</RELATED WORK>\n"
                input += "\n<REVIEW>\n" + k + ": " + arg + "\n</REVIEW>"
                messages = [
                    {"role": "user", "content": input},
                ]
                response = generative_model.encode(messages)
                response_one_key[arg] = response
            response_one_review[k] = response_one_key
        response_list.append(response_one_review)
    save_json_file(response_list, f'../result/{args.filename}/{args.filename}_rag_seagraph.json')
    logging.info("Step10: seagraph generation success!")