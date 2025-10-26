from llm_model import *
from utils import *
import torch
import os
import warnings
import logging
from utils import *
from torch_geometric.data import Data


def hbg_retrieve(arg_embeds, subgraph_embeds, theme_embeds, paper_ab_embeds, themes_papers, theme_description, referenced_papers_info, section_list):
    candidate_dict = {} 
    
    theme_score = cal_themes_scores(theme_embeds, arg_embeds, subgraph_embeds)
    sorted_theme_score = {key: value for key, value in sorted(theme_score.items(), key=lambda item: item[1], reverse=True)}

    num_select = section_list[0] if len(section_list) > section_list[0] else len(section_list)
    selected_theme = list(sorted_theme_score.keys())[:num_select]
    
    candidate_papers_ab = {}
    for theme in selected_theme:
        candidate_dict[theme_description[theme]] = sorted_theme_score[theme]
        for title in themes_papers[theme]:
            candidate_papers_ab[title] = {
                'ab_embed': paper_ab_embeds[title],
                'theme_score': sorted_theme_score[theme]
            }

    paper_ab_score = cal_abstract_scores(candidate_papers_ab, arg_embeds, subgraph_embeds)
    sorted_paper_ab_score = {key: value for key, value in sorted(paper_ab_score.items(), key=lambda item: item[1], reverse=True)}

    num_select = section_list[1] if len(section_list) > section_list[1] else len(section_list)
    selected_paper_ab = list(sorted_paper_ab_score.keys())[:num_select]
    candidate_papers = {}
    for title in selected_paper_ab:
        candidate_dict[referenced_papers_info[title]['ab']] = sorted_paper_ab_score[title]
        candidate_papers[title] = {
            'theme_score': candidate_papers_ab[title]['theme_score'],
            'paper_ab_score': sorted_paper_ab_score[title]
        }
   
    graph_list = []
    for title in selected_paper_ab:
        filename = referenced_papers_info[title]['path']
        graph = torch.load(f'../result/{args.filename}/graph/background/' + filename + '_' + str(args.similarity_threshold) + '_graph.pt')
        graph.title = []
        for _ in range(graph.node_embeds.size(0)):
            graph.title.append(title)
        graph_list.append(graph)

    merged_graph = merge_graphs(graph_list)
    candidate_paper_chunk = cal_chunk_scores(merged_graph, candidate_papers, arg_embeds, subgraph_embeds, section_list[2])

    candidate_dict.update(candidate_paper_chunk)
    sorted_candidate_dict = {key: value for key, value in sorted(candidate_dict.items(), key=lambda item: item[1], reverse=True)}
    selected_candidate = list(sorted_candidate_dict.keys())[:20]
    retrieve_content = ""
    for text in selected_candidate:
        retrieve_content += text + '\n\n'

    return retrieve_content

def cal_chunk_scores(graph, candidate_papers, arg_embeds, subgraph_embeds, selection_list):
    retrieve_content_list = {}

    arg_sim_withnodes = torch.mm(arg_embeds, graph.node_embeds.T).squeeze(0)
    top_values, top_indices = torch.topk(arg_sim_withnodes, k=selection_list[0], dim=0)

    current_nodes = top_indices
    all_selected_nodes = set(current_nodes.tolist())

    for iteration, num_select in enumerate(selection_list[1:], start=1):
        candidate_scores = []
        
        for node_idx in current_nodes:
            mask = (graph.edge_index[0] == node_idx) | (graph.edge_index[1] == node_idx)
            sub_edge_index = graph.edge_index[:, mask]
            neighbors = torch.unique(sub_edge_index)
            
            for neighbor_idx in neighbors:
                if neighbor_idx.item() in all_selected_nodes:
                    continue

                neighbor_embed = graph.node_embeds[neighbor_idx].unsqueeze(0)
                direct_similarity = torch.mm(neighbor_embed, arg_embeds.T).squeeze(0).item()
                semantic_similarity = torch.mm(neighbor_embed, graph.node_embeds[current_nodes].T).squeeze(0).mean().item()
                subgraph_similarity = torch.mm(neighbor_embed, subgraph_embeds.T).squeeze(0).item()

                title = graph.title[neighbor_idx]
                total_score = subgraph_similarity * 0.2 + semantic_similarity * 0.2 + direct_similarity * 0.4 + (candidate_papers[title]['paper_ab_score'] + candidate_papers[title]['theme_score']) * 0.1

                candidate_scores.append((total_score, neighbor_idx.item(), node_idx.item()))

        candidate_scores.sort(reverse=True, key=lambda x: x[0])
        select_num = 0
        for i in range(len(candidate_scores)):
            if graph.node_text[candidate_scores[i][1]] not in retrieve_content_list:
                new_nodes = [candidate_scores[i][1]]
                all_selected_nodes.update(new_nodes)
                retrieve_content_list["Title: " + graph.title[candidate_scores[i][1]] + " " + "#" * (iteration + 1) + graph.node_text[candidate_scores[i][1]]] = candidate_scores[i][0]
                select_num += 1
            if select_num == num_select:
                break
        current_nodes = torch.tensor(list(all_selected_nodes))
    return retrieve_content_list


def cal_themes_scores(theme_embeds, arg_embeds, subgraph_embeds):
    theme_scores = {}
    for theme, theme_embed in theme_embeds.items():
        theme_similarity = torch.mm(theme_embed, arg_embeds.T).squeeze(0).item()
        subgraph_similarity = torch.mm(theme_embed, subgraph_embeds.T).squeeze(0).item()
        theme_scores[theme] = 0.5 * theme_similarity + 0.5 * subgraph_similarity
    return theme_scores

def cal_abstract_scores(candidate_papers_ab, arg_embeds, subgraph_embeds):
    paper_ab_scores = {}
    for title, ab_info in candidate_papers_ab.items():
        paper_similarity = torch.mm(ab_info['ab_embed'], arg_embeds.T).squeeze(0).item()
        subgraph_similarity = torch.mm(ab_info['ab_embed'], subgraph_embeds.T).squeeze(0).item()
        paper_ab_scores[title] = (paper_similarity + ab_info['theme_score'] + subgraph_similarity) / 3
    return paper_ab_scores


def merge_graphs(graph_list):
    all_edge_index = []
    all_node_text = []
    all_node_embeds = []
    all_graph_title = []

    node_offset = 0

    for graph in graph_list:
        adjusted_edge_index = graph.edge_index + node_offset
        
        all_edge_index.append(adjusted_edge_index)
        all_node_text += graph.node_text
        all_graph_title += graph.title
        all_node_embeds.append(graph.node_embeds)
        
        node_offset += graph.node_embeds.size(0)

    merged_edge_index = torch.cat(all_edge_index, dim=1)
    merged_node_embeds = torch.cat(all_node_embeds, dim=0)

    merged_graph = Data(
        edge_index=merged_edge_index,
        node_text=all_node_text,
        node_embeds=merged_node_embeds,
        title=all_graph_title
    )

    return merged_graph


def count_subgraph_embeds(node_embeds, edge_index):
    node_indices = set()
    for i in range(len(edge_index)):
        if edge_index[i][0] != -1:
            node_indices.add(edge_index[i][0])
        if edge_index[i][1] != -1:
            node_indices.add(edge_index[i][1])
    subgraph_embeds = []
    for idx in node_indices:
        subgraph_embeds.append(node_embeds[idx].unsqueeze(0))
    return torch.mean(torch.cat(subgraph_embeds, dim=0), dim=0).unsqueeze(0)

if __name__ == '__main__':
    args = init()
    sentence_model = SentenceEncoder(args.gpu)

    review_json = read_json_file('../result/' + args.filename + '/review/raw_review_aug_' + args.filename + '.json')

    _, theme_description, _ = torch.load(f'../result/{args.filename}/graph/' + args.filename + '_themesv1.pt')
    themes_papers, referenced_papers_info = torch.load(f'../result/{args.filename}/graph/' + args.filename + '_related_papers_info.pt')
    
    paper_content = torch.load('../result/' + args.filename + '/' + args.filename + '_retrieved_smg.pt')

    theme_embeds = {}
    for k,v in theme_description.items():
        theme_embeds[k] = sentence_model.encode([v])

    paper_ab_embeds = {}
    for k,v in referenced_papers_info.items():
        paper_ab_embeds[k] = sentence_model.encode(['Title: ' + k + '\n' + 'Abstract: ' + v['ab']])
    paper_graph = torch.load(f'../result/{args.filename}/graph/paper_graph.pt')

    logging.info("Step9: Retrieve hierarchical background graph...")
    retrieved_content_list = []
    for i, review in tqdm(enumerate(review_json['reviews']), total=len(review_json['reviews'])):
        retrieve_one_review = {}
        for k in ['Strengths', 'Weaknesses', 'Questions']:
            retrieve_one_review[k] = {}
            for arg in review[k]:
                # print(f"Processing: {arg}")
                retrieve_one_review[k][arg] = {}
                # arg_embeds = sentence_model.encode([arg])
                arg_embeds = paper_content[i][k][arg]['arg_embeds']
                highlights = paper_content[i][k][arg]['highlights']

                subgraph_embeds = count_subgraph_embeds(paper_graph.node_embeds, paper_content[i][k][arg]['edge_index'])
                section_list = [3,6,[10,20,30]]
                related_work = hbg_retrieve(arg_embeds, subgraph_embeds, theme_embeds, paper_ab_embeds, themes_papers, theme_description, referenced_papers_info, section_list)
                retrieve_one_review[k][arg] = related_work
        retrieved_content_list.append(retrieve_one_review)
    save_json_file(retrieved_content_list, f'../result/{args.filename}/{args.filename}_retrieved_hbg.json')
    logging.info("Step9: Retrieve hierarchical background graph success!")
