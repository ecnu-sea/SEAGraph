from llm_model import *
from utils import *
import torch
from torch_geometric.data import Data
import logging
from utils import *


def q_n_iterative(graph, arg_embeds, selection_list):
    retrieve_content_list = {}
    retrieve_content = ""
    edge_index = []

    arg_sim_withnodes = torch.mm(arg_embeds, graph.node_embeds.T).squeeze(0)
    top_values, top_indices = torch.topk(arg_sim_withnodes, k=selection_list[0], dim=0)

    current_nodes = top_indices
    all_selected_nodes = set(current_nodes.tolist())

    for i in range(top_indices.size(0)):
        edge_index.append([-1, top_indices[i].item()])
        retrieve_content_list[graph.node_text[top_indices[i]]] = "#"

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

                total_score = 0.5 * semantic_similarity + 0.5 * direct_similarity

                candidate_scores.append((total_score, neighbor_idx.item(), node_idx.item()))

        candidate_scores.sort(reverse=True, key=lambda x: x[0])

        select_num = 0
        for i in range(len(candidate_scores)):
            if graph.node_text[candidate_scores[i][1]] not in retrieve_content_list:
                new_nodes = [candidate_scores[i][1]]
                all_selected_nodes.update(new_nodes)
                retrieve_content_list[graph.node_text[candidate_scores[i][1]]] = "#" * (iteration + 1)
                edge_index.append([candidate_scores[i][2], candidate_scores[i][1]])
                select_num += 1
            if select_num == num_select:
                break
        current_nodes = torch.tensor(list(all_selected_nodes))
    
    retrieve_content = ""
    for node, highlight in retrieve_content_list.items():
        retrieve_content += highlight + " " + node + '\n'
    return retrieve_content, edge_index


if __name__ == '__main__':
    args = init()

    review_json = read_json_file('../result/' + args.filename + '/review/raw_review_aug_' + args.filename + '.json')

    graph = torch.load(f'../result/{args.filename}/graph/paper_graph.pt')
    logging.info("Step 8: Retrieve semantic mind graph...")
    retrieved_content_list = []

    response_list = []
    sentence_model = SentenceEncoder(args.gpu)
    for i, review in tqdm(enumerate(review_json['reviews']), total=len(review_json['reviews'])):
        response_one_review = {}
        retrieve_one_review = {}
        for k in ['Strengths', 'Weaknesses', 'Questions']:
            response_one_key = {}
            retrieve_one_review[k] = {}
            for arg in review[k]:
                # print(f"Processing: {arg}")
                retrieve_one_review[k][arg] = {}
                arg_embeds = sentence_model.encode([arg])
                
                section_list = [3, 6, 9]
                highlights, edge_index = q_n_iterative(graph, arg_embeds, section_list)

                retrieve_one_review[k][arg]['arg_embeds'] = arg_embeds
                retrieve_one_review[k][arg]['highlights'] = highlights
                retrieve_one_review[k][arg]['edge_index'] = edge_index

        retrieved_content_list.append(retrieve_one_review)
    torch.save(retrieved_content_list, f'../result/{args.filename}/{args.filename}_retrieved_smg.pt')
