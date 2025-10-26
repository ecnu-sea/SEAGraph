from llm_model import SentenceEncoder
import numpy as np
import os 
import spacy
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import torch
import re
from torch_geometric.data import Data

def extract_last_bracket_content(line):
    matches = re.findall(r'\(.*?\)', line)
    if matches:
        return matches[-1]  
    return None

def build_edges_through_subtitle(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    edges = []
    num_lines = len(lines)
    for i in range(num_lines):
        current_content = extract_last_bracket_content(lines[i])
        if current_content:
            if i > 0:
                prev_content = extract_last_bracket_content(lines[i - 1])
                if prev_content == current_content:
                    edges.append([i, i - 1])
            
            if i < num_lines - 1:
                next_content = extract_last_bracket_content(lines[i + 1])
                if next_content == current_content:
                    edges.append([i, i + 1])
    if edges:
        edges_tensor = torch.tensor(edges, dtype=torch.long)
    else:
        edges_tensor = torch.empty((0, 2), dtype=torch.long) 

    return edges_tensor


def get_top_k_edges(chunks_sim, k):
    n = chunks_sim.size(0)

    diag_mask = ~torch.eye(n, dtype=torch.bool)

    masked_sim = torch.full_like(chunks_sim, float('-inf')) * ~diag_mask
    masked_sim[diag_mask] = 0
    chunks_sim_no_diag = chunks_sim * diag_mask + masked_sim

    topk_values, topk_indices = torch.topk(chunks_sim_no_diag.view(-1), k * n)

    edges = torch.stack(torch.unravel_index(topk_indices, chunks_sim.size()), dim=1)
    
    sym_edges = []
    for edge in edges:
        sym_edges.append(edge.tolist())  
        sym_edges.append([edge[1].item(), edge[0].item()])  

    if sym_edges:
        edges_tensor = torch.tensor(sym_edges, dtype=torch.long)
    else:
        edges_tensor = torch.empty((0, 2), dtype=torch.long) 

    return edges_tensor

def process_document(lines):
    result_lines = lines[:2]
    
    abstract_index = -1
    for i in range(2, min(15, len(lines))):
        if "Abstract" in lines[i]:
            abstract_index = i
            logging.info('Find abstract')
            break

    if abstract_index != -1:
        result_lines += lines[abstract_index:]
        abstract_line_in_result = len(result_lines) - len(lines[abstract_index:])
    else:
        result_lines += lines[2:]
        abstract_line_in_result = 0
    
    return result_lines, abstract_line_in_result

def split_by_references(lines):
    references_start = -1
    next_section_start = -1
    
    for i, line in enumerate(lines):
        if line.startswith("## References"):
            references_start = i
            break

    if references_start != -1:
        for j in range(references_start + 1, len(lines)):
            if lines[j].startswith("##"):
                next_section_start = j
                break

    if references_start != -1 and next_section_start != -1:
        references_section = lines[references_start:next_section_start]
        remaining_content = lines[:references_start] + lines[next_section_start:]
    elif references_start != -1:  
        references_section = lines[references_start:]
        remaining_content = lines[:references_start]
    else:
        references_section = []
        remaining_content = lines

    return references_section, remaining_content


def split(text, split_symbol):
    sentences = text.strip().split(split_symbol)
    chunks = [s.strip() for s in sentences if s]
    return chunks

def split_paper(read_path, save_path):
    with open(read_path, 'r') as f:
        text = f.read()
    chunks = split(text, '\n\n')
    return chunks

def process_text(paper_mmd_path, paper_process1_path):
    text = split_paper(paper_mmd_path, paper_process1_path)
    ref, remain = split_by_references(text)
    remain, index = process_document(remain) 
    nlp = spacy.load("en_core_web_sm")
    text = remain[:index+2]

    for i in remain[index+2:]:
        doc = nlp(i)
        sentences = list(doc.sents)
        for sentence in sentences:
            text.append(sentence.text)

    text = [t for t in text if len(t) > 8]

    with open(paper_process1_path, 'w') as f:
        for line in text:
            f.write(line)
            f.write('\n')

def calculate_similarity(chunk1, chunk2, sentence_model):
    embeddings = sentence_model.encode([chunk1,chunk2])
    embeddings = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings

    similarity = cosine_similarity([embeddings[0]],[embeddings[1]])[0][0]
    return similarity

def text2chunk(paper_process1_path, paper_chunk_path, sentence_model, similarity_threshold, length_threshold=750):
    with open(paper_process1_path, 'r') as f:
        text = f.readlines()
    current_main_title = ""
    current_sub_title = ""
    result = []
    new_lines = ['## Title\n']
    text[0] = text[0].lstrip('#').strip() + "\n"
    absract = "## Abstract:\n"

    for i, line in enumerate(text[:5]):
        if "Anonymous authors" in line or "Paper under double-blind review" in line:
            if absract not in new_lines:
                new_lines.append(absract)
            continue
        stripped_line = line.strip("# ").strip()
        if stripped_line.startswith("Abstract") and absract not in new_lines:
            new_lines.append(absract)
        else:
            new_lines.append(line)

    new_lines.extend(text[5:])
    
    buffer = ""
    for line in new_lines:
        line = line.strip()

        if line.startswith("##") and not line.startswith("###"):
            if buffer:
                result.append(f"{buffer} (excerpt from {current_main_title} {current_sub_title})")
                buffer = ""
            current_main_title = line[2:].strip()
            current_sub_title = ""
        
        elif line.startswith("###"):
            if buffer:
                result.append(f"{buffer} (excerpt from {current_main_title} {current_sub_title})")
                buffer = ""
            current_sub_title = line[3:].strip()

        else:
            full_title = f"{current_main_title} {current_sub_title}".strip()
            if buffer:
                similarity = calculate_similarity(buffer, line, sentence_model)
                combined_length = len(buffer) + len(line)

                if similarity > similarity_threshold and combined_length < length_threshold:
                    buffer += " " + line
                else:
                    result.append(f"{buffer} (excerpt from {full_title})")
                    buffer = line
            else:
                buffer = line

    if buffer:
        result.append(f"{buffer} (excerpt from {current_main_title} {current_sub_title})")

    result = [s for s in result if s] 
    with open(paper_chunk_path,'w') as f:
        for r in result:
            f.write(r)
            f.write('\n')

def paper2embeds(read_path, save_path, sentence_model):
    with open(read_path,'r') as f:
        chunks = f.readlines()
    node_embeds = sentence_model.encode(chunks)
    paper_embeds = {
        'text': chunks,
        'embeds': node_embeds
    }
    torch.save(paper_embeds, save_path)


def chunk2graph(text_path, embed_path, save_path, k=10):
    paper_embeds = torch.load(embed_path) 
    edge_subtitle = build_edges_through_subtitle(text_path)
    chunks_sim = torch.mm(paper_embeds['embeds'], paper_embeds['embeds'].T)
    new_edges = get_top_k_edges(chunks_sim, k)
    edge_index = torch.cat((edge_subtitle,new_edges),dim=0)
    data = Data(
        node_text=paper_embeds['text'],
        node_embeds=paper_embeds['embeds'],
        edge_index=edge_index.T,
    )
    torch.save(data, save_path)

if __name__ == '__main__':
    args = init()

    paper_dir = '../result/' + args.filename + '/paper/'
    paper_mmd_path = os.path.join(paper_dir, args.filename + '.mmd')
    paper_process1_path = os.path.join(paper_dir, args.filename + '_process1.txt')
    if not os.path.exists(paper_process1_path):
        logging.info('Step2: Paper text process1')
        process_text(paper_mmd_path, paper_process1_path)
        logging.info('Step2: Paper text process1 done')
    
    
    similarity_threshold = args.similarity_threshold
    paper_chunk_path = os.path.join(paper_dir, args.filename + '_' + str(similarity_threshold) + '.txt')
    if not os.path.exists(paper_chunk_path):
        logging.info('Step2: Paper text chunk')
        sentence_model = SentenceEncoder(args.gpu)
        text2chunk(paper_process1_path, paper_chunk_path, sentence_model, similarity_threshold)
        logging.info('Step2: Paper text chunk done')

    paper_embed_path = os.path.join(paper_dir, 'chunk_embeds.pt')
    if not os.path.exists(paper_embed_path):
        logging.info('Step2: Paper chunk embeds')
        try:
            sentence_model
        except:
            sentence_model = SentenceEncoder(args.gpu)
        paper2embeds(paper_chunk_path, paper_embed_path, sentence_model)
        logging.info('Step2: Paper chunk embeds done')
    
    paper_graph_path = '../result/' + args.filename + '/graph/' + 'paper_graph.pt'
    if not os.path.exists(paper_graph_path):
        logging.info('Step2: Paper graph')
        chunk2graph(paper_chunk_path, paper_embed_path, paper_graph_path)
        logging.info('Step2: Paper graph done')
    
