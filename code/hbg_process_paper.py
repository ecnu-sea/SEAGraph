import os
from hbg_utils import *
from smg import *


if __name__ == '__main__':
    args = init()
    bkg_paper_list = os.listdir('../result/' + args.filename + '/paper/background/mmd/')
    bkg_paper_dir = '../result/' + args.filename + '/paper/background/'
    
    logging.info('Step7: Background paper text process1')
    bkg_paper_mmd_dir = os.path.join(bkg_paper_dir, 'mmd')
    bkg_paper_process1_dir = os.path.join(bkg_paper_dir, 'process1')
    if not os.path.exists(bkg_paper_process1_dir):
        os.makedirs(bkg_paper_process1_dir)
    for paper in tqdm(bkg_paper_list):
        filename = paper.split('.mmd')[0]
        paper_mmd_path = os.path.join(bkg_paper_mmd_dir, paper)
        paper_process1_path = os.path.join(bkg_paper_process1_dir, filename + '_process1.txt')
        if not os.path.exists(paper_process1_path):
            process_text(paper_mmd_path, paper_process1_path)
    logging.info('Step7: Background paper text process1 done')

    logging.info('Step7: Background paper text chunk')
    similarity_threshold = args.similarity_threshold
    sentence_model = SentenceEncoder(args.gpu)
    bkg_paper_chunk_sim_dir = os.path.join(bkg_paper_dir, 'chunk_sim_' + str(similarity_threshold))
    if not os.path.exists(bkg_paper_chunk_sim_dir):
        os.makedirs(bkg_paper_chunk_sim_dir)
    for paper in tqdm(bkg_paper_list):
        filename = paper.split('.mmd')[0]
        paper_process1_path = os.path.join(bkg_paper_process1_dir, filename + '_process1.txt')
        paper_chunk_path = os.path.join(bkg_paper_chunk_sim_dir, filename + '_' + str(similarity_threshold) + '.txt')
        if not os.path.exists(paper_chunk_path):
            text2chunk(paper_process1_path, paper_chunk_path, sentence_model, args.similarity_threshold)
    logging.info('Step7: Background paper text chunk done')
    
    logging.info('Step7 Background paper chunk embeds')
    bkg_paper_embed_dir = os.path.join(bkg_paper_dir, 'embeds_' + str(similarity_threshold))
    if not os.path.exists(bkg_paper_embed_dir):
        os.makedirs(bkg_paper_embed_dir)
    try:
        sentence_model
    except:
        sentence_model = SentenceEncoder(args.gpu)
    for paper in tqdm(bkg_paper_list):
        filename = paper.split('.mmd')[0]
        paper_chunk_path = os.path.join(bkg_paper_chunk_sim_dir, filename + '_' + str(similarity_threshold) + '.txt')
        paper_embed_path = os.path.join(bkg_paper_embed_dir, filename + '_' + str(similarity_threshold) + '.pt')
        if not os.path.exists(paper_embed_path):
            paper2embeds(paper_chunk_path, paper_embed_path, sentence_model)
    logging.info('Step7: Background paper chunk embeds done')
    
    logging.info('Step7: Background paper graph')
    bkg_paper_graph_dir = '../result/' + args.filename + '/graph/background/'
    if not os.path.exists(bkg_paper_graph_dir):
        os.makedirs(bkg_paper_graph_dir)
    for paper in tqdm(bkg_paper_list):
        filename = paper.split('.mmd')[0]
        paper_chunk_path = os.path.join(bkg_paper_chunk_sim_dir, filename + '_' + str(similarity_threshold) + '.txt')
        paper_embed_path = os.path.join(bkg_paper_embed_dir, filename + '_' + str(similarity_threshold) + '.pt')
        paper_graph_path = os.path.join(bkg_paper_graph_dir, filename + '_' + str(similarity_threshold) + '_graph.pt')
        if not os.path.exists(paper_graph_path):
            chunk2graph(paper_chunk_path, paper_embed_path, paper_graph_path)
    logging.info('Step7: Background paper graph done')
