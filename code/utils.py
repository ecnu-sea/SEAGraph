import json
import os
import torch
import subprocess
from tqdm import tqdm
import tiktoken
import logging
from transformers import logging as hf_logging
from args import params


def read_txt_file(path):
    with open(path, 'r') as f:
        content = f.read()
    return content

def read_json_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_output(output, save_dir, data_id):
    with open(save_dir + data_id + ".txt", 'w') as f:
        f.write(output)
    print(f"The summary review of paper {data_id} has been saved.")
    f.close()

def get_subfile(path):
    subfiles = [d for d in os.listdir(path) if os.path.isfile(os.path.join(path, d))]
    return subfiles

def save_txt_file(text, save_path):
    with open(save_path, 'w') as f:
        f.write(text)
    f.close()

def save_json_file(dict, save_path):
    with open(save_path, 'w') as f:
        json.dump(dict, f, indent=4)
    f.close()

def num_tokens_from_string(string: str) -> int:
    """
    Calculates the number of tokens in a given text string according to a specific encoding.

    Args:
        text (str): The text string to be tokenized.

    Returns:
        int: The number of tokens the string is encoded into according to the model's tokenizer.
    """
    encoding = tiktoken.encoding_for_model('gpt-4-1106-preview')
    num_tokens = len(encoding.encode(string))
    return num_tokens

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def log_init(filename):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    hf_logging.set_verbosity_error()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  
    log_filename = f"../result/{filename}/{filename}.log"
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)  
    file_handler.setFormatter(formatter)


    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    console_handler.setFormatter(formatter)


def init():
    args = params()
    assert args.filename != ''
    make_dir(os.path.join('../result', args.filename))
    make_dir(os.path.join('../result', args.filename, 'paper'))
    make_dir(os.path.join('../result', args.filename, 'graph'))
    make_dir(os.path.join('../result', args.filename, 'review'))
    
    log_init(args.filename)
    return args