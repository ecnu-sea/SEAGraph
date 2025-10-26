import os
from utils import *
from args import params
import subprocess
import logging


def parse_pdf(file, output_dir, device, verbose=True):
    """
    Parse a PDF file using the Nougat tool.
    
    Args:
        file (str): Path to the PDF file to be parsed.
        output_dir (str): Directory where the parsed output will be saved.
        ifrm (bool): If True, delete the PDF file after parsing.
    """
    # nougat ../test.pdf -o ./ -m 0.1.0-base
    # result = subprocess.run(["conda", "deactivate"], capture_output=True, text=True)
    # command_parse = ["CUDA_VISIBLE_DEVICES="+device_num, "nougat", file, "-o", output_dir, "-m", "0.1.0-base", "--no-skipping", "--batchsize","8"]
    assert os.path.exists(file), "Input file does not exist: %s" % file
    assert os.path.exists(output_dir), "Output directory does not exist: %s" % output_dir

    command_parse = ["nougat", file, "-o", output_dir, "-m", "0.1.0-base", "--no-skipping", "--batchsize","8"]
    # run 
    if verbose:
        logging.info("Parsing PDF file:")
        logging.info("Input file: %s", file)
        logging.info("Output dir: %s", output_dir)
    try:
        result = subprocess.run(command_parse, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(device)})
        if result.returncode != 0:
            logging.error("Parse PDF file failed! Command failed with exit code: %d", result.returncode)
            logging.error("Parse PDF file failed! err: %s", result.stderr)
            raise Exception("Parse PDF file failed!")
    except FileNotFoundError as e:
        logging.error("Parse PDF file failed! The command 'nougat' was not found: %s", e)
        raise Exception("Command not found! Parse PDF file failed!")
    

if __name__ == "__main__":
    args = init()
    logging.info("Step1: Parse paper pdf...")
    if not os.path.exists('../result/' + args.filename + '/paper/' + args.filename + '.mmd'):
        parse_pdf('../data/paper_pdf/' + args.filename + '.pdf', '../result/' + args.filename + '/paper/', args.gpu)
    logging.info("Step1: Parse paper pdf success!")