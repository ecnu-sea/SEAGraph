import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    
def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--filename', type=str, default='')
    parser.add_argument('--similarity_threshold', type=float, default=0.2)
    parser.add_argument('--if_related_search', type=str2bool, default=True)
    parser.add_argument('--if_hot_search', type=str2bool, default=True)

    args, _ = parser.parse_known_args()
    return args