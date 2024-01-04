from utils.dataloader import get_dataloader
from tqdm import tqdm
import argparse

def train(config):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='doc_ner', choices=['doc_ner', 'ar_sep', 'doc_class'])
    parser.add_argument("--type", default='unmasked', choices=['masked', 'unmasked'])
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--max_token_num", default=1024, type=int)
    parser.add_argument("--half", default=True, type=bool)
    parser.add_argument("--sim_dim", default=4096, type=int)
    parser.add_argument("--model_name", default='MAGAer13/mplug-owl-llama-7b')
    parser.add_argument("--dataset_name", default='nielsr/funsd-layoutlmv3')
    args = parser.parse_args()
    print(args)
    config = vars(args)
    train(config)




