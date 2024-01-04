from utils.dataloader import get_dataloader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from peft import get_peft_model, LoraConfig, TaskType
import torch
from model_config.unmasked_mplug_owl import UnMaskedMplugOwlForTokenClassification
from model_config.masked_mplug_owl import MplugOwlForTokenClassification
import argparse

def train(config):
    if config['task'] == 'doc_ner':
        config['goal'] = 'train'
        dataloader_train = get_dataloader(config)
        config['goal'] = 'test'
        dataloader_test = get_dataloader(config)
        if config['type'] == 'unmasked':
            model = UnMaskedMplugOwlForTokenClassification.from_pretrained(
                config['model_name'],
                torch_dtype=torch.bfloat16,
                weight_loss=dataloader_train.dataset.weight,
                weight_flag=config['weight'])

        if config['type'] == 'masked':
            model = MplugOwlForTokenClassification.from_pretrained(
                config['model_name'],
                torch_dtype=torch.bfloat16,
                weight_loss=dataloader_train.dataset.weight,
                weight_flag=config['weight'])

    model.to(config['device'])
    for name, param in model.named_parameters():
        if 'vision_model' in name:
            # 默认vision不训练
            param.requires_grad = False
        elif 'language_model' in name:
            # 下面根据language状态进行修改
            param.requires_grad = True
        else:
            param.requires_grad = True

    epoch = 1000
    peft_config = LoraConfig(inference_mode=False, r=12, lora_alpha=32,
                             lora_dropout=0.1)
    model.language_model = get_peft_model(model.language_model, peft_config)
    model.language_model.print_trainable_parameters()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=1,
                                  verbose=True)
    for epoch_index in range(epoch):
        for step, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            output = model(**data)
            loss = output['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='doc_ner', choices=['doc_ner', 'ar_sep', 'doc_class'])
    parser.add_argument("--type", default='unmasked', choices=['masked', 'unmasked'])
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--max_token_num", default=1024, type=int)
    parser.add_argument("--half", default=True, type=bool)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight", default=True, type=bool)
    parser.add_argument("--sim_dim", default=4096, type=int)
    parser.add_argument("--model_name", default='MAGAer13/mplug-owl-llama-7b')
    parser.add_argument("--dataset_name", default='nielsr/funsd-layoutlmv3')
    args = parser.parse_args()
    print(args)
    config = vars(args)
    train(config)




