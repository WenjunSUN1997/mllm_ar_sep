from torch.utils.data import Dataset
from transformers import AutoTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration, MplugOwlModel
import pandas as pd
from model_config.unmasked_mplug_owl import UnMaskedMplugOwlForTokenClassification
from model_config.masked_mplug_owl import MplugOwlForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType
import torch
from collections import Counter
from datasets import load_dataset

class DocNERDataset(Dataset):
    def __init__(self, config):
        super(DocNERDataset, self).__init__()
        model_name = config['model_name']
        self.tokeizer = AutoTokenizer.from_pretrained(model_name)
        self.img_processor = MplugOwlImageProcessor.from_pretrained(model_name)
        self.size = self.img_processor.size.get('shortest_edge', 224)
        self.all_processor = MplugOwlProcessor(self.img_processor, self.tokeizer)
        self.max_token_num = config['max_token_num']
        self.half = config['half']
        self.device = config['device']
        self.config = config
        try:
            self.data = pd.read_csv(config['file_path'])
        except:
            self.data = load_dataset(config['dataset_name'])[config['goal']]
        self.weight = self.calcul_weight()

    def calcul_weight(self):
        label_list = []
        for label in self.data['ner_tags']:
            label_list += label

        element_counts = Counter(label_list)
        total_elements = len(label_list)
        element_percentages = {element: count / total_elements
                               for element, count in element_counts.items()}
        class_weights = {element: 1.0 / percentage
                         for element, percentage in element_percentages.items()}
        return [class_weights[x] for x in range(len(class_weights))]
        # if self.half:
        #     return torch.tensor([class_weights[x] for x in range(len(class_weights))]).bfloat16().to(self.device)
        # else:
        #     return torch.tensor([class_weights[x] for x in range(len(class_weights))]).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        tokens = self.data['tokens'][item]
        img = self.data['image'][item]
        img = [img.resize((self.size, self.size), 3)]
        image_features = self.img_processor(img, return_tensors='pt')['pixel_values']
        label = self.data['ner_tags'][item]
        output_tokenizer = self.tokeizer([tokens],
                                         is_split_into_words=True,
                                         padding="max_length",
                                         max_length=self.max_token_num,
                                         truncation=True,
                                         return_tensors='pt')
        word_ids = torch.tensor([[-100 if element is None else element
                                 for element in output_tokenizer.word_ids()]]).to(self.device)
        label = torch.tensor([label + [-100] * (self.max_token_num - len(label))]).to(self.device)
        output = {'pixel_values': image_features.squeeze(0),
                  'input_ids': output_tokenizer['input_ids'].squeeze(0),
                  'attention_mask': output_tokenizer['attention_mask'].squeeze(0),
                  'word_ids': word_ids.squeeze(0),
                  'label': label.squeeze(0)}
        if self.half:
            output = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in output.items()}

        output = {k: v.to(self.config['device']) for k, v in output.items()}
        return output

def test():
    config = {'model_name': 'MAGAer13/mplug-owl-llama-7b',
              # 'file_path': '../data/funsd_test',
              'dataset_name': 'nielsr/funsd-layoutlmv3',
              'goal': 'train',
              'max_token_num': 1024,
              'half': True,
              'device': 'cpu',
              'sim_dim': 4096}
    datasetor = DocNERDataset(config)
    a = datasetor[0]
    a = {k: v.unsqueeze(0) for k, v in a.items()}
    #
    # image_processor = MplugOwlImageProcessor.from_pretrained('MAGAer13/mplug-owl-llama-7b')
    # tokenizer = AutoTokenizer.from_pretrained('MAGAer13/mplug-owl-llama-7b')
    # processor = MplugOwlProcessor(image_processor=image_processor, tokenizer=tokenizer)
    # generate_kwargs = {
    #     'do_sample': True,
    #     'top_k': 5,
    #     'max_length': 512
    # }
    # from PIL import Image
    # import requests
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # images = [Image.open(requests.get(url, stream=True).raw)]
    # # images = [Image.open(_) for _ in ['18680715_1-0001.jpg']]
    # prompts = [
    #     '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    #     Human: <image>
    #     Human: Explain this image.
    #     AI: ''']
    # inputs = processor(text=prompts, images=images, return_tensors='pt')
    # inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    # inputs = {k: v.to(config['device']) for k, v in inputs.items()}
    #
    # model = MplugOwlForConditionalGeneration.from_pretrained(
    #     'MAGAer13/mplug-owl-llama-7b',
    #     torch_dtype=torch.bfloat16
    # )
    # model.to(config['device'])
    # a = model.generate(**inputs, **generate_kwargs)
    # a = model(**inputs)
    # print()
    peft_config = LoraConfig(inference_mode=False, r=12, lora_alpha=32,
                             lora_dropout=0.1)
    model = UnMaskedMplugOwlForTokenClassification.from_pretrained(
        'MAGAer13/mplug-owl-llama-7b',
        torch_dtype=torch.bfloat16,
        weight_loss=datasetor.weight,
        weight_flag=True
    )
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

    model.language_model = get_peft_model(model.language_model, peft_config)
    model.language_model.print_trainable_parameters()

    model.train()

    res = model(**a)

if __name__ == "__main__":
    test()
