from torch.utils.data.dataloader import DataLoader
from utils.datasetor_ner_doc import DocNERDataset

def get_dataloader(config):
    if config['task'] == 'doc_ner':
        datasetor = DocNERDataset(config)

        dataloader = DataLoader(datasetor,
                                batch_size=config['batch_size'],
                                shuffle=False)
        return dataloader
