import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def process_label(label):
    result = []
    batch_size = label.shape[0]
    for batch_index in range(batch_size):
        result += label[batch_index][label[batch_index] != -100].cpu().numpy().tolist()

    return result

def validate_doc_ner(model, dataloader_test):
    loss_all = []
    label_all = []
    prediction_all = []
    for step, data in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
        try:
            output = model(**data)
        except:
            continue

        loss = output['loss']
        loss_all.append(loss.item())
        prediction_all += output['prediction']
        label_all += process_label(data['label'])

    p = precision_score(label_all, prediction_all, average='weighted')
    r = recall_score(label_all, prediction_all, average='weighted')
    f1 = f1_score(label_all, prediction_all, average='weighted')
    loss = sum(loss_all) / len(label_all)
    print(f1)
    return {'p': p,
            'r': r,
            'f1': f1,
            'loss': loss}

def validate(config, model, dataloader_test):
    if config['task'] == 'doc_ner':
        return validate_doc_ner(model, dataloader_test)
