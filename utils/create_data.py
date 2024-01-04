from datasets import load_dataset
import os

def create_funsd():
    dataset = load_dataset('nielsr/funsd-layoutlmv3')
    train_part = dataset['train']
    test_part = dataset['test']
    train_part.to_csv('../data/funsd_train')
    test_part.to_csv('../data/funsd_test')

def create_cord():
    dataset = load_dataset('nielsr/cord-layoutlmv3')
    train_part = dataset['train']
    test_part = dataset['test']
    train_part.to_csv('../data/cord_train')
    test_part.to_csv('../data/cord_test')

if __name__ == "__main__":
    create_cord()