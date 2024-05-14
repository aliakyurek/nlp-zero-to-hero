import random
import re
import string
import torch
from torch.utils import data
from unidecode import unidecode
import pytorch_lightning as pl


class CharacterDataSet(data.IterableDataset):
    # handle only lower case characters,digits,space and '.'
    vocab = tuple(string.printable[:36]) + (' ', '.')

    def __init__(self, file_path):
        self.portion_size = 200
        self.iters_per_epoch = 32
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        # approximate non-ascii characters to ascii (like Ş to S) and lower chars
        file_content = unidecode(file_content).lower()
        # drop extra white spaces
        file_content = re.sub(' +', ' ', file_content)
        # drop special characters.
        self.file_content = re.sub('[^a-z0-9 .]', '', file_content)

    def __iter__(self):
        # in each epoch, we get random portions from text. So to have a limit, we use iters_per_epoch to stop yielding.
        for i in range(self.iters_per_epoch):
            portion = self.random_portion()  # "r second vaccine dose..."
            tokenized = self.char_to_tensor(portion)  # tokenize
            inputs = tokenized[:-1]
            targets = tokenized[1:]
            yield inputs, targets  # target is one char shift of input[-1] "programmin" -> "rogramming"

    def random_portion(self):
        start_index = random.randint(0, len(self.file_content) - self.portion_size)
        end_index = start_index + self.portion_size + 1
        return self.file_content[start_index:end_index]

    @staticmethod
    def char_to_tensor(text):
        lst = [CharacterDataSet.vocab.index(c) for c in text.lower()]
        tensor = torch.tensor(lst, dtype=torch.long)
        return tensor


class CharacterDataModule(pl.LightningDataModule):
    def __init__(self, file_path):
        super().__init__()
        self.train_dataset = None
        self.file_path = file_path

    # setup is called from every process across all the nodes. Setting state here is recommended.
    def setup(self, stage):
        self.train_dataset = CharacterDataSet(self.file_path)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=None)
