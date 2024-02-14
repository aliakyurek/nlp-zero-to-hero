import string
import re
import random
import unidecode

import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils import data
import torch.nn.functional as F
import base

class CharacterDataSet(data.IterableDataset):
    # handle only lower case characters,digits,space and '.'
    vocab = tuple(string.printable[:36]) + (' ','.')
    def __init__(self, file_path):
        self.portion_size = 200
        self.iters_per_epoch = 32
        with open(file_path,'r', encoding='utf-8') as f:
            file_content = f.read()

        # approximate non-ascii characters to ascii (like Ş to S) and lower chars
        file_content = unidecode.unidecode(file_content).lower()
        # drop extra white spaces
        file_content = re.sub(' +', ' ', file_content)
        # drop special characters.
        self.file_content = re.sub('[^a-z0-9 \.]', '', file_content)

    def __iter__(self):
        # in each epoch, we get random portions from text. So to have a limit, we use iters_per_epoch to stop yielding.
        for i in range(self.iters_per_epoch):
            portion = self.random_portion() # "r second vaccine dose..."
            tokenized = self.char_to_tensor(portion) # tokenize
            inputs = tokenized[:-1]
            targets = tokenized[1:]
            yield (inputs, targets) # target is one char shift of input[-1] "programmin" -> "rogramming"

    def random_portion(self):
        start_index = random.randint(0, len(self.file_content)-self.portion_size)
        end_index = start_index + self.portion_size + 1
        return self.file_content[start_index:end_index]

    @staticmethod
    def char_to_tensor(text):
        lst = [CharacterDataSet.vocab.index(c) for c in text]
        tensor = torch.tensor(lst,dtype=torch.long)
        return tensor

class CharacterDataModule(pl.LightningDataModule):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    # setup is called from every process across all the nodes. Setting state here is recommended.
    def setup(self, stage):
        self.train_dataset = CharacterDataSet(self.file_path)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=None)

class LSTMGenerator(nn.Module):
    def __init__(self, input_output_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=input_output_size, embedding_dim=embed_size)
        self.rnn = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, input_output_size)

    # character index = scalar
    def forward(self, character, hc_states=None):
        # embed layer accepts any shaped output. It just adds a new dimension containing embedded vector.
        embedded = self.embed(character) # [embedding_dim]

        # LSTMCell layer accepts [embedding_dim] and optional initial hidden and cell states
        hidden_state, cell_state = self.rnn(embedded, hc_states) # both [hidden_dim]
        output = self.fc(hidden_state) # [input_output_size]

        return output, (hidden_state, cell_state)

    # logits = [input_output_size], character index = scalar
    def loss_func(self, logits, character):
        # cross entropy expects unnormalized logits as input and class index as target
        return F.cross_entropy(logits, character)

class TextGenerationExperiment(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    # prime_input = text like "the"
    def forward(self, prime_input, predict_len=100, temperature=0.8):
        prime_tensor = CharacterDataSet.char_to_tensor(prime_input).to(self.device)
        generations = [t.item() for t in prime_tensor]
        hc_states = None
        for t in prime_tensor:
            output, hc_states = self.model(t, hc_states) # output [input_output_size], hc_states ([hidden_dim],[hidden_dim])

        for _ in range(predict_len):
            # Sample from the network as a multinomial distribution.
            # output_dist not have to sum to one but it should be non-negative, exp ensures that they are positive
            # and it keeps the right order. Temperature determines how sharp the probability distribution be for the
            # maximum one. Because when the outputs divided by a floating number 0..1, their value will be bigger
            # also considering the exponentiation.
            output_dist = output.div(temperature).exp()  # [input_output_size] # e^{logits / T}
            # multinomial keeps dimension, so use [0] to get scalar tensor
            t = torch.multinomial(output_dist, 1)[0] # scalar
            generations.append(t.item())
            output, hc_states = self.model(t, hc_states)
        gen_text = ''.join([CharacterDataSet.vocab[t] for t in generations])
        return gen_text


    def training_step(self, batch, batch_idx):
        loss = 0
        inputs, targets = batch
        hc_states = None

        for c in range(len(inputs)):
            output, hc_states = self.model(inputs[c], hc_states) # output [input_output_size], hc_states ([hidden_dim],[hidden_dim])
            loss += self.model.loss_func(output, targets[c])

        loss /= len(inputs)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.model.eval()
        with torch.no_grad():
            gen_text = self("the", predict_len=100, temperature=0.75)
        self.model.train()
        self.logger.experiment.add_text(tag="Generated", text_string=gen_text, global_step=self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


params = base.init_env("1/params.yml")
p = params['data']
data_module = CharacterDataModule(p['file_path'])
p = params['model']
model = LSTMGenerator(input_output_size=len(CharacterDataSet.vocab), **p)

pl_app = base.PlApp(data_module=data_module, model=model, cls_experiment=TextGenerationExperiment,
                    params=params)
pl_app.train()

