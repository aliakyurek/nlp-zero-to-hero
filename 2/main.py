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
    def __init__(self, file_path):
        self.portion_size = 200
        self.iters_per_step = 32
        with open(file_path,'r', encoding='utf-8') as f:
            file_content = f.read()

        file_content = unidecode.unidecode(file_content)
        self.file_content = re.sub(' +', ' ', file_content)

    def __iter__(self):
        for i in range(self.iters_per_step):
            tokenized = self.char_to_tensor(self.random_portion())
            inputs = tokenized[:-1]
            targets = tokenized[1:]
            yield (inputs, targets)

    def random_portion(self):
        start_index = random.randint(0, len(self.file_content)-self.portion_size)
        end_index = start_index + self.portion_size + 1
        return self.file_content[start_index:end_index]

    @staticmethod
    def char_to_tensor(text):
        for c in text:
            if(c not in string.printable):
                print(c)
                quit(0)
        lst = [string.printable.index(c) for c in text]
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
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, input_output_size)

    # characters [portion_size]
    # hc_states ([1,hidden_dim],[1,hidden_dim])
    def forward(self, characters, hc_states=None):
        # characters = [portion_size]
        # embed layer accepts any shaped output. It just adds a new dimension containing embedded vector.
        embedded = self.embed(characters)  # [portion_size,embedding_dim]

        # LSTM layer accepts [portion_size,embedding_dim].
        output, hc_states = self.rnn(embedded, hc_states) # output [portion_size,embedding_dim], hc_states ([1,hidden_dim],[1,hidden_dim])
        output = self.fc(output)
        # output = [1,input_output_size] while inferencing
        return output, hc_states

    def loss_func(self, outputs, targets):
        return F.cross_entropy(outputs,targets)

class TextGenerationExperiment(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    # prime_input = text like "asd"
    def forward(self, prime_input, predict_len=100, temperature=0.8):
        prime_tensor = CharacterDataSet.char_to_tensor(prime_input).to(self.device)
        generations = [t.item() for t in prime_tensor]
        output, hc_states = self.model(prime_tensor) # output [len(prime_input),input_output_size], hc_states ([1,hidden_dim],[1,hidden_dim])

        for _ in range(predict_len):
            # Sample from the network as a multinomial distribution from the last output
            # This gets the last output from model(prime_tensor), also it gets the single and therefore last output in the further execution
            output_dist = output[-1].div(temperature).exp() # [input_output_size] # e^{logits / T}
            t = torch.multinomial(output_dist, 1) # [1]
            generations.append(t.item())
            output, hc_states = self.model(t, hc_states) # output [1,input_output_size], hc_states ([1,hidden_dim],[1,hidden_dim])
        gen_text = ''.join([string.printable[t] for t in generations])
        return gen_text

    def training_step(self, batch, batch_idx):
        loss = 0
        inputs, targets = batch # inputs = [portion_size], targets = [portion_size]

        output, _ = self.model(inputs) # [portion_size, input_output_dim]
        loss = self.model.loss_func(output, targets)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.model.eval()
        with torch.no_grad():
            gen_text = self("Th", predict_len=100, temperature=0.75)
        self.model.train()
        self.logger.experiment.add_text(tag="Generated", text_string=gen_text, global_step=self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


params = base.init_env("2/params.yml")
p = params['data']
data_module = CharacterDataModule(p['file_path'])
p = params['model']
model = LSTMGenerator(input_output_size=len(string.printable), **p)

pl_app = base.PlApp(data_module=data_module, model=model, cls_experiment=TextGenerationExperiment,
                    params=params)
pl_app.train()

