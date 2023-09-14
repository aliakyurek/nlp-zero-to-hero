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
        self.rnn = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, input_output_size)

    def forward(self, character, hidden_state, cell_state):
        # character = 1
        # embed layer accepts any shaped output. It just adds a new dimension containing embedded vector.
        embedded = self.embed(character)
        # [embedding_dim]

        # LSTMCell layer accepts [embedding_dim].
        hidden_state,cell_state = self.rnn(embedded, (hidden_state, cell_state))
        output = self.fc(hidden_state)
        # hidden_state = [hidden_dim]
        # cell_state = [hidden_dim]
        # output = [input_output_size]

        return output, hidden_state, cell_state

    def get_zero_states(self):
        zero_hidden = torch.zeros(self.rnn.hidden_size).type_as(self.fc.weight)
        zero_cell = torch.zeros(self.rnn.hidden_size).type_as(self.fc.weight)
        return (zero_hidden,zero_cell)

    def loss_func(self, outputs, targets):
        return F.cross_entropy(outputs,targets)

class TextGenerationExperiment(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def generate(self, prime_input, predict_len=100, temperature=0.8):
        self.model.eval()
        generations = []
        hidden, cell_state = self.model.get_zero_states()
        prime_tensor = CharacterDataSet.char_to_tensor(prime_input).to(hidden.device)
        for t in prime_tensor:
            generations.append(t.item())
            with torch.no_grad():
                output, hidden, cell_state = self.model(t, hidden, cell_state)

        for _ in range(predict_len):
            # Sample from the network as a multinomial distribution
            output_dist = output.data.div(temperature).exp()  # e^{logits / T}
            t = torch.multinomial(output_dist, 1)[0]
            generations.append(t.item())
            with torch.no_grad():
                output, hidden, cell_state = self.model(t, hidden, cell_state)
        gen_text = ''.join([string.printable[t] for t in generations])
        self.model.train()
        return gen_text

    def on_train_start(self) -> None:
        pass
        # with torch.no_grad():
            # hidden_state, cell_state = self.model.get_zero_states()
            # self.logger.experiment.add_graph(self.model, (torch.tensor(1).to(hidden_state.device),hidden_state,cell_state))

    def training_step(self, batch, batch_idx):
        loss = 0
        inputs, targets = batch
        hidden_state, cell_state = self.model.get_zero_states()

        for c in range(len(inputs)):
            output, hidden_state, cell_state = self.model(inputs[c], hidden_state, cell_state)
            loss += self.model.loss_func(output, targets[c])

        loss /= len(inputs)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True) # on_step=True, on_epoch=True
        return loss

    def on_train_epoch_end(self):
        gen_text = self.generate("Th", predict_len=100, temperature=0.75)

        self.logger.experiment.add_text(tag="Generated", text_string=gen_text, global_step=self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


params = base.init_env("1/params.yml")
p = params['data']
data_module = CharacterDataModule(p['file_path'])
p = params['model']
model = LSTMGenerator(input_output_size=len(string.printable), **p)

pl_app = base.PlApp(data_module=data_module, model=model, cls_experiment=TextGenerationExperiment,
                    params=params)
pl_app.train()

