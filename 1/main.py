# 1_Character_Generation_with_LSTMCell
# In this experiment, we implement a recurrent neural network (specifically LSTMCell) based character generator using a covid-19 dataset.
# RNNCell or LSTMCell is the most basic element of sequential data processing where you iterate the input data and update the hidden state
# manually.
# Following classes are inside the external Python module base_text:
# + CharacterDataSet(data.IterableDataset): Parses and processes dataset.
# + CharacterDataModule(pl.LightningDataModule): Handles batching, data loading, utilizes CharacterDataSet.


import sys
import os
sys.path.append(os.getcwd())
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import base
import base_text
    
class LSTMGenerator(nn.Module):
    def __init__(self, input_output_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=input_output_size, embedding_dim=embed_size)
        self.rnn = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, input_output_size)

    # character index = scalar
    def forward(self, character, hc_states=None):
        # embed layer accepts any shaped output. It just adds a new dimension containing embedded vector.
        embedded = self.embed(character)  # [embedding_dim]

        # LSTMCell layer accepts [embedding_dim] and optional initial hidden and cell states
        hidden_state, cell_state = self.rnn(embedded, hc_states)  # both [hidden_dim]
        output = self.fc(hidden_state)  # [input_output_size]

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
        prime_tensor = base_text.CharacterDataSet.char_to_tensor(prime_input).to(self.device)
        generations = [t.item() for t in prime_tensor]
        hc_states = None
        output = None
        for t in prime_tensor:
            output, hc_states = self.model(t, hc_states)
            # output [input_output_size], hc_states ([hidden_dim],[hidden_dim])

        for _ in range(predict_len):
            # Sample from the network as a multinomial distribution.
            # output_dist not have to sum to one, but it should be non-negative, exp ensures that they are positive,
            # and it keeps the right order. Temperature determines how sharp the probability distribution be for the
            # maximum one. Because when the outputs divided by a floating number 0..1, their value will be bigger
            # also considering the exponentiation.
            output_dist = output.div(temperature).exp()  # [input_output_size] # e^{logits / T}
            # multinomial keeps dimension, so use [0] to get scalar tensor
            t = torch.multinomial(output_dist, 1)[0]  # t:scalar, 1: sample only one
            generations.append(t.item())
            output, hc_states = self.model(t, hc_states)
        gen_text = ''.join([base_text.CharacterDataSet.vocab[t] for t in generations])
        return gen_text

    def training_step(self, batch, batch_idx):
        loss = 0
        inputs, targets = batch
        hc_states = None

        for c in range(len(inputs)):
            # output [input_output_size], hc_states ([hidden_dim],[hidden_dim])
            output, hc_states = self.model(inputs[c], hc_states)
            loss += self.model.loss_func(output, targets[c])

        loss /= len(inputs)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.model.eval()
        with torch.no_grad():
            gen_text = self("The", predict_len=100, temperature=0.25)
        self.model.train()
        self.logger.experiment.add_text(tag="Generated", text_string=gen_text, global_step=self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

params = base.init_env("1/params.yml")
p = params['data']
data_module = base_text.CharacterDataModule(p['file_path'])
p = params['model']
model = LSTMGenerator(input_output_size=len(base_text.CharacterDataSet.vocab), **p)

pl_app = base.PlApp(data_module=data_module, model=model, cls_experiment=TextGenerationExperiment,
                    params=params)
pl_app.train()
