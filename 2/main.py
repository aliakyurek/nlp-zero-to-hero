# 2_Character_Generation_with_LSTM
# In this experiment, we replace LSTMCell with usual LSTM layer, so that we get rid of for loop in our
# LSTMGenerator module. This improves performance since custom for loop in training_step is removed
# and the sequence is internally traversed by fast PyTorch.

import string
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
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, input_output_size)

    # characters [portion_size]
    # hc_states ([1,hidden_dim],[1,hidden_dim])
    def forward(self, characters, hc_states=None):
        # characters = [portion_size]
        # embed layer accepts any shaped output. It just adds a new dimension containing embedded vector.
        embedded = self.embed(characters)  # [portion_size,embedding_dim]

        # LSTM layer accepts either 3D (for batched), or 2D inputs. When the input dimension is two, this is non-batched
        # and [portion_size,embedding_dim] is expected.
        output, hc_states = self.rnn(embedded, hc_states)
        # output [portion_size,embedding_dim], hc_states ([1,hidden_dim],[1,hidden_dim])
        output = self.fc(output)
        # output = [portion_size,input_output_size] while training
        # output = [1,input_output_size] during inference
        return output, hc_states

    def loss_func(self, outputs, targets):
        return F.cross_entropy(outputs, targets)


class TextGenerationExperiment(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    # prime_input = text like "asd"
    def forward(self, prime_input, predict_len=100, temperature=0.8):
        prime_tensor = base_text.CharacterDataSet.char_to_tensor(prime_input).to(self.device)
        generations = [t.item() for t in prime_tensor]
        output, hc_states = self.model(prime_tensor)
        # output [len(prime_input),input_output_size], hc_states ([1,hidden_dim],[1,hidden_dim])

        for _ in range(predict_len):
            # Sample from the network as a multinomial distribution from the last output
            # This gets the last output from model(prime_tensor), also it gets the single and therefore last
            # output in the further execution
            output_dist = output[-1].div(temperature).exp()  # [input_output_size] # e^{logits / T}
            t = torch.multinomial(output_dist, 1)  # t:[1], 1: sample only one
            generations.append(t.item())

            output, hc_states = self.model(t, hc_states)
            # output [1,input_output_size], hc_states ([1,hidden_dim],[1,hidden_dim])
        gen_text = ''.join([string.printable[t] for t in generations])
        return gen_text

    def training_step(self, batch, batch_idx):
        inputs, targets = batch  # inputs = [portion_size], targets = [portion_size]

        output, _ = self.model(inputs)  # [portion_size, input_output_dim]
        loss = self.model.loss_func(output, targets)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.model.eval()
        with torch.no_grad():
            gen_text = self("Th", predict_len=100, temperature=0.25)
        self.model.train()
        self.logger.experiment.add_text(tag="Generated", text_string=gen_text, global_step=self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


params = base.init_env("2/params.yml")
p = params['data']
data_module = base_text.CharacterDataModule(p['file_path'])
p = params['model']
model = LSTMGenerator(input_output_size=len(base_text.CharacterDataSet.vocab), **p)

pl_app = base.PlApp(data_module=data_module, model=model, cls_experiment=TextGenerationExperiment,
                    params=params)
pl_app.train()
