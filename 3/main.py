# IMPORTS
from torch import nn
import base
from base_translation import *

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.nn_embedding = nn.Embedding(input_dim, emb_dim)
        self.nn_rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    # src = [src len, batch size]
    def forward(self, src):
        embedded = self.nn_embedding(src)  # [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.nn_rnn(embedded)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # I just need the hidden and cell states. This accumulated hidden and cell state will represent
        # the input sentence somehow.
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.nn_embedding = nn.Embedding(output_dim, emb_dim)

        # Actually since we decode one by one, we could use LSTMCell here
        self.nn_rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        # the fully connected layer at the end will results a single word.
        self.nn_fc_out = nn.Linear(hid_dim, output_dim)

    # input = [batch size]
    # hidden = [n layers * n directions, batch size, hid dim]
    # cell = [n layers * n directions, batch size, hid dim]
    def forward(self, input, hidden, cell):
        # Even though in diagrams we see a sequence in the Decoder,
        # this sequence is implemented in Seq2Seq logic as the input here is only one starting word
        # or the next word
        input = input.unsqueeze(0) # [1, batch size]
        embedded = self.nn_embedding(input)# [1, batch size, emb dim]

        output, (hidden, cell) = self.nn_rnn(embedded, (hidden, cell))
        # output = [1, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # seq len and n directions will always be 1 in the decoder

        prediction = self.nn_fc_out(output.squeeze(0)) # [batch size, output dim]
        return prediction, (hidden, cell)


class Seq2Seq(nn.Module):
    # specials are expected in bos, eos, pad order
    def __init__(self, encoder, decoder, specials):
        super().__init__()

        self.m_encoder = encoder
        self.m_decoder = decoder
        self.specials = specials
        self.nn_loss = nn.CrossEntropyLoss(ignore_index=specials['pad'])

        for m in self.modules():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)

    # src [src len, batch size]
    # trg [trg len, batch size]
    # teacher_forcing_ratio is probability to use teacher forcing
    # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
    def forward(self, src, trg=None, teacher_forcing_ratio=0.75):
        inference = trg is None
        batch_size = src.shape[1] # 256
        trg_len = trg.shape[0] if not inference else 1000
        trg_vocab_size = self.m_decoder.output_dim # 5000

        # tensor to store decoder outputs,trg_len-1 is used since we don't need bos in the output.
        outputs = torch.zeros(trg_len-1, batch_size, trg_vocab_size).to(src.device) # [trg_len-1, batch_size, output_dim]

        # hidden state and cell state of the encoder are to be used as the initial states of the decoder
        hidden, cell = self.m_encoder(src) # [num layers, batch size, hidden dim]

        # get the first input to the decoder (<bos> tokens) for all batches
        input = src[0, :] # [batch size]

        # When training/testing our model, we always know how many words are in our target sentence, so we stop
        # generating words once we hit that many. During inference it is common to keep generating words until the
        # model outputs an <eos> token or after a certain amount of words have been generated.
        # here we can't break the loop etc. based on eos, since we operate on batches.
        for t in range(trg_len-1):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, (hidden, cell) = self.m_decoder(input, hidden, cell) # output [batch_size,output_dim], [num layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1) # [256]

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t+1] if teacher_force else top1
            if inference:
                if input.item() == self.specials['eos']:
                    return outputs[:t,...]

        # if we are in inferences mode and output not generated so far, return None
        if inference:
            return None
        return outputs

class TranslationExperiment(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    # This function is really intended to take sentences to translate
    def forward(self, X):
        results = []
        for s in X:
            # tensorize and append&prepend bos and eos
            t = TranslationDataSet.tensorize(s, "de").to(self.device)

            # add dummy batch dimension and make teacher_forcing_ratio=0 as we'll use always what's predicted before.
            outs = self.model(t.unsqueeze(dim=1), teacher_forcing_ratio=0.)
            if outs is not None:
                word_ids = outs.argmax(-1).squeeze(dim=1).tolist()
                translation = " ".join(TranslationDataSet.token_to_int['en'].lookup_tokens(word_ids))
            else:
                translation = "No translation"
            results.append(translation)
        return results


    def step(self, batch, batch_idx):
        src, trg, _ = batch  # src=[de_padded_seq, batch_size], trg=[en_padded_seq,batch_size]
        output = self.model(src, trg)  # [trg_len-1, batch_size, vocab_size]

        # Remember that target (or ground truth in other words could be either one hot encoding or class index)
        # In our setup, it's class index, there for it has one less dimension than OHEd input. Therefore,
        # we flatten sequence and batch dimension.
        # ignore the bos in target using trg[1:]
        output = output.flatten(0, 1)  # [trg_len-1*batch_size, trg_vocab_size]
        trg = trg[1:].flatten() # [trg_len-1*batch_size]
        loss = self.model.nn_loss(output, trg)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        # we need to specify batch size explicitly, since pl doesn't know that first rank is seqlen and it can
        # have different values for src and trg
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True, on_step=True, batch_size=batch[0].shape[1])
        self.log("train_ppl", loss.exp().item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=batch[0].shape[1])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False, batch_size=batch[0].shape[1])
        return loss

    def on_validation_epoch_end(self):
        sentence = "Eine Frau spielt ein Lied."
        # pl automatically sets model to eval mode and disables grad
        translation = self([sentence])[0]
        self.logger.experiment.add_text("Translation",f"{sentence}->{translation} | Loss:{self.trainer.logged_metrics['val_loss'].item():.3f}",
                                        # global_step=self.global_step)
                                        global_step = self.current_epoch )


params = base.init_env("3/params.yml")
p = params['data']
data_module = TranslationDataModule(batch_size=p['batch_size'],
                                    src_lang=p['src_lang'],
                                    trg_lang=p['trg_lang'],
                                    max_tokens=p['max_tokens'])
p = params['model']
model = Seq2Seq(
    Encoder(input_dim=data_module.input_dim, emb_dim=p['emb_dim'],
            hid_dim=p['hid_dim'], n_layers=p['n_layers'], dropout=p['dropout']),
    Decoder(output_dim=data_module.output_dim, emb_dim=p['emb_dim'],
            hid_dim=p['hid_dim'], n_layers=p['n_layers'], dropout=p['dropout']),
            specials={'bos':data_module.bos_idx,
                      'eos':data_module.eos_idx,
                      'pad':data_module.pad_idx})

pl_app = base.PlApp(data_module=data_module, model=model, cls_experiment=TranslationExperiment,
                         params=params)
pl_app.train()

'''sentence = "Sehr gut"
pl_app.experiment.eval()
with torch.no_grad():
    translation = pl_app.experiment.forward([sentence])[0]'''












