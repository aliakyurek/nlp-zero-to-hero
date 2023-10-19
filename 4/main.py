# IMPORTS
from torch import nn
import base
from base_translation import *

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.nn_embedding = nn.Embedding(input_dim, emb_dim)
        self.nn_rnn = nn.GRU(emb_dim, hid_dim)

    # src = [src len, batch size]
    def forward(self, src):
        embedded = self.nn_embedding(src)  # [src len, batch size, emb dim]

        outputs, hidden = self.nn_rnn(embedded)# [src len, batch size, hid dim], [1, batch size, hid dim]

        # I just need the hidden state. This accumulated hidden state will represent
        # the input sentence somehow.
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.output_dim = output_dim
        self.nn_embedding = nn.Embedding(output_dim, emb_dim)

        self.nn_rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        # the fully connected layer at the end will results a single word.
        self.nn_fc_out = nn.Linear(emb_dim + hid_dim*2, output_dim)

    # input = [batch size]
    # hidden = [1, batch size, hid dim]
    # context = [1, batch size, hid dim]
    def forward(self, input, hidden, context):
        # Even though in diagrams we see a sequence in the Decoder,
        # this sequence is implemented in Seq2Seq logic as the input here is only one starting word
        # or the next word
        input = input.unsqueeze(0) # [1, batch size]
        embedded = self.nn_embedding(input)# [1, batch size, emb dim]

        ###################
        rnn_input = torch.cat((embedded, context), dim=-1) # [1, batch size, emb dim + hid_dim]

        output, hidden  = self.nn_rnn(rnn_input, hidden)
        # output = [1, batch size, hid dim] Here 1 is due to only one element in sequence
        # hidden = [n layers * n directions (1), batch size, hid dim] Here 1 is due to only one layer in GRU

        output = torch.cat((rnn_input, hidden), dim=-1) # [1, batch size, emb_dim + hid dim*2]

        prediction = self.nn_fc_out(output.squeeze(0)) # [batch size, trg_vocab_size]
        return prediction, hidden


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

    # src = [src len, batch size]
    # trt = [trt len, batch size]
    # teacher_forcing_ratio is probability to use teacher forcing
    # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
    def forward(self, src, trg=None, teacher_forcing_ratio=0.75):
        inference = trg is None
        trg_len = trg.shape[0] if not inference else 1000
        # list to store decoder outputs
        outputs = []

        # hidden state of the encoder is to be used as the initial hidden state of the decoder
        context = self.m_encoder(src) # [num_layers, batch_size, hidden_dim]

        hidden = context

        # get the first input to the decoder (<bos> tokens) for all batches
        input = src[0, :] # [batch_size]

        # When training/testing our model, we always know how many words are in our target sentence, so we stop
        # generating words once we hit that many. During inference it is common to keep generating words until the
        # model outputs an <eos> token or after a certain amount of words have been generated.
        # here we can't break the loop etc. based on eos, since we operate on batches.
        for t in range(trg_len-1):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.m_decoder(input, hidden, context) # output [batch_size, output_dim], [num layers, batch size, hidden dim]

            # add predictions to the list
            outputs.append(output)

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input, if not, use predicted token
            if teacher_force:
                input = trg[t + 1] # [batch_size]
            else:
                # get the highest predicted token from our predictions
                input = output.argmax(1)  # [batch_size]

            if inference:
                if input.item() == self.specials['eos']:
                    return torch.stack(outputs[:-1])

        # if we are in inferences mode and output not generated so far, return None
        if inference:
            return None
        return torch.stack(outputs) # [trg_len-1, batch_size, output_dim]

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
        sentence = "Ich liebe dich."
        # pl automatically sets model to eval mode and disables grad
        translation = self([sentence])[0]
        self.logger.experiment.add_text("Translation",f"{sentence}->{translation} | Loss:{self.trainer.logged_metrics['val_loss'].item():.3f}",
                                        # global_step=self.global_step)
                                        global_step = self.current_epoch )


params = base.init_env("4/params.yml")
p = params['data']
data_module = TranslationDataModule(batch_size=p['batch_size'],
                                    src_lang=p['src_lang'],
                                    trg_lang=p['trg_lang'],
                                    max_tokens=p['max_tokens'])
p = params['model']
model = Seq2Seq(
    Encoder(input_dim=data_module.input_dim, emb_dim=p['emb_dim'],
            hid_dim=p['hid_dim']),
    Decoder(output_dim=data_module.output_dim, emb_dim=p['emb_dim'],
            hid_dim=p['hid_dim']),
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












