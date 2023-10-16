# IMPORTS
from torch import nn
import torch.nn.functional as F
import base
from base_translation import *

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, bidirectional):
        super().__init__()
        self.nn_embedding = nn.Embedding(input_dim, emb_dim)

        # actually 1 layer will be used in this model, but bidirectonal will be enabled
        self.nn_rnn = nn.GRU(emb_dim, hid_dim, bidirectional=bidirectional)

        # since there are both forward and backward hidden states, reduce them to one hidden state
        self.nn_fc = nn.Linear(hid_dim*2, hid_dim)

    # src = [src len, batch size]
    def forward(self, src):
        embedded = self.nn_embedding(src) # [src len, batch size, emb dim]

        # As we use bidirectional single layer GRU, in the output&hidden.

        outputs, hidden = self.nn_rnn(embedded)
        # outputs = [src len, batch size, hid_dim * n directions] In the 3rd axis, [:hid_dim] is result of forward,[hid_dim:] is result of backward
        # hidden = [n layers * n directions, batch size, hid dim]
        # hidden[0] and hidden[1] will be forward and backward hidden states contexts

        # to create context vector, we need to harmonize these multiple hidden states as the decoder accepts
        # a single hidden state.
        # First permute as [batch size, n layers * n directions, hid dim] and then combine forward and backward hidden states
        # in the last dimension. the result is [batch,hid dim*2]
        hidden = hidden.permute(1,0,2).reshape(hidden.shape[1],-1)
        hidden = torch.tanh(self.nn_fc(hidden)) # [batch,hid dim*2] -> [batch,hid dim] -> tanh activation to squash to [-1,1] again.
        return outputs, hidden # [src_len, batch_size, hid_dim*2], [batch_size, hid_dim]


class Attention(nn.Module):
    # Intuitively, this layer takes what we have decoded so far and all of what we have encoded to produce a vector
    # that represents which words in the source sentence we should pay the most attention to in order to correctly
    # predict the next word to decode
    # Do not confuse attention in this model with self(dot) attention. Here the concat attention used
    def __init__(self, hid_dim):
        super().__init__()
        # hid_dim*2 comes from bidirectional encoder outputs
        # hid_dim comes from decoder previous hidden state
        self.nn_attn = nn.Linear(hid_dim*2 + hid_dim, hid_dim)
        # This is necessary before softmaxing. So finally we'll get softmaxed [batch_size, src_len] tensor
        self.nn_v = nn.Linear(hid_dim, 1, bias=False)

    # dec_hidden = [batch size, hid_dim]
    # encoder_outputs = [src len, batch size, hid_dim * 2]
    def forward(self, dec_hidden, encoder_outputs):
        src_len, batch_size, _ = encoder_outputs.shape

        # we calculate the energy between the previous decoder hidden state and the encoder hidden states. This can be
        # thought of as calculating how well each encoder output "matches" the previous decoder hidden state
        # To do this we need to repeat decoder hidden state src_len times
        # but first we need to add repeating dimension.
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1) # [batch_size, src_len, hid_dim]

        # we need to swap batch and sequence dimension
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [batch_size, src_len, hid_dim*2]

        # The probability a_ij , or its associated energy e_ij , reflects the importance of the annotation h_j with
        # respect to the previous hidden state s_i−1 in deciding the next state si and generating yi. Intuitively,
        # this implements a mechanism of attention in the decoder so called concat attention or Bahdanou attention

        energy = torch.tanh(self.nn_attn(torch.cat((dec_hidden, encoder_outputs), #
                                                   dim=-1)))
        # energy = [batch_size, src_len, hid_dim*2 + hid_dim] -> [batch_size, src_len, hid_dim] -> Activation
        out = self.nn_v(energy) # [batch_size, src_len, 1]
        attention = out.squeeze(2) # [batch_size, src_len]

        # With softmax we can now create weights for weighted sum
        weights = F.softmax(attention, dim=1) # [batch_size, src_len]

        weights = weights.unsqueeze(1)  # [batch_size, 1, src_len]

        # we can see this as a way of getting another type of context vector. Because a hidden state like tensor
        # is created at the end, but it's weighted.
        weighted_context = torch.bmm(weights, encoder_outputs)  # [batch_size, 1, hid_dim*2]
        return weighted_context.permute(1, 0, 2)  # [1, batch_size, hid_dim*2]


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.nn_embedding = nn.Embedding(output_dim, emb_dim)
        self.nn_attention = attention

        # hid_dim*2 is for bidirectional encoder weighted outputs
        self.nn_rnn = nn.GRU(hid_dim*2 + emb_dim, hid_dim)
        # the fully connected layer at the end will result a single word.
        # hid_dim*2 is for bidirectional encoder weighted output, hid_dim is for decoder hidden state
        self.nn_fc_out = nn.Linear(hid_dim*2 + hid_dim + emb_dim, output_dim)

    # input = [batch size]
    # hidden = [batch size, hid dim]
    # encoder_outputs = [src_len, batch size, hid dim*2]
    def forward(self, input, hidden, encoder_outputs):
        # Add sequence dimension
        input = input.unsqueeze(0) # [1, batch size]

        embedded = self.nn_embedding(input) # [1, batch size, emb dim]

        weighted_context = self.nn_attention(hidden, encoder_outputs) # [1, batch_size, hid_dim*2]

        # we feed embeddings and weighted context to the RNN
        rnn_input = torch.cat((embedded, weighted_context), dim=-1) # [1, batch_size, hid_dim*2 + emb_dim]

        # Add sequence dimension to hidden and RNN it. BTW output and hidden are same there's only one sequence in the decoder
        _, hidden = self.nn_rnn(rnn_input, hidden.unsqueeze(0))

        # we feed embeddings [1, batch_size, emb_dim], hidden state [1, batch_size, hid_dim], and weighted attn [1, batch_size, hid_dim*2] to the FC
        fc_input = torch.cat((rnn_input, hidden), dim=-1) # [1, batch_size, emb_dim + hid_dim*2 + hid_dim]

        prediction = self.nn_fc_out(fc_input.squeeze(0)) # [batch_size, trg_vocab_size]
        return prediction, hidden.squeeze(0)


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
        batch_size = src.shape[1] # 256
        trg_len = trg.shape[0] if not inference else 1000
        trg_vocab_size = self.m_decoder.output_dim # 5000

        # tensor to store decoder outputs,trg_len-1 is used since we don't need bos in the output.
        outputs = torch.zeros(trg_len-1, batch_size, trg_vocab_size).to(src.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_outputs, hidden = self.m_encoder(src) # enc_outputs [src_len, batch_size, hid_dim*2], [batch_size, hid_dim]

        # get the first input to the decoder (<bos> tokens) for all batches
        input = src[0, :] # [batch_size]

        # When training/testing our model, we always know how many words are in our target sentence, so we stop
        # generating words once we hit that many. During inference it is common to keep generating words until the
        # model outputs an <eos> token or after a certain amount of words have been generated.
        # here we can't break the loop etc. based on eos, since we operate on batches.
        for t in range(trg_len-1):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.m_decoder(input, hidden, enc_outputs) # output [batch_size, trg_vocab_size], [batch_size, hid_dim]

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


params = base.init_env("5/params.yml")
p = params['data']
data_module = TranslationDataModule(batch_size=p['batch_size'],
                                    src_lang=p['src_lang'],
                                    trg_lang=p['trg_lang'],
                                    max_tokens=p['max_tokens'])
p = params['model']

attn = Attention(hid_dim=p['hid_dim'])
enc = Encoder(input_dim=data_module.input_dim, emb_dim=p['emb_dim'],
                hid_dim=p['hid_dim'], bidirectional=p['bidirectional'])

dec = Decoder(output_dim=data_module.output_dim, emb_dim=p['emb_dim'],
                hid_dim=p['hid_dim'], attention=attn)
model = Seq2Seq(enc, dec, specials={'bos':data_module.bos_idx,
                                             'eos':data_module.eos_idx,
                                             'pad':data_module.pad_idx})

pl_app = base.PlApp(data_module=data_module, model=model, cls_experiment=TranslationExperiment,
                         params=params)
pl_app.train()

'''sentence = "Sehr gut"
pl_app.experiment.eval()
with torch.no_grad():
    translation = pl_app.experiment.forward([sentence])[0]'''












