# 3_Language_translation_with_LSTM
# In this experiment, we implement a basic German-English translator using most basic usage of LSTM.
# The encoder encodes the sentence into a context and this context is provided to decoder as its
# initial state.
# Following classes are inside the external Python module base_translation:
# + Seq2Seq(torch.nn.Module): Contains encoder and decoder, handles target sequence traversing, teacher forcing etc.
# + TranslationExperiment(pl.LightningModule): Contains Seq2Seq, loss function, optimizer etc. Handles learning and
#   sentence translation.
# + TranslationDataSet(data.Dataset): Downloads dataset, tokenize them, converts tokens to tensors.
# + TranslationDataModule(pl.LightningDataModule): Handles batching, collating and data loading.

from torch import nn
import base
import base_translation


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.nn_embedding = nn.Embedding(input_dim, emb_dim)
        # batch_first = False by default
        self.nn_rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    # src = [src len, batch size]
    def forward(self, src):
        embedded = self.nn_embedding(src)  # [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.nn_rnn(embedded)
        # outputs = [src len, batch size, hid dim (*2 when bidirectional)]
        # hidden = [n layers (*2 when bidirectional), batch size, hid dim]
        # cell = [n layers (*2 when bidirectional), batch size, hid dim]

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
        # the fully connected layer at the end will result a single word.
        self.nn_fc_out = nn.Linear(hid_dim, output_dim)

    # trg = [batch size]
    # hidden = [n layers (*2 when bidirectional), batch size, hid dim]
    # cell = [n layers (*2 when bidirectional), batch size, hid dim]
    def forward(self, trg, hidden_cell):
        # Even though in diagrams we see a sequence in the Decoder,
        # this sequence is implemented in Seq2Seq for-loop logic as the input here is only one starting word
        # or the next word. So there's no sequence. But the NN expects seq_len, batch_size, emb_size
        trg = trg.unsqueeze(0)  # [1, batch size]
        embedded = self.nn_embedding(trg)  # [1, batch size, emb dim]

        output, hidden_cell = self.nn_rnn(embedded, hidden_cell)
        # output = [1, batch size, hid dim (*2 when bidirectional)]
        # hidden = [n layers (*2 when bidirectional), batch size, hid dim]
        # cell = [n layers (*2 when bidirectional), batch size, hid dim]
        # seq len and n directions will always be 1 in the decoder

        prediction = self.nn_fc_out(output.squeeze(0))  # [batch size, output dim]
        return prediction, hidden_cell


params = base.init_env("3/params.yml")
p = params['data']
data_module = base_translation.TranslationDataModule(batch_size=p['batch_size'],
                                    src_lang=p['src_lang'],
                                    trg_lang=p['trg_lang'],
                                    max_tokens=p['max_tokens'])
p = params['model']
model = base_translation.Seq2Seq( 
    Encoder(input_dim=data_module.input_dim, emb_dim=p['emb_dim'],
            hid_dim=p['hid_dim'], n_layers=p['n_layers'], dropout=p['dropout']),
    Decoder(output_dim=data_module.output_dim, emb_dim=p['emb_dim'],
            hid_dim=p['hid_dim'], n_layers=p['n_layers'], dropout=p['dropout']),
    specials={'bos': data_module.bos_idx,
              'eos': data_module.eos_idx,
              'pad': data_module.pad_idx})

pl_app = base.PlApp(data_module=data_module, model=model, cls_experiment=base_translation.TranslationExperiment,
                    params=params)
pl_app.train()

'''sentence = "Sehr gut"
pl_app.experiment.eval()
with torch.no_grad():
    translation = pl_app.experiment.forward([sentence])[0]'''
