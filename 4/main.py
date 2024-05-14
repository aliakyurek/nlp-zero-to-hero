# 4_Language_translation_with_GRU
# In this experiment, we replace LSTM with GRU. Also, we want decoder to use hidden state of encoder
# (so called context), not only once at the beginning of decoder but also for each iteration.
# Therefore in decoder:
#   we concat embedding+context and feed this to GRU.
#   we concat GRU output and previous (embedding+context) and feed this to the FC

import torch
from torch import nn
import base
import base_translation

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
        return {'context': hidden}


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.output_dim = output_dim
        self.nn_embedding = nn.Embedding(output_dim, emb_dim)

        self.nn_rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        # the fully connected layer at the end will result a single word.
        self.nn_fc_out = nn.Linear(emb_dim + hid_dim*2, output_dim)

    # trg = [batch size]
    # hidden = [1, batch size, hid dim]
    # context = [1, batch size, hid dim]
    def forward(self, trg, dec_input):
        context = dec_input['context']
        hidden = dec_input.get('hidden',context)

        # Even though in diagrams we see a sequence in the Decoder,
        # this sequence is implemented in Seq2Seq logic as the input here is only one starting word
        # or the next word
        trg = trg.unsqueeze(0) # [1, batch size]
        embedded = self.nn_embedding(trg)# [1, batch size, emb dim]

        ###################
        rnn_input = torch.cat((embedded, context), dim=-1) # [1, batch size, emb dim + hid_dim]

        _, hidden = self.nn_rnn(rnn_input, hidden)
        # _ (output) = [1, batch size, hid dim] Here 1 is due to only one element in sequence
        # hidden = [1, batch size, hid dim] Here 1 is due to one layer non-bidir GRU

        output = torch.cat((rnn_input, hidden), dim=-1) # [1, batch size, emb_dim + hid dim*2]

        prediction = self.nn_fc_out(output.squeeze(0)) # [batch size, trg_vocab_size]
        dec_input['hidden'] = hidden
        return prediction, dec_input


params = base.init_env("4/params.yml")
p = params['data']
data_module = base_translation.TranslationDataModule(batch_size=p['batch_size'],
                                    src_lang=p['src_lang'],
                                    trg_lang=p['trg_lang'],
                                    max_tokens=p['max_tokens'])
p = params['model']
model = base_translation.Seq2Seq(
    Encoder(input_dim=data_module.input_dim, emb_dim=p['emb_dim'],
            hid_dim=p['hid_dim']),
    Decoder(output_dim=data_module.output_dim, emb_dim=p['emb_dim'],
            hid_dim=p['hid_dim']),
            specials={'bos':data_module.bos_idx,
                      'eos':data_module.eos_idx,
                      'pad':data_module.pad_idx})

pl_app = base.PlApp(data_module=data_module, model=model, cls_experiment=base_translation.TranslationExperiment,
                         params=params)
pl_app.train()

'''sentence = "Sehr gut"
pl_app.experiment.eval()
with torch.no_grad():
    translation = pl_app.experiment.forward([sentence])[0]'''
