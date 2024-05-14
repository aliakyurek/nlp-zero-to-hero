# Language_translation_with_GRU_and_Bahdanau
# In the previous experiment, the decoder used the compressed hidden state of encoder (so called context), but we want
# decoder to access non-compressed source sentence.
# In this one, we implement a basic attention mechanism called Bahdanau (or concat) attention as we want decoder to use
# hidden state of encoder not only once at the beginning of decoder, but also for each iteration in decoder.
# Using this attention tell us how much we should attend to each token in the source sequence.
# In encoder:
#   we use 1 layer birectional GRU that results hid_dim*2 output but reduce it to hid_dim using a FC linear.
#   we concat GRU output and previous (embedding+context) and feed this to the FC
# https://www.youtube.com/watch?v=mDZil99CtSU

import torch
from torch import nn
import torch.nn.functional as F
import base
import base_translation


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, bidirectional):
        super().__init__()
        self.nn_embedding = nn.Embedding(input_dim, emb_dim)

        # actually 1 layer will be used in this model, but birectional will be enabled
        self.nn_rnn = nn.GRU(emb_dim, hid_dim, bidirectional=bidirectional)

        # since there are both forward and backward hidden states, reduce them to one hidden state
        self.nn_fc = nn.Linear(hid_dim*2, hid_dim)

    # src = [src len, batch size]
    def forward(self, src):
        embedded = self.nn_embedding(src) # [src len, batch size, emb dim]

        # As we use bidirectional single layer GRU, in the output&hidden.
        outputs, hidden = self.nn_rnn(embedded)
        # outputs = [src len, batch size, hid_dim (*2 when bidirectional)]
        # In the 3rd axis, [:hid_dim] is result of forward,[hid_dim:] is result of backward
        # hidden = [n layers (*2 when bidirectional), batch size, hid dim]
        # hidden[0] and hidden[1] will be forward and backward hidden states contexts

        # to create context vector, we need to harmonize these multiple hidden states as the decoder accepts
        # a single hidden state as the initial state.
        # First permute hidden as [batch size, n layers (*2 when bidirectional), hid dim] and then combine forward
        # and backward hidden states in the last dimension. the result is [batch,hid dim*2]
        hidden = hidden.permute(1,0,2).flatten(1)

        hidden = torch.tanh(self.nn_fc(hidden)) # [batch,hid dim*2] -> [batch,hid dim] -> tanh activation to squash to [-1,1] again.
        hidden = hidden.unsqueeze(0)
        return {'enc_outputs':outputs, 'context':hidden} # [src_len, batch_size, hid_dim*2], [batch_size, hid_dim]


class Attention(nn.Module):
    # Intuitively, this layer takes what we have decoded so far and all of what we have encoded to produce a vector
    # that represents which words in the source sentence we should pay the most attention to in order to correctly
    # predict the next word to decode
    # Do not confuse attention in this model with self(dot) attention. Here the concat attention used.
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

        # we calculate the energy between the previous decoder hidden state and the encoder hidden state. This can be
        # thought of as calculating how well each encoder output "matches" the previous decoder hidden state
        # To do this we need to repeat decoder hidden state src_len times
        # but first we need to add repeating dimension.
        dec_hidden = dec_hidden.permute(1,0,2).repeat(1, src_len, 1) # [batch_size, src_len, hid_dim]

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

    # trg = [batch size]
    # hidden = [1, batch size, hid dim]
    # encoder_outputs = [src_len, batch size, hid dim*2]
    def forward(self, trg, dec_input):
        encoder_outputs = dec_input['enc_outputs']
        context = dec_input['context']
        hidden = dec_input.get('hidden',context)

        # Add sequence dimension
        trg = trg.unsqueeze(0) # [1, batch size]

        embedded = self.nn_embedding(trg) # [1, batch size, emb dim]

        weighted_context = self.nn_attention(hidden, encoder_outputs) # [1, batch_size, hid_dim*2]

        # we feed embeddings and weighted context to the RNN
        rnn_input = torch.cat((embedded, weighted_context), dim=-1) # [1, batch_size, hid_dim*2 + emb_dim]

        _, hidden = self.nn_rnn(rnn_input, hidden)
        # _ (output) = [1, batch size, hid dim] Here 1 is due to only one element in sequence
        # hidden = [1, batch size, hid dim] Here 1 is due to one layer non-bidir GRU

        # we feed embeddings [1, batch_size, emb_dim], hidden state [1, batch_size, hid_dim], and weighted attn [1, batch_size, hid_dim*2] to the FC
        fc_input = torch.cat((rnn_input, hidden), dim=-1) # [1, batch_size, emb_dim + hid_dim*2 + hid_dim]

        prediction = self.nn_fc_out(fc_input.squeeze(0)) # [batch_size, trg_vocab_size]
        dec_input['hidden'] = hidden
        return prediction, dec_input


params = base.init_env("5/params.yml")
p = params['data']
data_module = base_translation.TranslationDataModule(batch_size=p['batch_size'],
                                    src_lang=p['src_lang'],
                                    trg_lang=p['trg_lang'],
                                    max_tokens=p['max_tokens'])
p = params['model']

attn = Attention(hid_dim=p['hid_dim'])
enc = Encoder(input_dim=data_module.input_dim, emb_dim=p['emb_dim'],
                hid_dim=p['hid_dim'], bidirectional=p['bidirectional'])

dec = Decoder(output_dim=data_module.output_dim, emb_dim=p['emb_dim'],
                hid_dim=p['hid_dim'], attention=attn)
model = base_translation.Seq2Seq(enc, dec, specials={'bos':data_module.bos_idx,
                                             'eos':data_module.eos_idx,
                                             'pad':data_module.pad_idx})

pl_app = base.PlApp(data_module=data_module, model=model, cls_experiment=base_translation.TranslationExperiment,
                         params=params)
pl_app.train()

'''sentence = "Sehr gut"
pl_app.experiment.eval()
with torch.no_grad():
    translation = pl_app.experiment.forward([sentence])[0]'''