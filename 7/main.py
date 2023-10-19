# IMPORTS
from torch import nn
import base
from base_translation import *

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_hid_dim, n_layers, n_heads, ff_dim,  dropout, max_seq_length):
        super().__init__()
        self.nn_tok_embedding = nn.Embedding(input_dim, emb_hid_dim)
        self.nn_pos_embedding = nn.Embedding(max_seq_length, emb_hid_dim)
        self.nn_layers = nn.ModuleList([EncoderLayer(emb_hid_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.nn_dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(emb_hid_dim, dtype=torch.float))

    # src = [batch size, src len]
    # src_mask = [batch size, 1, 1, src len]
    def forward(self, src, src_mask):
        batch_size, src_len = src.shape

        pos = torch.arange(src_len).to(src.device) # [src len]
        # Operation:Positional encoding
        pos_emb = self.nn_pos_embedding(pos) # [src len, emb hid dim]
        # Operation:Input embedding
        tokens = self.nn_tok_embedding(src) * self.scale.to(src.device) # [batch size, src len, hid dim]
        # Operation:Embedding+Pos. encoding
        src = self.nn_dropout(tokens + pos_emb) # [batch size, src len, hid dim] !pos_emb broadcasted during operation

        for layer in self.nn_layers:
            src = layer(src, src_mask)
        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, ff_dim, dropout):
        super().__init__()
        self.nn_pwff_layer_norm = nn.LayerNorm(hid_dim)
        self.nn_positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, ff_dim, dropout)
        self.nn_mhsa_layer_norm = nn.LayerNorm(hid_dim)
        self.nn_mh_self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.nn_dropout = nn.Dropout(dropout)

    # src = [batch size, src len, emb hid dim]
    # src_mask = [batch size, 1, 1, src len]
    def forward(self, src, src_mask):
        # Operation:Multi-head attention
        weighted_src, _ = self.nn_mh_self_attention(src, src, src, src_mask)
        # Operation:First Add&Norm
        weighted_normalized_src = self.nn_mhsa_layer_norm(src + self.nn_dropout(weighted_src)) # [batch size, src len, hid dim]
        # Operation:Feed forward
        pwffed_weighted_normalized_src = self.nn_positionwise_feedforward(weighted_normalized_src) # [batch size, src len, hid dim]
        # Operation:Second Add&Norm
        final_out = self.nn_pwff_layer_norm(weighted_normalized_src + self.nn_dropout(pwffed_weighted_normalized_src)) # [batch size, src len, hid dim]
        return final_out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.per_head_dim = hid_dim // n_heads

        # we don't use dedicated linear layers for each dim, instead combine them.
        # we couldn't also combine q,k,v into one linear layer but it'd be more complex
        self.nn_fc_q = nn.Linear(hid_dim, hid_dim)
        self.nn_fc_k = nn.Linear(hid_dim, hid_dim)
        self.nn_fc_v = nn.Linear(hid_dim, hid_dim)

        self.nn_fc_o = nn.Linear(hid_dim, hid_dim)
        self.nn_dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.per_head_dim, dtype=torch.float))

    # query = [batch size, query len, hid dim]
    # key = [batch size, key len, hid dim] in decoder, this will come from encoder during cross attention
    # value = [batch size, value len, hid dim] in decoder, this will come from encoder during cross attention
    # mask = [batch size, 1, 1, src len]
    def forward(self, query, key, value, mask=None):
        batch_size, q_len, _ = query.shape
        _, k_len, _ = key.shape
        _, v_len, _ = value.shape

        Q = self.nn_fc_q(query) # [batch size, query len, hid dim]
        K = self.nn_fc_k(key)  # [batch size, key len, hid dim]
        V = self.nn_fc_v(value) # [batch size, value len, hid dim]

        Q = Q.reshape(batch_size, q_len, self.n_heads, self.per_head_dim).permute(0, 2, 1, 3) # [batch size, n heads, query len, per head dim]
        K = K.reshape(batch_size, k_len, self.n_heads, self.per_head_dim).permute(0, 2, 1, 3) # [batch size, n heads, key len, per head dim]
        V = V.reshape(batch_size, v_len, self.n_heads, self.per_head_dim).permute(0, 2, 1, 3) # [batch size, n heads, value len, per head dim]
        # Operation:Scaled Dot-product attention
        # K.transpose(-2, -1) [batch size, n heads, per head dim, key len]
        # energy is non-normalized attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(Q.device) # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(~mask, float('-inf')) # n_heads and query len dimensions of mask broadcasted

        attention = torch.softmax(energy, dim=-1) # [batch size, n heads, query len, key len] Rows summed to one in the last dimension

        weighted_V = torch.matmul(self.nn_dropout(attention), V) # [batch size, n heads, query len, per head dim]
        # Operation:Concat in MHA
        weighted_V = weighted_V.permute(0, 2, 1, 3) # [batch size, query len, n heads, per head dim]
        weighted_V = weighted_V.reshape(batch_size, -1, self.hid_dim)  # [batch size, query len, emb hid dim]
        # above actually concatenation of all heads are done implicitly using permute and reshape
        # Operation:Linear in MHA
        weighted_V = self.nn_fc_o(weighted_V) # [batch size, query len, emb hid dim]
        return weighted_V, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, ff_dim, dropout):
        super().__init__()
        self.nn_fc_1 = nn.Linear(hid_dim, ff_dim)
        self.nn_fc_2 = nn.Linear(ff_dim, hid_dim)
        self.nn_dropout = nn.Dropout(dropout)

    # x = [batch size, seq len, hid dim]
    def forward(self, x):
        x = self.nn_dropout(torch.relu(self.nn_fc_1(x))) # [batch size, seq len, pf dim]
        x = self.nn_fc_2(x) # [batch size, seq len, hid dim]
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_hid_dim, n_layers, n_heads, ff_dim, dropout, max_seq_length):
        super().__init__()
        self.nn_tok_embedding = nn.Embedding(output_dim, emb_hid_dim)
        self.nn_pos_embedding = nn.Embedding(max_seq_length, emb_hid_dim)
        self.nn_layers = nn.ModuleList([DecoderLayer(emb_hid_dim, n_heads, ff_dim, dropout)
                                        for _ in range(n_layers)])
        self.nn_fc_out = nn.Linear(emb_hid_dim, output_dim)
        self.nn_dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(emb_hid_dim,dtype=torch.float))
        self.max_seq_length = max_seq_length

    # trg = [batch size, trg len]
    # trg_mask = [batch size, 1, trg len, trg len]
    # enc_src = [batch size, src len, emb hid dim]
    # src_mask = [batch size, 1, 1, src len]
    def forward(self, trg, trg_mask, enc_src, src_mask):
        batch_size, trg_len = trg.shape

        pos = torch.arange(trg_len).to(trg.device) # [trg len]
        # Operation:Positional encoding
        pos_emb = self.nn_pos_embedding(pos) # [trg len, emb hid dim]
        # Operation:Output embedding
        tokens = self.nn_tok_embedding(trg) * self.scale.to(trg.device) # [batch_size, trg len, emb hid dim]
        # Operation:Embedding+Pos. encoding
        trg = self.nn_dropout(tokens + pos_emb) # [batch size, trg len, hid dim] !pos_emb broadcasted during operation

        for layer in self.nn_layers:
            trg, attention = layer(trg, trg_mask, enc_src, src_mask)  # attention = [batch size, n heads, trg len, src len]

        output = self.nn_fc_out(trg) # [batch size, trg len, output dim]
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, ff_dim, dropout):
        super().__init__()
        self.nn_pwff_layer_norm = nn.LayerNorm(hid_dim)
        self.nn_positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, ff_dim, dropout)
        self.nn_mhea_layer_norm = nn.LayerNorm(hid_dim)
        self.nn_mh_cross_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.nn_mhsa_layer_norm = nn.LayerNorm(hid_dim)
        self.nn_mh_self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.nn_dropout = nn.Dropout(dropout)

    # trg = [batch size, trg len, hid dim]
    # trg_mask = [batch size, 1, trg len, trg len]
    # enc_src = [batch size, src len, hid dim]
    # src_mask = [batch size, 1, 1, src len]
    def forward(self, trg, trg_mask, enc_src, src_mask):
        # Operation:Multi-head self attention
        weighted_trg, _ = self.nn_mh_self_attention(query=trg, key=trg, value=trg, mask=trg_mask)
        # Operation:First Add&Norm
        weighted_normalized_trg = self.nn_mhsa_layer_norm(trg + self.nn_dropout(weighted_trg)) # [batch size, trg len, hid dim]
        # Operation:Multi-head cross attention
        cross_weighted_trg, attention = self.nn_mh_cross_attention(query=weighted_normalized_trg, key=enc_src, value=enc_src, mask=src_mask)
        # Operation:Second Add&Norm
        cross_weighted_normalized_trg = self.nn_mhea_layer_norm(weighted_normalized_trg + self.nn_dropout(cross_weighted_trg)) # [batch size, trg len, hid dim]
        # Operation:Feed forward
        pwffed_cross_weighted_normalized_trg = self.nn_positionwise_feedforward(cross_weighted_normalized_trg)
        # Operation:Third Add&Norm
        final_out = self.nn_pwff_layer_norm(cross_weighted_normalized_trg +
                                            self.nn_dropout(pwffed_cross_weighted_normalized_trg)) # [batch size, trg len, hid dim]

        # attention = [batch size, n heads, trg len, src len]
        return final_out, attention

class Seq2Seq(nn.Module):
    # specials are expected in bos, eos, pad order
    def __init__(self, encoder, decoder, specials):
        super().__init__()
        self.m_encoder = encoder
        self.m_decoder = decoder
        self.specials = specials
        self.nn_loss = nn.CrossEntropyLoss(ignore_index=specials['pad'])

        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param.data)

    # [batch size, src len]
    def make_src_mask(self, src):
        src_mask = (src != self.specials['pad']).unsqueeze(1).unsqueeze(2) # [batch size, 1, 1, src len]
        return src_mask

    # [batch size, trg len]
    def make_trg_mask(self, trg):
        trg_len = trg.shape[1]
        trg_pad_mask = (trg != self.specials['pad']).unsqueeze(1).unsqueeze(2) # [batch size, 1, 1, trg len]
        trg_sub_mask = torch.tril(torch.full((trg_len, trg_len), True, device=trg.device)) # [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask # [batch size, 1, trg len, trg len]
        return trg_mask

    # src = [batch size, src len]
    # trg = [batch size, trg len]
    def forward(self, src, trg=None):
        inference = trg is None
        if inference:
            trg_indexes = [self.specials['bos']]
            trg_len = self.m_decoder.max_seq_length
        else:
            trg_len = 1 # we need looping only for inference. For training, no loop is necessary

        for _ in range(trg_len):
            if(inference):
                trg = torch.LongTensor(trg_indexes).unsqueeze(0).to(src.device)

            src_mask = self.make_src_mask(src)  # [batch size, 1, 1, src len]
            trg_mask = self.make_trg_mask(trg)  # [batch size, 1, trg len, trg len]

            enc_src = self.m_encoder(src, src_mask)  # [batch size, src len, hid dim]
            output, attention = self.m_decoder(trg, trg_mask, enc_src, src_mask) # output [1,n,5000] during inference, else [batch size, trg len, output_dim]

            if inference:
                pred_token = output[:, -1].argmax(-1).item()
                if(pred_token == self.specials['eos']):
                    # [1, n, output_dim]: starts with first word of translation, ends with <eos>
                    # [1, n_heads, src.shape[1], src.shape[1]]: attention for <bos>, words, <eos>
                    return output[:, :-1, :], attention # skip <eos> with :-1 in sequence dimension
                trg_indexes.append(pred_token)

        if inference:
            return None, None
        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]
        return output, attention


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

            # add dummy batch dimension
            outs, attn = self.model(t.unsqueeze(dim=0))
            if outs is not None:
                word_ids = outs.argmax(-1).squeeze(dim=0).tolist()
                translation = " ".join(TranslationDataSet.token_to_int['en'].lookup_tokens(word_ids))
                '''attention_map = get_attention_map(
                    TranslationDataSet.vocab_transform['de'].lookup_tokens(t.tolist()),
                    TranslationDataSet.vocab_transform['en'].lookup_tokens(word_ids), attn)'''
                attention_map = None
            else:
                translation = "No translation"
                attention_map = None
            results.append({
                "translation": translation,
                "attention_map": attention_map
            })
        return results

    def step(self, batch, batch_idx):
        src, trg, _ = batch # [batch_size, de_padded_seq],[batch_size,en_padded_seq]

        # with trg[:, :-1]) we trim eos in the target during forward operations,and want the decoder to produce it
        output, _ = self.model(src, trg[:, :-1]) # [batch size, trg len - 1, output dim]

        # we need to flatten first two dimension to be able to calculate cross entropy
        output = output.flatten(0,1) # [batch size * trg len - 1, output dim]
        # for loss, now we trim bos in the target
        trg = trg[:, 1:].reshape(-1) # [batch size * trg len - 1]
        loss = self.model.nn_loss(output, trg)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        # we need to specify batch size explicitly, since pl doesn't know that first rank is seqlen and it can
        # have different values for src and trg
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True, on_step=True)
        self.log("train_ppl", loss.exp().item(), prog_bar=True, on_epoch=True, on_step=False)
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


params = base.init_env("7/params.yml")
p = params['data']
data_module = TranslationDataModule(batch_size=p['batch_size'],
                                    src_lang=p['src_lang'],
                                    trg_lang=p['trg_lang'],
                                    max_tokens=p['max_tokens'],batch_first=True)
p = params['model']

enc = Encoder(input_dim=5000, **p)
dec = Decoder(output_dim=5000, **p)


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












