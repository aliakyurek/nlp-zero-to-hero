import random
import torch
from torch import nn
from torch.utils import data
import pytorch_lightning as pl
import torchtext as tt
from typing import Optional
from functools import partial
from datasets import load_dataset


class TranslationDataSet(data.Dataset):
    SPECIAL_TOKENS = ('<unk>', '<pad>', '<bos>', '<eos>')

    # following three will be filled with build method
    # will handle token to int conversion like token_to_int['en']['hello'] => 1273
    token_to_int = {}
    # will handle splitting sentences like token_splitter['en]('hello world') => ['hello','world']
    token_splitter = {}
    lang_pair = None

    def __init__(self, split, randomized_size=None):
        # class constructor must be called after build called once at the beginning.
        super().__init__()  # An exception can ve raised here for this purpose

        # the tokenized data will be kept here as array of pairs
        self.tokenized_data = []
        # if randomized_size is passed, Dataset will return random samples and will behave as it has
        # this randomized size. Good when big batchsizes can't be used and small batchsize take to long
        # to complete the epoch
        self.randomized_size = randomized_size

        ds = self.ds[split]['translation']  # ds: [{'de': 'Vielen Dank.','en':'Thank you.'},...]

        for raw in ds:
            pair = []
            for lang in self.lang_pair:
                # get sentence in the enumerated language as tokenized tensors and append it as one element of pair.
                tensor_ = self.tensorize(raw[lang], lang)
                pair.append(tensor_)
            # after two elements of pair filled, add pair to data list.
            self.tokenized_data.append(pair)
        # sort the list of tensor pairs so that batches will have source sentences with similar lengths
        # for more efficient learning, but notice that we lose this efficiency when randomized_size is used.
        self.tokenized_data.sort(key=lambda x: len(x[0]), reverse=True)

    def __getitem__(self, idx):
        # if we use random sampled dataset, return a random entry
        if self.randomized_size:
            idx = random.randint(0, len(self.tokenized_data) - 1)
        return self.tokenized_data[idx]

    def __len__(self):
        return self.randomized_size if self.randomized_size else len(self.tokenized_data)

    # tensorise a sentence
    @classmethod
    def tensorize(cls, sentence, lang):
        token_list = [cls.SPECIAL_TOKENS.index('<bos>')]

        # Before tokenizing, to have better modelling, I lower everything.
        for token in cls.token_splitter[lang](sentence.lower()):
            token_list.append(cls.token_to_int[lang][token])
            # tokens must be integer, therefore I specify dtype as long.

        token_list.append(cls.SPECIAL_TOKENS.index('<eos>'))
        return torch.tensor(token_list, dtype=torch.long)

    # builds the vocab for both languages
    @classmethod
    def build(cls, lang_pair, max_tokens):
        cls.lang_pair = lang_pair
        s, t = cls.lang_pair

        # be aware that vocab should be built using only training data
        cls.ds = load_dataset('iwslt2017', f'iwslt2017-{s}-{t}')

        # DatasetDict({
        #     train: Dataset({
        #         features: ['translation'],
        #         num_rows: 206112
        #     })

        unk_idx = cls.SPECIAL_TOKENS.index('<unk>')
        for lang in lang_pair:
            # Create torchtext's Vocab object. As tokenizer, we use spacy
            tokenizer = tt.data.utils.get_tokenizer('spacy', language=lang)
            # I limit the number of tokens to 10000. That means most frequent 10000 tokens will have a value.
            # Another rule is that a word must occur at least 2 times to have a value.
            # Specials tokens passed and if special_first True, they have the first integer values of mapping like
            # cls.token_to_int[lang]["<pad>"] => 1. token_to_int has no idea about their content and meaning.
            # What important is to have unique values of them and use them correctly.
            # map(lambda x:tokenizer(x[i].lower()), iter) This generator is used to traverse
            # all entry of pairs and tokenize the correct language of pair
            cls.token_to_int[lang] = tt.vocab.build_vocab_from_iterator(
                map(lambda x: tokenizer(x[lang].lower()), cls.ds['train']['translation']),
                min_freq=2,
                specials=cls.SPECIAL_TOKENS,
                special_first=True, max_tokens=max_tokens)
            cls.token_splitter[lang] = tokenizer
            # This is very important setting. If a passed token is not in vocabulary, a default value is returned.
            cls.token_to_int[lang].set_default_index(unk_idx)


class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, src_lang, trg_lang, batch_size, max_tokens, batch_first=False):
        super().__init__()
        self.collate_fn = None
        self.test_dataset = None
        self.valid_dataset = None
        self.train_dataset = None
        self.lang_pair = (src_lang, trg_lang)
        self.batch_size = batch_size
        self.batch_first = batch_first

        TranslationDataSet.build(self.lang_pair, max_tokens)
        # Once the vocabularies are built based on criteria (min_freq, max_tokens)
        # len returns the number of unique tokens the vocab. These dimensions will be used
        # in layer dimensions
        self.input_dim = len(TranslationDataSet.token_to_int[self.lang_pair[0]])
        self.output_dim = len(TranslationDataSet.token_to_int[self.lang_pair[1]])

        # Actually we never prepend/append these strings to the sentences, but having a list of them makes them unique,
        # and it's possible to use them as if they are enumerations.
        self.pad_idx = TranslationDataSet.SPECIAL_TOKENS.index('<pad>')
        self.bos_idx = TranslationDataSet.SPECIAL_TOKENS.index('<bos>')
        self.eos_idx = TranslationDataSet.SPECIAL_TOKENS.index('<eos>')

    # function is to be used as so called collate_fn for dataloader.
    # A collate_fn is called after getting batch sized data from dataset and applied to this data.
    # Actually we could append these bos and eos while creating the dataset, but it's not the best practice.
    @staticmethod
    def generate_batch(batch_first, pad_idx, data_batch):
        # data_batch is a list of pair of tensors, dispatch to specific language tensor lists.
        # data_batch list(batch_size)
        # data_batch[i] = list(2)
        # # data_batch[i][0] = tensor [src_len] of ith sample in batch
        # # data_batch[i][1] = tensor [trg_len] of ith sample in batch
        (de_batch, en_batch) = zip(*data_batch)
        lengths = [t.shape[0] for t in de_batch]

        # pad_sequence automatically creates batches from lists using padding, and by default,
        # it uses batch as second dim.
        de_batch = torch.nn.utils.rnn.pad_sequence(de_batch, padding_value=pad_idx, batch_first=batch_first)
        en_batch = torch.nn.utils.rnn.pad_sequence(en_batch, padding_value=pad_idx, batch_first=batch_first)
        return de_batch, en_batch, lengths

    # setup is called from every process across all the nodes. Setting state here is recommended.
    def setup(self, stage: Optional[str] = None):

        if stage == "fit":
            self.train_dataset = TranslationDataSet(split="train")
            self.valid_dataset = TranslationDataSet(split="validation")

        if stage == "test":
            self.test_dataset = TranslationDataSet(split="test")

        self.collate_fn = partial(self.generate_batch, self.batch_first, self.pad_idx)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                               collate_fn=self.collate_fn)

    def val_dataloader(self):
        return data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,

                               collate_fn=self.collate_fn)

class Seq2Seq(nn.Module):
    # specials are expected in bos, eos, pad order
    def __init__(self, encoder, decoder, specials):
        super().__init__()

        self.m_encoder = encoder
        self.m_decoder = decoder
        self.specials = specials
        self.nn_loss = nn.CrossEntropyLoss(ignore_index=specials['pad'])

        for m in self.modules():
            for name, param in m.named_parameters():
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
        trg_len = trg.shape[0] if not inference else 1000
        # list to store decoder predictions
        predictions = []

        # hidden state and cell state of the encoder are to be used as the initial states of the decoder
        dec_input = self.m_encoder(src)

        # get the first input to the decoder (<bos> tokens) for all batches, actually getting <bos> from trg
        # would make it more readable, but for inference trg will be None. So we use src.
        input = src[0, :]  # [batch size]

        # When training/testing our model, we always know how many words are in our target sentence, so we stop
        # generating words once we hit that many. During inference, it is common to keep generating words until the
        # model outputs an <eos> token or after a certain amount of words has been generated.
        # here we can't break the loop etc. based on eos, since we operate on batches.
        # (trg_len - 1) is to skip <bos> in trg, since we already provide it with input=src[0, :] above
        for t in range(trg_len - 1):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            prediction, dec_input = self.m_decoder(input, dec_input)
            # output [batch_size,output_dim], [num layers, batch size, hidden dim]

            # add predictions to the list
            predictions.append(prediction)

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input, if not, use predicted token
            if not inference and teacher_force:
                input = trg[t + 1]  # [batch_size]
            else:
                # get the highest predicted token from our predictions
                input = prediction.argmax(1)  # [batch_size]

            if inference:
                if input.item() == self.specials['eos']:
                    # torch.stack combines tensors in an iterable, in a new first dimension.
                    return torch.stack(predictions[:-1])

        # if we are in inferences mode and output not generated so far, return None
        if inference:
            return None
        return torch.stack(predictions)  # [trg_len-1, batch_size, output_dim]


class TranslationExperiment(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.m_seq2seq = model
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.Adam(self.m_seq2seq.parameters(), lr=self.lr)

    # This function is really intended to take sentences to translate
    def forward(self, X):
        results = []
        for s in X:
            # tensorize and append&prepend bos and eos
            t = TranslationDataSet.tensorize(s, "de").to(self.device)  #

            # add dummy batch dimension
            outs = self.m_seq2seq(t.unsqueeze(dim=1))  # [src_len, 1, output_dim]
            if outs is not None:
                word_ids = outs.argmax(-1).squeeze(dim=1).tolist()
                translation = " ".join(TranslationDataSet.token_to_int['en'].lookup_tokens(word_ids))
            else:
                translation = "No translation"
            results.append(translation)
        return results

    def step(self, batch, batch_idx):
        src, trg, _ = batch  # src=[de_padded_seq, batch_size], trg=[en_padded_seq,batch_size]
        output = self.m_seq2seq(src, trg)  # [trg_len-1, batch_size, output_dim]

        # The loss function only knows two dimensions (batch and classes), so it has no idea of sequences.
        # Therefore, we need to flatten sequence and batch dimension.
        # Also remember that, loss target (or ground truth in other words) could be one hot encoding or class index
        # In our setup, it's class index, therefore it has one less dimension than OHEd input.
        # ignore the bos in target using trg[1:]
        output = output.flatten(0, 1)  # [trg_len-1*batch_size, output_dim]
        trg = trg[1:].flatten()  # [trg_len-1*batch_size]
        loss = self.m_seq2seq.nn_loss(output, trg)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        # we need to specify batch size explicitly, since pl doesn't know that first rank is seqlen, and it can
        # have different values for src and trg
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True, on_step=True, batch_size=batch[0].shape[1])
        self.log("train_ppl", loss.exp().item(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=batch[0].shape[1])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False, batch_size=batch[0].shape[1])
        return loss

    def on_validation_epoch_end(self):
        sentence = "Ich liebe dich."
        # pl automatically sets model to eval mode and disables grad
        translation = self([sentence])[0]
        text = f"{sentence}->{translation} | Loss:{self.trainer.logged_metrics['val_loss'].item():.3f}"
        self.logger.experiment.add_text("Translation", text, global_step=self.current_epoch)

