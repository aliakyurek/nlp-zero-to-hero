import random
import torch
from torch.utils import data
import pytorch_lightning as pl
import torchtext as tt
from typing import Optional
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class TranslationDataSet(data.Dataset):
    #### class members ####b
    SPECIAL_TOKENS=('<unk>', '<pad>', '<bos>', '<eos>')

    # following three will be filled with build method
    # will handle token to int conversion like token_to_int['en']['hello'] => 1273
    token_to_int = {}
    # will handle splitting sentences like token_splitter['en]('hello world') => ['hello','world']
    token_splitter = {}
    lang_pair = None
    #### class members ####

    def __init__(self, split, randomized_size=None):
        # class constructor must be called after build called once at the beginning.
        super().__init__() # An exception can ve raised here for this purpose

        # the tokenized data will be kept here as array of pairs
        self.tokenized_data = []
        # if randomized_size is passed, Dataset will return random samples and will behave as it has
        # this randomized size. Good when big batchsizes can't be used and small batchsize take to long
        # to complete the epoch
        self.randomized_size = randomized_size

        ds = tt.datasets.IWSLT2017(root='data', split=split, language_pair=self.lang_pair)
        # next(iter(ds)) => ('Vielen Dank, Chris.\n', 'Thank you so much, Chris.\n')
        for raw in ds:
            pair = []
            for i, lang in enumerate(self.lang_pair):
                # get sentence in the enumerated language as tokenized tensors and append it as one element of pair.
                # and also skip the "\n" at the end of sentences in the dataset
                tensor_ = self.tensorize(raw[i][:-1], lang)
                pair.append(tensor_)
            # after two elements of pair filled, add pair to data list.
            self.tokenized_data.append(pair)
        # sort the list of tensor pairs so that batches will have source sentences with similar lengths
        # for more efficient learning, but notice that we lose this efficiency when randomized_size is used.
        self.tokenized_data.sort(key=lambda x:len(x[0]),reverse=True)

    def __getitem__(self, idx):
        # if we use random sampled dataset, return a random entry
        if self.randomized_size:
            idx = random.randint(0,len(self.tokenized_data)-1)
        return self.tokenized_data[idx]

    def __len__(self):
        return self.randomized_size if self.randomized_size else len(self.tokenized_data)

    # tensorise a sentence
    @classmethod
    def tensorize(cls, sentence, lang):
        token_list = []
        token_list.append(cls.SPECIAL_TOKENS.index('<bos>'))

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

        # be aware that vocab should be built using only training data
        ds = tt.datasets.IWSLT2017(root='data', split="train", language_pair=lang_pair)
        # next(iter(ds)) => ('Vielen Dank, Chris.\n', 'Thank you so much, Chris.\n')
        unk_idx = cls.SPECIAL_TOKENS.index('<unk>')
        for i,lang in enumerate(lang_pair):
            # Create torchtext's Vocab object. As tokenizer we use spacy
            tokenizer = tt.data.utils.get_tokenizer('spacy', language=lang)
            # I limit the number of tokens to 10000. That means most frequent 10000 tokens will have a value.
            # Another rule is that a word must occur at least 2 times to have a value.
            # Specials tokens passed and if special_first True, they have the first integer values of mapping like
            # cls.token_to_int[lang]["<pad>"] => 1. token_to_int has no idea about their content and meaning.
            # What important is to have unique values of them and use them correctly.
            # map(lambda x:tokenizer(x[i].lower()), iter) This generator is used to traverse
            # all entry of pairs and tokenize the correct language of pair
            cls.token_to_int[lang] = tt.vocab.build_vocab_from_iterator(map(lambda x:tokenizer(x[i].lower()), ds),
                                                                           min_freq=2,
                                                                           specials=cls.SPECIAL_TOKENS,
                                                                           special_first=True,max_tokens=max_tokens)
            cls.token_splitter[lang] = tokenizer
            # This is very important setting. If a passed token is not in vocabulary, a default value is returned.
            cls.token_to_int[lang].set_default_index(unk_idx)


class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, src_lang, trg_lang, batch_size, max_tokens, batch_first=False):
        super().__init__()
        self.lang_pair = (src_lang,trg_lang)
        self.batch_size = batch_size
        self.batch_first = batch_first

        TranslationDataSet.build(self.lang_pair, max_tokens)
        # Once the vocabularies are built based on createria (min_freq, max_tokens)
        # len returns the number of unique tokens the vocab. These dimensions will be used
        # in layer dimensions
        self.input_dim = len(TranslationDataSet.token_to_int[self.lang_pair[0]])
        self.output_dim = len(TranslationDataSet.token_to_int[self.lang_pair[1]])

        # Actually we never prepend/append these strings to the sentences, but having a list of them makes them unique
        # and it's possible to use them as if they are enumerations.
        self.pad_idx = TranslationDataSet.SPECIAL_TOKENS.index('<pad>')
        self.bos_idx = TranslationDataSet.SPECIAL_TOKENS.index('<bos>')
        self.eos_idx = TranslationDataSet.SPECIAL_TOKENS.index('<eos>')

    # function is to be used as so called collate_fn for dataloader.
    # A collate_fn is called after getting batch sized data from dataset and applied to this data.
    # Actually we could append these bos and eos while creating the dataset but it's not the best practice.
    @staticmethod
    def generate_batch(batch_first, pad_idx, data_batch):
        # data_batch is a list of pair of tensors, dispatch to specific language tensor lists.
        # data_batch list(batch_size)
        # data_batch[i] = list(2)
        # # data_batch[i][0] = tensor [src_len] of ith sample in batch
        # # data_batch[i][1] = tensor [trg_len]
        lengths = []
        (de_batch, en_batch) = zip(*data_batch)
        lengths = [t.shape[0] for t in de_batch]

        # pad_sequence automatically creates batches from lists using padding, and by default, it uses batch as second dim.
        de_batch = torch.nn.utils.rnn.pad_sequence(de_batch, padding_value=pad_idx, batch_first=batch_first)
        en_batch = torch.nn.utils.rnn.pad_sequence(en_batch, padding_value=pad_idx, batch_first=batch_first)
        return de_batch, en_batch, lengths

    # prepare_data is called from the main process. It is not recommended to assign state here. Just downloading files
    # so that the further calls won't download again.
    def prepare_data(self):
        tt.datasets.IWSLT2017(root='data', split=('train', 'valid', 'test'), language_pair=self.lang_pair)

    # setup is called from every process across all the nodes. Setting state here is recommended.
    def setup(self, stage: Optional[str] = None):

        if stage == "fit":
            self.train_dataset = TranslationDataSet(split="train")
            self.valid_dataset = TranslationDataSet(split="valid")

        if stage == "test":
            self.test_dataset = TranslationDataSet(split="test")

        self.collate_fn = partial(self.generate_batch, self.batch_first, self.pad_idx)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)


def get_attention_map(sentence, translation, attention):
    fig,ax = plt.subplots(figsize=(10, 10))

    attention = attention.cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=10)

    x_ticks = [''] + sentence
    y_ticks = [''] + translation

    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set_yticklabels(y_ticks)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    return plt.gcf()