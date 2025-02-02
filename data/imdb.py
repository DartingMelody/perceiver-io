import argparse
import glob
import os
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import IMDB
from tokenizers import Tokenizer
from tokenizers.normalizers import Replace

from perceiver.tokenizer import (
    create_tokenizer,
    train_tokenizer,
    save_tokenizer,
    load_tokenizer,
    PAD_TOKEN
)


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_split(root, split):
    if split not in ['train', 'test']:
        raise ValueError(f'invalid split: {split}')

    raw_x = []
    raw_y = []

    for i, label in enumerate(['neg', 'pos']):
        path_pattern = os.path.join(root, f'IMDB/aclImdb/{split}/{label}', '*.txt')
        for name in glob.glob(path_pattern):
            with open(name, encoding='utf-8') as f:
                raw_x.append(f.read())
                raw_y.append(i)

    return raw_x, raw_y


class IMDBDataset(Dataset):
    def __init__(self, root, split):
        self.raw_x, self.raw_y = load_split(root, split)

    def __len__(self):
        return len(self.raw_x)

    def __getitem__(self, index):
        return self.raw_y[index], self.raw_x[index]


class Collator:
    def __init__(self, tokenizer: Tokenizer, max_seq_len: int):
        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding(pad_id=self.pad_id, pad_token=PAD_TOKEN)
        self.tokenizer.enable_truncation(max_length=max_seq_len)

    def collate(self, batch):
        ys, xs = zip(*batch)
        xs_ids = [x.ids for x in self.tokenizer.encode_batch(xs)]
        xs_ids = torch.tensor(xs_ids)
        pad_mask = xs_ids == self.pad_id
        return torch.tensor(ys), xs_ids, pad_mask

    def encode(self, samples):
        batch = [(0, sample) for sample in samples]
        return self.collate(batch)[1:]


class IMDBDataModule(pl.LightningDataModule):
    def __init__(self,
                 root='.cache',
                 max_seq_len=512,
                 vocab_size=10003,
                 batch_size=64,
                 num_workers=3,
                 pin_memory=False):
        super().__init__()
        self.root = root
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.ds_train = None
        self.ds_valid = None

        self.tokenizer_path = os.path.join(self.root, f'imdb-tokenizer-{vocab_size}.json')
        self.tokenizer = None
        self.collator = None

    @classmethod
    def create(cls, args: argparse.Namespace):
        return cls(root=args.root,
                   max_seq_len=args.max_seq_len,
                   vocab_size=args.vocab_size,
                   batch_size=args.batch_size,
                   num_workers=args.num_workers,
                   pin_memory=args.pin_memory)

    @classmethod
    def setup_parser(cls, parser):
        group = parser.add_argument_group('data')
        group.add_argument('--root', default='.cache', help=' ')
        group.add_argument('--max_seq_len', default=512, type=int, help=' ')
        group.add_argument('--vocab_size', default=10003, type=int, help=' ')
        group.add_argument('--batch_size', default=64, type=int, help=' ')
        group.add_argument('--num_workers', default=2, type=int, help=' ')
        group.add_argument('--pin_memory', default=False, action='store_true', help=' ')
        return parser

    def prepare_data(self, *args, **kwargs):
        if not os.path.exists(os.path.join(self.root, 'IMDB')):
            # download and extract IMDB data
            IMDB(root=self.root)

        if not os.path.exists(self.tokenizer_path):
            # load raw IMDB train data
            raw_x, _ = load_split(root=self.root, split='train')

            # train and save tokenizer
            tokenizer = create_tokenizer(Replace('<br />', ' '))
            train_tokenizer(tokenizer, data=raw_x, vocab_size=self.vocab_size)
            save_tokenizer(tokenizer, self.tokenizer_path)

    def setup(self, stage=None):
        self.tokenizer = load_tokenizer(self.tokenizer_path)
        self.collator = Collator(self.tokenizer, self.max_seq_len)

        self.ds_train = IMDBDataset(root=self.root, split='train')
        self.ds_valid = IMDBDataset(root=self.root, split='test')

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          shuffle=True,
                          collate_fn=self.collator.collate,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.ds_valid,
                          shuffle=False,
                          collate_fn=self.collator.collate,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
