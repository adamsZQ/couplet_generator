#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/9/19 7:49 PM
# @Author  : zchai
import itertools
import os

import torch
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor, Seq2SeqPredictor
from allennlp.training.trainer import Trainer
from torch import optim

from couplet_generator.my_logger import Logger
from couplet_generator.utils import conf

logger = Logger(__name__).get_logger()


class Seq2SeqAllen:

    def __init__(self, training=False):
        self.training = training
        config = conf['seq2seq_allen']
        self.model_path = config['model_path']
        self.vocab_path = config['vocab_path']
        prefix = config['processed_data_prefix']
        train_file = config['train_data']
        valid_file = config['test_data']
        src_embedding_dim = config['src_embedding_dim']
        trg_embedding_dim = config['trg_embedding_dim']
        hidden_dim = config['hidden_dim']
        epoch = config['epoch']
        patience = config['patience']

        if torch.cuda.is_available():
            self.cuda_device = 0
        else:
            self.cuda_device = -1

        self.reader = Seq2SeqDatasetReader(
                        source_tokenizer=WordTokenizer(),
                        target_tokenizer=WordTokenizer(),
                        source_token_indexers={'tokens': SingleIdTokenIndexer()},
                        target_token_indexers={'tokens': SingleIdTokenIndexer()})

        if self.training:
            self.train_dataset = self.reader.read(os.path.join(prefix, train_file))
            self.valid_dataset = self.reader.read(os.path.join(prefix, valid_file))

            self.vocab = Vocabulary.from_instances(self.train_dataset + self.valid_dataset,
                                                   min_count={'tokens': 3})
        else:
            self.vocab = Vocabulary.from_files(self.vocab_path)

        src_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size('tokens'),
                                  embedding_dim=src_embedding_dim)

        encoder = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(src_embedding_dim, hidden_dim, batch_first=True))

        source_embedder = BasicTextFieldEmbedder({"tokens": src_embedding})

        self.model = SimpleSeq2Seq(vocab=self.vocab, source_embedder=source_embedder, encoder=encoder,
                                   max_decoding_steps=20,
                                   target_embedding_dim=trg_embedding_dim,
                                   use_bleu=True)

        optimizer = optim.Adam(self.model.parameters())
        iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])
        # 迭代器需要接受vocab，在训练时可以用vocab来index数据
        iterator.index_with(self.vocab)

        self.model.cuda(self.cuda_device)

        if training:
            self.trainer = Trainer(model=self.model,
                                   optimizer=optimizer,
                                   iterator=iterator,
                                   patience=patience,
                                   train_dataset=self.train_dataset,
                                   validation_dataset=self.valid_dataset,
                                   serialization_dir=self.model_path,
                                   num_epochs=epoch,
                                   cuda_device=self.cuda_device)

        if not self.training:
            with open(os.path.join(self.model_path, 'best.th'), 'rb') as f:
                self.model.load_state_dict(torch.load(f))
            self.model.cuda(self.cuda_device)
            self.model.training = self.training
            self.predictor = Seq2SeqPredictor(self.model, dataset_reader=self.reader)

    def train(self):
        self.vocab.save_to_files(self.vocab_path)
        self.trainer.train()

    def predict(self, sentence):
        if not self.training:
            return self.predictor.predict(sentence)
        else:
            logger.warning('Mode is in training mode!')
