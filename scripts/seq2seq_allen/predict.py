#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19/9/19 10:15 AM
# @Author  : zchai
from couplet_generator.my_logger import Logger
from couplet_generator.seq2seq_allen import Seq2SeqAllen

logger = Logger(__name__).get_logger()


def predict():
    seq2seq = Seq2SeqAllen(training=False)

    result = seq2seq.predict('乍 暖 还 寒')

    logger.info(result)


if __name__ == '__main__':
    predict()