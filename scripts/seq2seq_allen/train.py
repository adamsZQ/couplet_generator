#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/9/19 8:54 PM
# @Author  : zchai
import sys
import os


CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../../')))

from couplet_generator.seq2seq_allen import Seq2SeqAllen


def train():
    model = Seq2SeqAllen()
    model.train()


if __name__ == '__main__':
    train()