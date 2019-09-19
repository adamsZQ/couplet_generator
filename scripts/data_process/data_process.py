#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/9/19 9:14 PM
# @Author  : zchai
import csv
import json
import os

from couplet_generator.utils import conf


def process():
    config = conf['seq2seq_allen']
    origin_prefix = config['origin_data_prefix']
    target_prefix = config['small_data_prefix']

    train_source_file = config['origin_train_source_data']
    train_target_file = config['origin_train_target_data']

    test_source_file = config['origin_test_source_data']
    test_target_file = config['origin_test_target_data']

    for source_file, target_file in [(train_source_file, train_target_file), (test_source_file, test_target_file)]:
        if source_file.split('/')[0] == 'train':
            file = 'train_data.tsv'
        else:
            file = 'test_data.tsv'
        with open(os.path.join(origin_prefix, source_file), mode='r', encoding='utf-8') as f:
            source_data_list = f.readlines()
            source_data_list = source_data_list[:100]

        with open(os.path.join(origin_prefix, target_file), mode='r', encoding='utf-8') as f:
            target_data_list = f.readlines()
            target_data_list = target_data_list[:100]

        with open(os.path.join(target_prefix, file), mode='w', encoding='utf-8') as w:
            tsv_v = csv.writer(w, delimiter='\t')
            for source_data, target_data in zip(source_data_list, target_data_list):
                tsv_v.writerow([source_data.strip(), target_data.strip()])


if __name__ == '__main__':
    process()