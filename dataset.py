import codecs
import copy
import numpy as np
import os
import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset



class SignVideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length


    def encode(self, seq):
        encoded_seq = []
        for unit in seq:
            if unit in self.word_to_ix:
                encoded_seq.append(self.word_to_ix[unit])
            else:
                encoded_seq.append(self.word_to_ix['<unk>'])
        return encoded_seq



    def get_targets_dict(self,captions):

        targets_dict = {}
        for line in captions:
            contents = captions[line]['final_captions'][0]
            if len(contents) < 0 or len(contents) > self.max_target_len:
                continue
            if self.config.encoding:
                labels = self.encode(contents)
            else:
                raise NotImplementedError
            targets_dict[line]=labels
        return targets_dict


    def pad(self, inputs, max_length=None):
        dim = len(inputs.shape)
        if dim == 1:
            if max_length is None:
                max_length = self.max_target_len
            pad_zeros_mat = np.zeros([1, max_length - inputs.shape[0]], dtype=np.int32)
            padded_inputs = np.column_stack([inputs.reshape(1, -1), pad_zeros_mat])
        elif dim == 2:
            if max_length is None:
                max_length = self.max_video_len
            feature_dim = inputs.shape[1]
            pad_zeros_mat = np.zeros([max_length - inputs.shape[0], feature_dim])
            padded_inputs = np.row_stack([inputs, pad_zeros_mat])
        else:
            raise AssertionError(
                'Features in inputs list must be one vector or two dimension matrix! ')
        return padded_inputs




    def __init__(self, config,mode):
        super(SignVideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data
        self.config = config
        self.captions = json.load(open(config.caption_json))
        info = json.load(open(config.info_json))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        self.splits = info['videos']
        self.feats_dir = config.feats_dir
        self.max_target_len = config.max_target_length
        self.max_video_len = config.max_input_length
        self.targets_dict = self.get_targets_dict(self.captions)
        self.vid_duration = config.vid_duration
        self.duration = config.duration

        if self.mode == 'train':
            print('vocab size is ', len(self.ix_to_word))
            print('number of train videos: ', len(self.splits['train']))
            print('number of val videos: ', len(self.splits['val']))
            # print('number of test videos: ', len(self.splits['test']))
            print('load feats from %s' % (self.feats_dir))
            print('max sequence length in data is', self.max_target_len)




    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load
        if self.mode == 'val':
            ix = self.splits['val'][ix]
        elif self.mode == 'test':
            ix = self.splits['test'][ix]
        elif self.mode == 'train':
            ix = self.splits['train'][ix]
        fc_feat = []

        fc_feat.append(np.load(os.path.join(self.feats_dir, 'video%i.npy' % (ix))))
        fc_feat = np.concatenate(fc_feat, axis=1)
        target = self.targets_dict['video%i'% (ix)]
        target = np.array(target)
        inputs_length = np.array(fc_feat.shape[0]).astype(np.int64)
        target_length = np.array(target.shape[0]).astype(np.int64)
        features = self.pad(fc_feat).astype(np.float32)
        target = self.pad(target).astype(np.int64).reshape(-1)
        return features,inputs_length,target,target_length

    def __len__(self):
        return len(self.splits[self.mode])


