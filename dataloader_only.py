import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        self.captions = json.load(open(opt["caption_json"]))
        info = json.load(open(opt["info_json"]))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']

        self.splits = info['videos']
        self.feats_dir = opt["feats_dir"]
        self.c3d_feats_dir = opt['c3d_feats_dir']
        self.with_c3d = opt['with_c3d']
        self.max_len = opt["max_len"]
        self.max_video_len = opt['video_duration']

        if self.mode == 'train':
            print('vocab size is ', len(self.ix_to_word))
            print('number of train videos: ', len(self.splits['train']))
            #TODO 德国的没有val集
            print('number of val videos: ', len(self.splits['val']))
            print('number of test videos: ', len(self.splits['test']))
            print('load feats from %s' % (self.feats_dir))
            print('max sequence length in data is', self.max_len)



        # load in the sequence data



    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load
        if self.mode == 'val':
            ix = self.splits['val'][ix]
            # ix = ix + len(self.splits['train'])
        elif self.mode == 'test':
            ix = self.splits['test'][ix]
            # ix = ix + len(self.splits['train']) + len(self.splits['val'])
        elif self.mode == 'train':
            ix = self.splits['train'][ix]

        fc_feat = []

        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % (ix))))

        '''[array([[-1.09368485, 0.90057115, 1.07806664, 0.57392767],
                [-0.68921935, -1.12460755, 1.49306539, 0.73024426],
                [-1.04000375, -0.78627929, 1.54668868, -0.32196869]])]

            a=np.concatenate(a,axis=1)
            a
            array([[-1.09368485,  0.90057115,  1.07806664,  0.57392767],
                [-0.68921935, -1.12460755,  1.49306539,  0.73024426],
                [-1.04000375, -0.78627929,  1.54668868, -0.32196869]])   
                '''
        #去头截尾 前14 后10
        fc_feat = np.concatenate(fc_feat, axis=1)
        #TODO 德国手语数据不需要去头截尾
        length,_ = fc_feat.shape
        fc_feat = fc_feat[14:length-10,:]
        clip_num, _ = fc_feat.shape
        #TODO 中国是256 德国是128
        if clip_num < self.max_video_len:
            final_fc_feat = np.zeros([self.max_video_len, 2048])
            for i in range(0,clip_num):
                final_fc_feat[i]=fc_feat[i]
            # fill_0 = 80-clip_num
            # for i in range(0, clip_num):
            #     final_fc_feat[i+fill_0] = fc_feat[i]
            # for i in range(0, 30 - clip_num):
            #     final_fc_feat[i + clip_num] = fc_feat[clip_num - 1]
        elif clip_num > self.max_video_len:
            samples = np.round(np.linspace(
                        0, clip_num - 1, self.max_video_len))
            final_fc_feat = np.zeros([len(samples), 2048])
            index = 0
            for i in samples:
                final_fc_feat[index] = fc_feat[int(i)]
                index += 1
            # final_fc_feat = fc_feat
        elif clip_num == self.max_video_len:
            final_fc_feat=fc_feat

        # elif clip_num > 30:
        #     samples = np.round(np.linspace(
        #         0, clip_num - 1, 30))
        #     final_fc_feat = np.zeros([len(samples), 512])
        #     index = 0
        #     for i in samples:
        #         final_fc_feat[index] = fc_feat[int(i)]
        #         index += 1
            # fc_feat=fc_feat[samples,:,:]
        if self.with_c3d == 1:
            c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video%i.npy' % (ix)))
            c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
            # x轴复制一次，y轴复制fc.shape[0]次
            fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
        label = np.ones(self.max_len)*(-1)
        mask = np.zeros(self.max_len)
        captions = self.captions['video%i' % (ix)]['final_captions']
        gts = np.ones((len(captions), self.max_len))*(-1)
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                # cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]

        # random select a caption for this video
        cap_ix = random.randint(0, len(captions) - 1)
        label = gts[cap_ix]
        non_zero = (label == -1).nonzero()
        # mask[:int(non_zero[0][0]) ] = 1

        data = {}
        data['fc_feats'] = torch.from_numpy(final_fc_feat).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        # data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['gts'] = torch.from_numpy(gts).long()
        # data['fc_feats'] = final_fc_feat
        # data['labels'] = label
        # data['masks'] = mask
        # data['gts'] = gts
        data['video_ids'] = 'video%i' % (ix)
        data['clip_num']=clip_num
        return data

    def __len__(self):
        return len(self.splits[self.mode])