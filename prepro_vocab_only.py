import re
import json
import argparse
import numpy as np
import jieba


def build_vocab(vids, params):
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    for vid, caps in vids.items():
        for cap in caps['captions']:
            # import pdb
            # pdb.set_trace()
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            # ws = cap.split(' ')
            for w in ws:
                counts[w] = counts.get(w, 0) + 1
    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')
    for vid, caps in vids.items():
        caps = caps['captions']
        vids[vid]['final_captions'] = []
        for cap in caps:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            # caption = [
            #     '<sos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
            caption = [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws]
            vids[vid]['final_captions'].append(caption)
        # for cap in caps:
        #     ws = re.sub(r'[.!,;?]', ' ', cap).split()
        #     # caption = [
        #     #     '<sos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
        # caption =[w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws]
        # vids[vid]['final_captions']=caption
    return vocab


def main(params):
    videos = json.load(open(params['input_json'], 'r'))['sentences']
    video_caption = {}
    for i in videos:
        if i['video_id'] not in video_caption.keys():
            video_caption[i['video_id']] = {'captions': []}
        video_caption[i['video_id']]['captions'].append(i['caption'])
    vocab = build_vocab(video_caption, params)
    itow = {i + 1: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table
    # wtoi['<eos>'] = 1
    # itow[1] = '<eos>'
    # wtoi['<sos>'] = 2
    # itow[2] = '<sos>'
    itow[0]='-'
    wtoi['-']=0
    # itow = {i+1 : w for i, w in enumerate(vocab)}
    # wtoi = {w: i+1 for i, w in enumerate(vocab)}  # inverse table
    # wtoi['<eos>'] = 0
    # itow[0] = '<eos>'
    # wtoi['<sos>'] = 1
    # itow[1] = '<sos>'

    out = {}
    out['ix_to_word'] = itow
    out['word_to_ix'] = wtoi
    # out['videos'] = {'train': [], 'val': [], 'test': []}
    out['videos'] = {'train': [], 'val': []}

    videos = json.load(open(params['input_json'], 'r'))['videos']
    # part_videos=[]
    # for i in videos:
    #     v_i = int(i['video_id'][5:])
    #
    #     # if not ((v_i >= 0 and v_i <= 749) or (v_i >= 1250 and v_i <= 2249) or (v_i >= 6750 and v_i <= 8499) or (
    #     #         v_i >= 12000 and v_i <= 12749) or (v_i >= 10250 and v_i <= 11749) or (v_i >= 21500 and v_i <= 22749) \
    #     #         or (v_i >= 9750 and v_i <= 9999) or (v_i >= 9250 and v_i <= 9499)):
    #     #     continue
    #     # if  ((int(i['video_id'][5:])>=0) and (int(i['video_id'][5:])<=9999)):
    #     #     # part_videos[i]=videos[i]
    #     part_videos.append(i)
    for i in videos:
        out['videos'][i['split']].append(int(i['id']))
    json.dump(out, open(params['info_json'], 'w'))
    json.dump(video_caption, open(params['caption_json'], 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', type=str, default='/media/ext1/gaoliqing/cslt/json_file_split1/train-val.json',
                        help='msr_vtt videoinfo json')
    parser.add_argument('--info_json', default='/media/ext1/gaoliqing/cslt/json_file_split1/info.json',
                        help='info about iw2word and word2ix')
    parser.add_argument('--caption_json', default='/media/ext1/gaoliqing/cslt/json_file_split1/caption.json', help='caption json file')


    parser.add_argument('--word_count_threshold', default=1, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
