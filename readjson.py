import json
import numpy as np
data =json.load(open('/media/ext1/gaoliqing/cslt/json_file_split1/train-val.json','r'))
min1 = 0
num = []
count = 0
for i ,v in enumerate(data):
    tmp = len(data[v]['final_captions'][0])
    num.append(tmp)
    if tmp<=20:
        count+=1
    if tmp>=min1:
        min1 = tmp

mean_num=np.mean(num)
media  = np.median(num)

cp_data = {}
cp_data['videos']=data['videos']

sents = data['sentences']
cp_data['sentences']=[]
max = 0
for i in sents:
    tmp={}
    sen="".join(i['caption'].split(' '))
    tmp_str=''
    for j in range(len(sen)):
        if j == len(sen)-1:
            if len(sen) >= max :
                max=len(sen)
            tmp_str+=sen[j]
        else:
            tmp_str+=sen[j]+'.'
    vid = i['video_id']
    tmp['caption']=tmp_str
    tmp['video_id']=vid
    cp_data['sentences'].append(tmp)
print (max)
# json.dump(cp_data, open('./json_files/corpus_sign_no_split_word.json', 'w'))
a=1