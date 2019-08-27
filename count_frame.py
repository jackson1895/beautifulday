import os
from tqdm import tqdm
root_path = '/media/ext/gaoliqing/public_JPG'
frames = []
for v in tqdm(os.listdir(root_path)):
    for vid_path in os.listdir(os.path.join(root_path,v)):
        n_frames = int(open(os.path.join(root_path,v,vid_path,'n_frames')).read())
        frames.append(n_frames)

maxnum=max(frames)
print (maxnum)
