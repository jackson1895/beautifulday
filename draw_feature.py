import numpy as np
import os
from PIL import Image
def draw_feature(draw_dir,dir):
    if not os.path.exists(draw_dir):
        print(draw_dir+' do not exist')
        os.mkdir(draw_dir)
    for id in os.listdir(dir):
        vid_path = os.path.join(dir,id)
        vid = np.load(vid_path)
        frames,_=vid.shape
        out_put_v_path = os.path.join(draw_dir,id)
        if not os.path.exists(out_put_v_path):
            print(out_put_v_path + ' do not exist')
            os.mkdir(out_put_v_path)
        for index in range(frames):
            img_path = 'img-{05d}.jpg'%(index)
            img_path = os.path.join(out_put_v_path,img_path)
            img = Image.fromarray(vid[index,:,:,:],mode='RGB')
            img.save(img_path)

def cacul_max(dir):
    num=0
    for i in os.listdir(dir):
        numpy_path = os.path.join(dir,i)
        tmp_numpy = np.load(numpy_path)
        dim = tmp_numpy.shape[0]
        if dim>=num:
            num=dim

    return num


if __name__ == '__main__':
    dir = '/home/gaoliqing/lhb/video-classification-3d-cnn-pytorch-master/features/'
    draw_dir ='/home/gaoliqing/lhb/video-classification-3d-cnn-pytorch-master/draw_features/'
    # draw_feature(draw_dir,dir)
    max_num = cacul_max(dir)
    print(max_num)