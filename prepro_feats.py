import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from pretrainedmodels  import utils
from PIL import Image
C, H, W = 3, 224, 224

def crop_video(dir):
    for index in os.listdir(dir):
        if index.endswith('.jpg'):
            jpg_path = os.path.join(dir,index)
            img = Image.open(jpg_path)
            # if img.size[0] > 200:
            img1 = img.crop((123, 60, 303, 240))
            img1 = img1.resize((224,224),Image.ANTIALIAS)
            img1.save(jpg_path)
        else:
            raise ValueError


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   # '-y',
                                   '-i', video,  # input file
                                   # '-vf', "scale=400:300",  # input file
                                   '-s', "427x240",
                                   # '-qscale:v', "2",  # quality for JPEG
                                   '{}/image_%05d.jpg'.format(dst)]

        # comnd = ["ffmpeg -i ",video,'  -s 427x240 {}/image_%05d.jpg'.format(dst)]
        # subprocess.call('ffmpeg -i ',video,'  -s 427x240 {}/image_%05d.jpg'.format(dst),
        #                 shell=True)
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % (dir_fc))
    video_list = glob.glob(os.path.join(params['video_path'], '*.avi'))
    # video_list = os.listdir(params['video_path'])
    # path_video_list = []
    # for i in range(len(video_list)):
    #     path_video_list.append(os.path.join(params['video_path'],video_list[i]))
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split(".")[0]
        # video_id = video.split("/")[-1]
        # v_i = int(video_id[5:])
        # if v_i==50:
        #     dst = params['model'] + '_' + video_id
        #     extract_frames(video, dst)
        #     crop_video(dst)
        #     a=1
        # if not ((v_i >= 0 and v_i<=749) or (v_i >= 1250 and v_i<=2249)or (v_i >= 6750 and v_i<=8499) or (v_i >= 12000 and v_i<=12749) \
        #      or (v_i >= 10250 and v_i<=11749) or (v_i >= 21500 and v_i<=22749)\
        #         or (v_i >= 9750 and v_i<=9999)or (v_i >= 9250 and v_i<=9499)):
        #     continue
        if os.path.exists(os.path.join(dir_fc,video_id+'.npy')):
            continue
        dst = params['model'] + '_' + video_id
        # dst = os.path.join(params['video_path'],video_id,'1')
        extract_frames(video, dst)
        crop_video(dst)
        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        # samples = np.round(np.linspace(
        #     0, len(image_list) - 1, params['n_frame_steps']))
        # image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros((len(image_list), C, H, W))
        for iImg in range(len(image_list)):
            img = load_image_fn(image_list[iImg])
            images[iImg] = img
        with torch.no_grad():
            fc_feats = model(images.cuda()).squeeze()
        img_feats = fc_feats.cpu().numpy()
        # Save the inception features
        outfile = os.path.join(dir_fc, video_id + '.npy')
        np.save(outfile, img_feats)
        # cleanup
        shutil.rmtree(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='/media/ext1/gaoliqing/dataset/5000video_features/', help='directory to store features')
    # parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=40,
    #                     help='how many frames to sampler per video')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='/media/ext/gaoliqing/public_dataset/res1/val', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='resnet152',
                        help='the CNN model you want to use to extract_feats')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    if params['model'] == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    else:
        print("doesn't support %s" % (params['model']))

    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)
    
    model = model.cuda()
    extract_feats(params, model, load_image_fn)
