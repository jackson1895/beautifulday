import os
import numpy as np
import cv2
from glob import glob
import timeit
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
_IMAGE_SIZE = 224


def cal_for_frames(video_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path.format('u'), "{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path.format('v'), "{:06d}.jpg".format(i)),
                    flow[:, :, 1])


def extract_flow(video_path, flow_path):
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)
    print('complete:' + flow_path)
    return
def is_color_image(url):
    im=Image.open(url)
    pix=im.convert('RGB')
    width=im.size[0]
    height=im.size[1]
    oimage_color_type="Grey Image"
    is_color=[]
    for x in range(width):
        for y in range(height):
            r,g,b=pix.getpixel((x,y))
            r=int(r)
            g=int(g)
            b=int(b)
            if (r==g) and (g==b):
                pass
            else:
                oimage_color_type='Color Image'
    return oimage_color_type

if __name__ == '__main__':
    video_paths = "./resnet152_video50"
    flow_paths = "./resnet152_video50_flow"
    if not os.path.exists(flow_paths):
        os.mkdir(flow_paths)
    # video_lengths = 109
    begin=timeit.default_timer()
    extract_flow(video_paths, flow_paths)
    end=timeit.default_timer()
    image_list = sorted(glob(os.path.join(flow_paths, '*.jpg')))
    mean_video=[]
    for i in tqdm(range(len(image_list))):
        i_p = image_list[i]
        img = cv2.imread(i_p)
        # is_dack = is_color_image(i_p)
        min_pixel = img[:,:,0].min()
        # img = Image.open(i_p)
        # transform = transforms.ToTensor()
        # tensor = transform(img)
        # m_i = torch.mean(tensor)
        mean_video.append(min_pixel)

    print(mean_video)
    # print(end-begin)
