import os
import torch
from PIL import Image
new_root_out = "/home/lhb/Disk2T/lihaibo/dataset/3D-Resnet/jpg_new"
for root_out, dirs_out, files_out in os.walk("/home/lhb/Disk2T/lihaibo/dataset/3D-Resnet/jpg", topdown=True):
    if root_out == "/home/lhb/Disk2T/lihaibo/dataset/3D-Resnet/jpg":
        for dir_out in dirs_out:
            dirs = os.path.join(root_out, dir_out)
            new_path1 = os.path.join(new_root_out, dir_out)
            for root2, dirs2, files2 in os.walk(dirs):
                if root2 == dirs:
                    for dir in dirs2:
                        new_root = os.path.join(root2, dir)
                        new_path2 = os.path.join(new_path1, dir)
                        if not os.path.exists(new_path2):
                            os.makedirs(new_path2)
                        for root1, dirs1, files1 in os.walk(new_root):
                            if root1 == new_root:
                                a = len(files1)
                                for filename in files1:
                                    pic_root = os.path.join(root1, filename)
                                    name, suf = os.path.splitext(filename)
                                    if suf == '.jpg':
                                        if int(name[-3:]) > 14 and (a - int(name[-3:])) > 10:
                                            n = int(name[-3:])
                                            if n < 100:
                                                name = name[:-2]
                                                m = n - 14
                                                if m < 10:
                                                    name = name + '0'
                                                    name = name + str(m)
                                                else:
                                                    name = name + str(m)
                                            else:
                                                name = name[:-3]
                                                m = n - 14
                                                if m > 99:
                                                    name = name + str(m)
                                                else:
                                                    name = name + '0'
                                                    name = name + str(m)
                                            name = name + suf
                                            new_path3 = os.path.join(new_path2, name)
                                            img = Image.open(pic_root)
                                            img.save(new_path3)
