import numpy as np
import cv2 as cv
import os
import shutil
import glob
import json
from scipy.spatial.transform import Rotation as R

def prep_folders(base_dir):
    folders = ['train', 'test', 'val']
    for folder in folders:
        full_path = os.path.join(base_dir, folder)
        print(full_path)
        if(not os.path.exists(full_path)):
            os.makedirs(full_path)


base_input_dir = "./base_data"
base_output_dir = "./"
images_dir = os.path.join(base_input_dir, "image")
rel_data = json.load(open(os.path.join(base_input_dir, "camera_intrinsic_extrinsic.json"), 'r'))
prep_folders(base_output_dir)

train_portion = 0.6
test_portion = 0.3
val_portion = 0.1

#assert(train_portion + test_portion + val_portion == 1.0, "train/test/val split must == 1.0")

num_cameras = 20
num_frames = 30
W = rel_data['intrinsic']['resX']
H = rel_data['intrinsic']['resY']
fx = rel_data['intrinsic']['fx']

camera_angle_x = 2 * np.arctan2(W, 2 * fx)

entries = []

convertCV_GL = True

K_gl = np.array([[1,0,0,0],\
                [0.0, -1, 0, 0.0],\
                [0.0, 0.0, -1, 0],\
                [0.0, 0.0, 0.0, 1.0]])

for cam_idx in range(num_cameras):
    for frame_idx in range(num_frames):
        image_file_path = os.path.join(images_dir, "camera_{0:0>3d}".format(cam_idx), "frame_{0:0>2d}.png".format(frame_idx))
        camera_extrinsic = np.array(rel_data['extrinsic']["camera_{0:0>3d}".format(cam_idx)])
        if convertCV_GL:
            camera_extrinsic = np.matmul(camera_extrinsic, np.linalg.inv(K_gl))
        entry = {"image_fname":image_file_path, "extrinsic":camera_extrinsic, "time":float(frame_idx) / float(num_frames-1)}
        entries.append(entry)

np.random.shuffle(entries)

total_entries = num_cameras * num_frames
train_cutoff = int(train_portion * total_entries)
test_cutoff = int(test_portion * total_entries)
val_cutoff = int(val_portion * total_entries)

train_data = sorted(entries[0:train_cutoff], key=lambda entry: entry['time'])
test_data = sorted(entries[train_cutoff:train_cutoff+test_cutoff], key=lambda entry: entry['time'])
val_data = sorted(entries[train_cutoff+test_cutoff:len(entries)], key=lambda entry: entry['time'])

train_json = {}
train_json['camera_angle_x'] = camera_angle_x
train_json['frames'] = []
i=0
previous_time = None
for entry in train_data:
    if(previous_time is None or entry['time'] != previous_time):
        previous_time = entry['time']
        img_dest = os.path.join(base_output_dir, 'train','r_{0:0>3}'.format(i))
        img_src = entry['image_fname']
        shutil.copyfile(img_src, img_dest+".png")
        train_entry = {"file_path": img_dest, "time":entry["time"], "transform_matrix": entry['extrinsic'].tolist()}
        train_json['frames'].append(train_entry)
        i = i + 1
    else:
        continue
json.dump(train_json, open("transforms_train.json", 'w'),indent=1)

test_json = {}
test_json['camera_angle_x'] = camera_angle_x
test_json['frames'] = []
i=0
for entry in test_data:
    img_dest = os.path.join(base_output_dir, 'test','r_{0:0>3}'.format(i))
    img_src = entry['image_fname']
    shutil.copyfile(img_src, img_dest+".png")
    test_entry = {"file_path": img_dest, "time":entry["time"], "transform_matrix": entry['extrinsic'].tolist()}
    test_json['frames'].append(test_entry)
    i = i + 1
json.dump(test_json, open("transforms_test.json", 'w'),indent=1)

val_json = {}
val_json['camera_angle_x'] = camera_angle_x
val_json['frames'] = []
i=0
for entry in val_data:
    img_dest = os.path.join(base_output_dir, 'val','r_{0:0>3}'.format(i))
    img_src = entry['image_fname']
    shutil.copyfile(img_src, img_dest+".png")
    val_entry = {"file_path": img_dest, "time":entry["time"], "transform_matrix": entry['extrinsic'].tolist()}
    val_json['frames'].append(val_entry)
    i = i + 1
json.dump(val_json, open("transforms_val.json", 'w'),indent=1)
