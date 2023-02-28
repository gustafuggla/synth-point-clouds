import os
import numpy as np
import utils
from multiprocessing import Pool


def read_pc(scene_path, file_name):
    """Reads a single point cloud from file"""
    pc = np.genfromtxt(os.path.join(scene_path, file_name), delimiter=' ')
    pc_class = np.full((pc.shape[0], 1), CLASS_DICT[file_name.split('.')[0]])
    pc = np.c_[pc, pc_class]


    return pc


def assemble_scene(scene_num):
    """Reads and assembles all files for a single scene"""
    scene_path = os.path.join(POOL_DIR, scene_num)
    file_names = os.listdir(scene_path)
    file_names = [f for f in file_names if f.split('.')[0] in classes]
    file_name = file_names[0]
    pc = read_pc(scene_path, file_name)

    for file_name in file_names[1:]:
        segment = read_pc(scene_path, file_name)
        pc = np.r_[pc, segment]

    pc[:,:3] -= pc[:,:3].mean(axis=0)

    scenes = [pc]

    if AUGMENT:
        pc2 = utils.rotate_pc(pc.copy(), 180, 2)
        pc2[:,:3] -= pc2[:,:3].mean(axis=0)

        pc3 = pc.copy()
        pc3[:,0] = -pc3[:,0]
        pc3[:,:3] -= pc3[:,:3].mean(axis=0)

        pc4 = pc.copy()
        pc4[:,1] = -pc4[:,1]
        pc4[:,:3] -= pc4[:,:3].mean(axis=0)

        scenes = scenes + [pc2, pc3, pc4]

    i = 0
    for scene in scenes:

        if SAVETXT:
            np.savetxt(os.path.join(TXT_DIR, f'scene_{POOL}_{scene_num}_{i}.txt'), scene)
        else:
            utils.create_blocks(scene, f'{POOL}_{scene_num}_{i}', H5_DIR, 1024)

        i += 1


DATA_DIR = 'data/scenes/segmented'
TXT_DIR = ''
H5_DIR = ''
SAVETXT = True
AUGMENT = False
n_cpu = 4


CLASS_DICT = {'ground': 0,
              'cross': 1,
              'portal': 2,
              'barrier': 3,
              'wire': 4,
              'pole': 5}

# Classes to be included in scene
classes = ['ground', 'cross', 'barrier', 'portal', 'pole', 'wire']

for POOL in ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']:
    global POOL_DIR
    POOL_DIR = os.path.join(DATA_DIR, POOL)
    scene_nums = os.listdir(POOL_DIR)
    p = Pool(n_cpu)
    p.map(assemble_scene, scene_nums)
p.close()
