import numpy as np
import os
import utils
from scipy.spatial import KDTree
import place_objects as place
import get_sample_objects as get_sample
import get_mesh_objects as get_mesh
from multiprocessing import Pool
import time


def get_samples():
    """Creates disctionary of pools and object types"""

    object_samples = {}

    if USE_MESH_OBJECTS:
        for i in range(1,6):
            # Add samples from pools
            pool_i = os.path.join(POOL_DIR, f'pool{i}')
            samples = os.listdir(pool_i)
            pool_samples = {}
            for object_type in ['tree']:
                pool_samples[object_type] = []
                for sample_name in [s for s in samples if object_type in s]:
                    pool_samples[object_type].append(np.genfromtxt(os.path.join(pool_i, sample_name)))
            object_samples[i] = pool_samples

            # Add mesh samples
            for object_type in ['barrier', 'cross', 'portal', 'pole', 'polearm', 'wire', 'geom']:
                object_dir = os.path.join(MESH_DIR, object_type)
                sample_list = []
                for file_name in os.listdir(object_dir):
                    sample_list.append(np.genfromtxt(os.path.join(object_dir, file_name)))
                object_samples[object_type] = sample_list

    else:
        for i in range(1,6):
            pool_i = os.path.join(POOL_DIR, f'pool{i}')
            samples = os.listdir(pool_i)
            pool_samples = {}
            for object_type in ['barrier', 'cross', 'portal', 'pole', 'tree', 'wire']:
                pool_samples[object_type] = []
                for sample_name in [s for s in samples if object_type in s]:
                    pool_samples[object_type].append(np.genfromtxt(os.path.join(pool_i, sample_name)))
            object_samples[i] = pool_samples

    return object_samples


def add_objects(scene, left_rail, right_rail, lower_road, upper_road):
    """Add objects to scene"""

    # Store coordinates in KDTree to find the ground height at specific xy locations
    height_map = KDTree(scene[:,:3])

    # Rotation angle theta perpendicular to road - all samples are initially oriented along y-axis
    road_vector = np.array([1, lower_road(1) - lower_road(0)])
    theta = np.arctan2(road_vector[1], road_vector[0])
    theta = np.degrees(theta)

    # Place objects
    scene = add_barriers(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta)
    scene = add_cross_signs(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta)
    scene = add_portals(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta)
    scene = add_trees(scene, left_rail, right_rail, lower_road, upper_road, height_map)
    scene = add_poles(scene, left_rail, right_rail, height_map)
    scene = add_wires(scene, height_map)

    return scene


def add_barriers(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta):
    if USE_MESH_OBJECTS:
        left_barrier = get_mesh.barrier(SAMPLES, CLASS_DICT)
        right_barrier = get_mesh.barrier(SAMPLES, CLASS_DICT)

    else:
        left_barrier = get_sample.barrier(SAMPLES, POOL, CLASS_DICT)
        right_barrier = get_sample.barrier(SAMPLES, POOL, CLASS_DICT)
    scene = place.barriers(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta, left_barrier, right_barrier)

    return scene


def add_cross_signs(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta):
    if USE_MESH_OBJECTS:
        cross_signs = (get_mesh.cross_sign(theta, SAMPLES, CLASS_DICT) for _ in range(4))

    else:
        cross_signs = (get_sample.cross_sign(theta, SAMPLES, POOL, CLASS_DICT) for _ in range(4))
    scene = place.cross_signs(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta, cross_signs)

    return scene


def add_portals(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta):
    if USE_MESH_OBJECTS:
        portals = (get_mesh.portal(theta, SAMPLES, CLASS_DICT) for _ in range(2))

    else:
        portals = (get_sample.portal(theta, SAMPLES, POOL, CLASS_DICT) for _ in range(2))
    scene = place.portals(scene, left_rail, right_rail, lower_road, upper_road, height_map, portals)

    return scene


def add_poles(scene, left_rail, right_rail, height_map):
    if USE_MESH_OBJECTS:
        pole_1 = get_mesh.pole(SAMPLES, CLASS_DICT)
        pole_2 = get_mesh.pole(SAMPLES, CLASS_DICT)

    else:
        pole_1 = get_sample.pole(SAMPLES, POOL, CLASS_DICT)
        pole_2 = get_sample.pole(SAMPLES, POOL, CLASS_DICT)
    scene = place.poles(scene, left_rail, right_rail, height_map, pole_1, pole_2)

    return scene


def add_wires(scene, height_map):
    if USE_MESH_OBJECTS:
        wire = get_mesh.wire(SAMPLES, SCENE_Y_MIN, SCENE_Y_MAX, CLASS_DICT)

    else:
        wire = get_sample.wire(SAMPLES, POOL, CLASS_DICT)

    scene = place.wires(scene, height_map, wire)
    scene = crop_scene(scene)

    return scene


def add_trees(scene, left_rail, right_rail, lower_road, upper_road, height_map):
    trees = (get_sample.tree(SAMPLES, POOL, CLASS_DICT) for _ in range(9))
    scene = place.trees(scene, left_rail, right_rail, lower_road, upper_road, height_map, trees)
    scene = crop_scene(scene)

    return scene


def crop_scene(scene):
    scene = scene[scene[:,0] >= SCENE_X_MIN]
    scene = scene[scene[:,1] >= SCENE_Y_MIN]
    scene = scene[scene[:,0] <= SCENE_X_MAX]
    scene = scene[scene[:,1] <= SCENE_Y_MAX]

    return scene


def generate_scene(i):
    # Set random seed - necessary for pool processing and repeatability
    seed = int(''.join([str(POOL), str(i)]))
    np.random.seed(seed)

    scene, left_rail, right_rail, lower_road, upper_road = get_sample.crossing(POOL_DIR, POOL, CLASS_DICT)
    scene[:,2] -= scene[:,2].min()

    global SCENE_X_MIN, SCENE_Y_MIN, SCENE_X_MAX, SCENE_Y_MAX
    SCENE_X_MIN, SCENE_Y_MIN = scene[:,:2].min(axis=0)
    SCENE_X_MAX, SCENE_Y_MAX = scene[:,:2].max(axis=0)

    scene = add_objects(scene, left_rail, right_rail, lower_road, upper_road)

    # Mirror and rotate
    if np.random.choice([True, False]):
        scene[:,0] = -scene[:,0]
    if np.random.choice([True, False]):
        scene = utils.rotate_pc(scene, 180, 2)

    # Center
    scene[:,:3] -= scene[:,:3].mean(axis=0)

    if SAVETXT:
        np.savetxt(os.path.join(TXT_DIR, f'scene_pool{POOL}_{i}.txt'), scene)
    else:
        utils.create_blocks(scene, f'pool{POOL}_{i}', H5_DIR, 1024)

if __name__ == '__main__':
    # Object classes
    CLASS_DICT = {'ground': 0,
                  'tree': 0,
                  'rail': 0,
                  'cross': 1,
                  'portal': 2,
                  'barrier': 3,
                  'wire': 4,
                  'pole': 5}

    # Data directories
    POOL_DIR = 'data/scenes/pools'
    MESH_DIR = 'data/mesh_geometries'

    # Parameters
    SAVETXT = True
    USE_MESH_OBJECTS = False
    TXT_DIR = ''
    H5_DIR = ''
    n_cpu = 4
    n_scenes = 36

    # Samples
    SAMPLES = get_samples()

    # Create scenes
    start_time = time.time()
    for POOL in [1,2,3,4,5]:
        p = Pool(n_cpu)
        p.map(generate_scene, range(n_scenes))
    p.close()
    print(f'Created {5*n_scenes} scenes in {time.time() - start_time} seconds')
