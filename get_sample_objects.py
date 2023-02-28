import numpy as np
import utils
import os


def get_point_cloud(pc_path, class_name, class_dict):
    pc = np.genfromtxt(pc_path)
    pc_class = np.full((pc.shape[0], 1), class_dict[class_name])
    pc = np.c_[pc, pc_class]

    return pc


def get_rail_road_extents(rail, road):
    left_rail = rail[:,0].min()
    right_rail = rail[:,0].max()
    min_road_x = road[:,0].min()
    max_road_x = road[:,0].max()
    left_road_edge = road[road[:,0] < min_road_x + 0.5]
    right_road_edge = road[road[:,0] > max_road_x - 0.5]
    road_attitude = (right_road_edge[:,1].mean() - left_road_edge[:,1].mean()) / (max_road_x - min_road_x)
    lower_road = lambda x: left_road_edge[:,1].min() + road_attitude * x
    upper_road = lambda x: left_road_edge[:,1].max() + road_attitude * x

    return left_rail, right_rail, lower_road, upper_road


def crossing(pool_dir, pool, class_dict):
    # Choose a single ground surface and find corresponding road and railroad
    all_samples = os.listdir(os.path.join(pool_dir, f'pool{pool}'))
    ground_samples = [s for s in all_samples if 'ground' in s]
    ground_sample = np.random.choice(ground_samples)
    sample_index = ground_sample.split('_')[1]

    # Read files
    ground = get_point_cloud(os.path.join(pool_dir, f'pool{pool}', f'ground_{sample_index}_0.txt'), 'ground', class_dict)
    road = get_point_cloud(os.path.join(pool_dir, f'pool{pool}', f'road_{sample_index}_0.txt'), 'ground', class_dict)
    rail = get_point_cloud(os.path.join(pool_dir, f'pool{pool}', f'rail_{sample_index}_0.txt'), 'rail', class_dict)

    # Get boundaries of road and railroad
    left_rail, right_rail, lower_road, upper_road = get_rail_road_extents(rail, road)

    crossing = np.r_[ground, road, rail]

    return crossing, left_rail, right_rail, lower_road, upper_road


def barrier(samples, pool, class_dict):
    barrier = np.random.choice(samples[pool]['barrier']).copy()
    barrier = utils.set_anchor_point(barrier)
    barrier = utils.rotate_pc(barrier, np.random.normal(0, 2), 0)
    barrier = utils.rotate_pc(barrier, np.random.normal(0, 5), 1)

    barrier_class = np.full((barrier.shape[0], 1), class_dict['barrier'])
    barrier = np.c_[barrier, barrier_class]

    return barrier


def cross_sign(theta, samples, pool, class_dict):
    cross = np.random.choice(samples[pool]['cross']).copy()
    cross = utils.set_anchor_point(cross)

    cross = utils.rotate_pc(cross, np.random.normal(0, 3), 0)
    cross = utils.rotate_pc(cross, np.random.normal(0, 3), 1)
    cross = utils.rotate_pc(cross, theta + np.random.uniform(-5, 5), 2)
    cross = cross[cross[:,2]>=np.random.uniform(0, 0.1)]

    cross_class = np.full((cross.shape[0], 1), class_dict['cross'])
    cross = np.c_[cross, cross_class]

    return cross


def portal(theta, samples, pool, class_dict):
    portal = np.random.choice(samples[pool]['portal']).copy()
    portal[:,:2] -= portal[:,:2].mean(axis=0)
    portal[:,2] -= portal[:,2].min()
    portal = utils.rotate_pc(portal, np.random.normal(0, 2), 0)
    portal = utils.rotate_pc(portal, np.random.normal(0, 2), 1)
    portal = utils.rotate_pc(portal, theta + np.random.uniform(-5, 5), 2)

    portal_class = np.full((portal.shape[0], 1), class_dict['portal'])
    portal = np.c_[portal, portal_class]

    return portal


def pole(samples, pool, class_dict):
    pole = np.random.choice(samples[pool]['pole']).copy()
    pole_class = np.full((pole.shape[0], 1), class_dict['pole'])
    pole = np.c_[pole, pole_class]
    pole[:,:2] -= pole[:,:2].mean(axis=0)
    pole[:,2] -= pole[:,2].min()

    return pole


def wire(samples, pool, class_dict):
    wire = np.random.choice(samples[pool]['wire']).copy()
    wire_class = np.full((wire.shape[0], 1), class_dict['wire'])
    wire = np.c_[wire, wire_class]
    wire[:,:2] -= wire[:,:2].mean(axis=0)
    wire[:,2] -= wire[:,2].min()

    return wire


def tree(samples, pool, class_dict):
    tree = np.random.choice(samples[pool]['tree']).copy()
    tree_class = np.full((tree.shape[0], 1), class_dict['tree'])
    tree = np.c_[tree, tree_class]
    tree[:,:2] -= tree[:,:2].mean(axis=0)
    tree[:,2] -= tree[:,2].min()

    return tree
