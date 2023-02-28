import numpy as np
from fbm import FBM
import h5py
import os


def split_barrier(barrier):
    """Splits a barrier into base and arms"""
    y_min = barrier[:,1].min()
    arms, base = barrier[barrier[:,1] >= y_min + 1.3], barrier[barrier[:,1] < y_min + 1.3]

    return arms, base


def get_intensity(pc, mean, stdev):
    """Creates an intensity profile for a single object based
    on a mean and standard deviation"""
    i = np.random.normal(mean, stdev, pc.shape[0])
    i[i>255] = 255
    i[i<0] = 0

    return i


def pairwise_shuffle(a, b):
    """Shuffles two lists in sync"""
    c = list(zip(a, b))
    np.random.shuffle(c)
    a, b = zip(*c)

    return a, b


def create_blocks(pc, i, h5_dir, num_points):
    """Divides a scene into blocks and saves it in h5 format"""
    # Parameters
    block_size = 1
    stride = block_size

    x_min, y_min, z_min = pc[:,:3].min(axis=0)
    pc[:,:3] -= np.array([x_min, y_min, z_min])
    x_max, y_max, z_max = pc[:,:3].max(axis=0)

    data_list, label_list = [], []
    for x in np.arange(0, x_max + stride, stride):
        for y in np.arange(0, y_max + stride, stride):
            pc_tmp = np.copy(pc)
            block = pc_tmp[pc_tmp[:,0] >= x]
            block = block[block[:,0] < x + block_size]
            block = block[block[:,1] >= y]
            block = block[block[:,1] < y + block_size]

            points_in_block = block.shape[0]

            if points_in_block == 0:
                continue

            if points_in_block > num_points:
                block = block[np.random.choice(points_in_block, num_points), :]
            elif points_in_block < num_points:
                dup_points = block[np.random.choice(points_in_block, num_points - points_in_block), :]
                block = np.concatenate((block, dup_points))

            assert block.shape[0] == num_points

            xyz = block[:,:3] - block[:,:3].min(axis=0)
            rgb = block[:,3:6] / 255
            norm_loc = block[:,:3] / x_max

            data = np.c_[xyz, rgb, norm_loc]
            labels = block[:,6]
            data_list.append(data)
            label_list.append(labels)

    data_list, label_list = pairwise_shuffle(data_list, label_list)

    f = h5py.File(os.path.join(h5_dir, f'scene_{i}.h5'), 'w')

    f['data'] = np.array(data_list)
    f['label'] = np.array(label_list).astype(int)


def rotate_point(p, R):
    """Rotates a single point"""
    return np.matmul(R, p)


def rotate_pc(pc, angle, axis, degrees=True):
    """Rotates a point cloud"""
    if degrees:
        angle = angle * np.pi / 180

    if axis == 0:
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle), np.cos(angle)]])

    elif axis == 1:
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]])

    elif axis == 2:
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])

    xyz = pc[:, :3]
    new_xyz = np.apply_along_axis(rotate_point, 1, xyz, R)
    pc = np.c_[new_xyz, pc[:, 3:]]

    return pc


def set_anchor_point(pc):
    """Sets the origin of a point cloud to its lowest point, e.g. the base
    of a pole"""
    z_min = pc[:,2].min()
    anchor_region = pc[pc[:,2] < z_min + 0.1]
    x_mean, y_mean = anchor_region[:,:2].mean(axis=0)
    pc[:,0] -= x_mean
    pc[:,1] -= y_mean
    pc[:,2] -= pc[:,2].min()

    return pc


def scale_pc(pc, size, axis):
    """Rescales a point cloud to a target size"""
    scale_factor = size / abs(pc[:,axis].max() - pc[:,axis].min())
    pc[:,:3] *= scale_factor

    return pc


def downsample(pc, num_points):
    """Reduces the number of points by random downsampling"""
    if num_points < pc.shape[0]:
        pc = pc[np.random.choice(pc.shape[0], num_points)]

    return pc


def fbm_point_removal(pc, axis):
    """Removes points along an axis according to a generated fBm sample"""
    pc = pc[pc[:,axis].argsort()]
    f = FBM((pc.shape[0]), np.random.uniform(0.1, 0.3))
    fbm_sample = f.fbm()[1:]
    fbm_sample -= np.median(fbm_sample)
    p = np.percentile(fbm_sample, np.random.uniform(70, 90))
    pc = pc[fbm_sample<p]

    return pc


def alter_shape_fbm(pc, axis_sort, axis_alter, scale_factor):
    """Shifts points across an axis according to a generated fBm sample"""
    pc = pc[pc[:,axis_sort].argsort()]
    f = FBM((pc.shape[0]), np.random.uniform(0.1, 0.8))
    fbm_sample = f.fbm()[1:]
    fbm_sample -= np.median(fbm_sample)
    fbm_sample /= max(abs(fbm_sample.min()), fbm_sample.max())

    filter_size = np.random.randint(10, 50)
    smoothed_sample = np.zeros(fbm_sample.shape)
    for i in range(fbm_sample.shape[0]):
        smoothed_sample[i] = np.mean(fbm_sample[max(0, i-filter_size):min(i+filter_size,fbm_sample.shape[0]-1)])

    pc[:,axis_alter] += fbm_sample * scale_factor

    return pc


def alter_intensity_fbm(pc, axis):
    """Alters point intensity along an axis according to a generated fBm sample"""
    pc = pc[pc[:,axis].argsort()]
    f = FBM((pc.shape[0]), np.random.uniform(0.6, 0.8))
    fbm_sample = f.fbm()[1:]
    fbm_sample -= np.median(fbm_sample)
    fbm_sample /= max(abs(fbm_sample.min()), fbm_sample.max())

    scale_factor = np.random.uniform(30, 50)
    for col in [3,4,5]:
        new_i = pc[:,col] + fbm_sample * scale_factor
        new_i[new_i<0] = 0
        new_i[new_i>255] = 255
        pc[:,col] = new_i

    return pc


def scale_tree_fbm(pc):
    """Changes the horizontal scale of a tree along the vertical axis
    according to a generated fBm sample"""
    pc = pc[pc[:,2].argsort()]
    f = FBM((pc.shape[0]), np.random.uniform(0.1, 0.8))
    fbm_sample = f.fbm()[1:]
    fbm_sample -= np.min(fbm_sample)
    fbm_sample /= fbm_sample.max()
    fbm_sample *= np.random.uniform(0.5, 1.5)
    fbm_sample += np.random.uniform(0.5, 1.5)

    pc[:,0] *= fbm_sample
    pc[:,1] *= fbm_sample

    return pc


def create_scan_lines(object, line_width, line_gap):
    """Removes point in thin strips along an axis, simulating the
    appearance of scan lines"""
    y = object[:,1].min()
    y_max = object[:,1].max()
    scan_lines = []
    while y < y_max:
        scan_line = object[object[:,1] <= y+line_width]
        scan_lines.append(scan_line)
        object =  object[ object[:,1] > y+line_width+line_gap]
        y += line_width + line_gap

    scanned_object = scan_lines[0]
    for line in scan_lines[1:]:
        scanned_object = np.r_[scanned_object, line]

    return scanned_object


def create_ghost_copy(pc):
    """Duplicates the point cloud and adds a slight offset between the copies"""
    n_points = pc.shape[0]
    ghost = pc.copy()
    ghost[:,:3] += np.random.uniform(-0.1, 0.1, 3)
    pc = np.r_[pc, ghost]
    pc = downsample(pc, n_points)

    return pc


def generate_veg_base(num_points, mean):
    """Creates a base point cloud used to generate vegetation"""
    veg = []
    for _ in range(num_points):
        az = np.random.uniform(0, 2*np.pi)
        el = np.random.uniform(0, 2*np.pi)
        r = abs(np.random.normal(mean, 1))
        x = r * np.sin(el) * np.cos(az)
        y = r * np.sin(el) * np.sin(az)
        z = r * np.cos(el)
        veg.append(np.array([x, y, z]))
    veg = np.array(veg)

    return veg


def generate_shrub(num_points, mean, target_width, target_height):
    """Creates a single shrub"""
    shrub = generate_veg_base(num_points, mean)

    width = shrub[:,0].max() - shrub[:,0].min()
    shrub[:,:2] *= target_width/width

    height = shrub[:,2].max() - shrub[:,2].min()
    shrub[:,2] *= target_height/height

    i_shrub = get_intensity(shrub, np.random.uniform(100, 150), np.random.uniform(10, 50))
    shrub_class = np.full((shrub.shape[0], 1), 99) # Temp class to separate from ground
    shrub = np.c_[shrub, i_shrub, i_shrub, i_shrub, shrub_class]

    if np.random.uniform(0,1) < 0.25:
        shrub = alter_shape_fbm(shrub, 2, np.random.choice([0,1]), np.random.uniform(0.1, 0.3))

    if np.random.uniform(0,1) < 0.25:
        shrub = fbm_point_removal(shrub, 2)

    if np.random.uniform(0,1) < 0.75:
        shrub = alter_intensity_fbm(shrub, np.random.randint(0,3))

    if np.random.uniform(0,1) < 0.25:
        shrub = create_scan_lines(shrub, np.random.uniform(0.01, 0.02), np.random.uniform(0.04, 0.08))

    return shrub


def generate_tree():
    """Creates a single tree"""
    tree_height = np.random.uniform(10, 50)
    tree_width = np.random.uniform(4, 8)
    num_points = int(np.random.uniform(10, 30) * tree_height * tree_width**2)
    tree = generate_veg_base(num_points, np.random.uniform(3,6))

    width = tree[:,0].max() - tree[:,0].min()
    tree[:,:2] *= tree_width/width

    height = tree[:,2].max() - tree[:,2].min()
    tree[:,2] *= tree_height/height

    tree = tree[tree[:,2] > tree[:,2].mean()]
    tree[:,:2] -= tree[:,:2].mean(axis=0)
    tree[:,2] -= tree[:,2].min()

    i_tree = get_intensity(tree, np.random.uniform(100, 180), np.random.uniform(10, 50))
    tree_class = np.full((tree.shape[0], 1), 99) # Temp class to separate from ground
    tree = np.c_[tree, i_tree, i_tree, i_tree, tree_class]

    if np.random.uniform(0,1) < 0.5:
        tree = alter_shape_fbm(tree, 2, np.random.choice([0,1]), np.random.uniform(0.25, 0.5)*tree_width)

    if np.random.uniform(0,1) < 0.5:
        tree = scale_tree_fbm(tree)

    if np.random.uniform(0,1) < 0.25:
        tree = fbm_point_removal(tree, 2)

    if np.random.uniform(0,1) < 0.75:
        tree = alter_intensity_fbm(tree, np.random.randint(0,3))

    if np.random.uniform(0,1) < 0.25:
        tree = create_scan_lines(tree, np.random.uniform(0.02, 0.04), np.random.uniform(0.05, 0.1))

    return tree
