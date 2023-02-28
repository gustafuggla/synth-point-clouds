import os
import numpy as np
import utils
from fbm import FBM
from scipy.spatial import KDTree
import place_objects as place
import get_mesh_objects as get
from multiprocessing import Pool
import time


def get_samples():
    """Creates a disctionary of object types"""
    samples = {}
    for object_type in ['barrier', 'cross', 'portal', 'pole', 'polearm', 'wire', 'box', 'sign', 'geom']:
        object_dir = os.path.join(SAMPLE_DIR, object_type)
        sample_list = []
        for file_name in os.listdir(object_dir):
            sample_list.append(np.genfromtxt(os.path.join(object_dir, file_name)))
        samples[object_type] = sample_list

    return samples


def undulate_height_fbm(ground, axis, left_rail, right_rail, lower_road, upper_road):
    """Alters the ground elevation according to a generated fBm sample. Ther magnitude
    is based on the distance to the road and the railroad"""

    # Create smoothed fBm sample
    filter_size = np.random.randint(10, 50)
    scale_factor = np.random.uniform(0.05, 0.3)
    vertical_shift = np.random.uniform(-1, 0)
    ground = ground[ground[:,axis].argsort()]
    f = FBM((ground.shape[0]), np.random.uniform(0.4, 0.9))
    fbm_sample = f.fbm()
    smoothed_sample = np.zeros(fbm_sample.shape)
    for i in range(fbm_sample.shape[0]):
        smoothed_sample[i] = np.mean(fbm_sample[max(0, i-filter_size):min(i+filter_size,fbm_sample.shape[0]-1)])
    smoothed_sample -= smoothed_sample.min()
    smoothed_sample /= smoothed_sample.max()
    smoothed_sample += vertical_shift

    # Change elevation of ground points
    for i, p in enumerate(ground):
        if p[0] < left_rail or p[0] > right_rail:
            if p[1] < lower_road(p[0]) or p[1] > upper_road(p[0]):
                magnitude = np.sqrt(min(abs(p[0] - left_rail), abs(p[0] - right_rail)) * min(abs(p[1] - lower_road(p[0])), abs(p[1] - upper_road(p[0]))))
                p[2] += magnitude * smoothed_sample[i] * scale_factor

    return ground


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def create_ground():
    """Creates the initial ground surface"""
    width = 60
    length = np.random.uniform(15, 40)
    point_density = np.random.randint(50, 1500) # Number of points per sqm
    n_points = np.sqrt(point_density) # Number of points per m
    xs = np.linspace(0, width,int(n_points*width))
    ys = np.linspace(0, length, int(n_points*length))
    xy = cartesian_product(xs, ys)
    z = np.zeros((xy.shape[0], 1))
    ground = np.c_[xy, z]
    ground[:,0] -= ground[:,0].mean()

    # Create railroad boundaries
    n_rails = np.random.randint(1,4)
    rail_width = np.random.uniform(2.4, 2.6) * n_rails
    left_rail, right_rail = -rail_width/2, rail_width/2

    # Create road boundaries
    road_width = np.random.uniform(3, 8)
    road_center_y = np.random.uniform(length/2 - 3, length/2 + 3)
    road_attitude = np.random.uniform(-0.3, 0.3)
    lower_road = lambda x: road_center_y - road_width/2 + road_attitude * x
    upper_road = lambda x: road_center_y + road_width/2 + road_attitude * x

    # Add noise and crop width
    ground += np.random.normal(0, np.random.uniform(0.005, 0.02), (ground.shape[0], 3))
    new_width = np.random.uniform(25, 40) + rail_width
    ground = ground[ground[:,0] > -new_width/2]
    ground = ground[ground[:,0] < new_width/2]

    # Create height undulations along x and y axes
    ground = undulate_height_fbm(ground, 0, left_rail, right_rail, lower_road, upper_road)
    ground = undulate_height_fbm(ground, 1, left_rail, right_rail, lower_road, upper_road)

    # Add intensity and class
    i_ground = utils.get_intensity(ground, np.random.uniform(50, 150), np.random.uniform(10, 50))
    ground_class = np.full((ground.shape[0], 1), CLASS_DICT['ground'])
    ground = np.c_[ground, i_ground, i_ground, i_ground, ground_class]
    ground = utils.alter_intensity_fbm(ground, np.random.randint(0,3))

    # Alter ground
    ground = utils.alter_intensity_fbm(ground, np.random.randint(0,3))
    if np.random.choice([True, False]):
        ground = utils.fbm_point_removal(ground, np.random.randint(0,2))

    return ground, left_rail, right_rail, lower_road, upper_road


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
    scene = add_poles(scene, left_rail, right_rail, height_map)
    scene = add_wires(scene, height_map)
    scene = add_boxes_and_signs(scene, left_rail, right_rail, lower_road, upper_road, height_map)
    scene = add_vegetation(scene, left_rail, right_rail, lower_road, upper_road, height_map)

    return scene


def add_barriers(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta):
    left_barrier = get.barrier(SAMPLES, CLASS_DICT)
    right_barrier = get.barrier(SAMPLES, CLASS_DICT)
    scene = place.barriers(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta, left_barrier, right_barrier)

    return scene


def add_cross_signs(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta):
    cross_signs = (get.cross_sign(theta, SAMPLES, CLASS_DICT) for _ in range(4))
    scene = place.cross_signs(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta, cross_signs)

    return scene


def add_portals(scene, left_rail, right_rail, lower_road, upper_road, height_map, theta):
    portals = (get.portal(theta, SAMPLES, CLASS_DICT) for _ in range(2))
    scene = place.portals(scene, left_rail, right_rail, lower_road, upper_road, height_map, portals)

    return scene


def add_poles(scene, left_rail, right_rail, height_map):
    pole_1 = get.pole(SAMPLES, CLASS_DICT)
    pole_2 = get.pole(SAMPLES, CLASS_DICT)
    scene = place.poles(scene, left_rail, right_rail, height_map, pole_1, pole_2)

    return scene


def add_wires(scene, height_map):
    wire = get.wire(SAMPLES, SCENE_Y_MIN, SCENE_Y_MAX, CLASS_DICT)
    scene = place.wires(scene, height_map, wire)
    scene = crop_scene(scene)

    return scene


def get_box():
    """Gets a single cable box geometry"""
    # Get and center box sample
    box = np.random.choice(SAMPLES['box']).copy()
    box[:,:2] -= box[:,:2].mean()
    box[:,2] -= box[:,2].min()
    box *= np.random.uniform(0.8, 1.2, 3)

    # Scale and determine number of points
    num_points = np.random.randint(200, 1000)
    scale_factor = np.random.uniform(1, 3)
    box *= scale_factor
    num_points = int(num_points * scale_factor**2)

    # Rotate
    box = utils.rotate_pc(box, np.random.uniform(0, 360), 2)

    if np.random.uniform(0,1) < 0.25:
        box = utils.create_scan_lines(box, np.random.uniform(0.01, 0.02), np.random.uniform(0.03, 0.06))

    # Downsample
    box = utils.downsample(box, num_points)

    # Add intensity and class
    i_box = utils.get_intensity(box, np.random.uniform(120, 180), np.random.uniform(20, 50))
    box_class = np.full((box.shape[0], 1), 99) # Temp class to separate from ground
    box = np.c_[box, i_box, i_box, i_box, box_class]

    return box


def get_sign():
    """Gets a single sign geometry"""
    # Get and center sign sample
    sign = np.random.choice(SAMPLES['sign']).copy()
    sign[:,:2] -= sign[:,:2].mean()
    sign[:,2] -= sign[:,2].min()
    sign *= np.random.uniform(0.8, 1.2, 3)

    # Rotate
    sign = utils.rotate_pc(sign, np.random.uniform(0, 360), 2)

    if np.random.uniform(0,1) < 0.25:
        sign = utils.create_scan_lines(sign, np.random.uniform(0.01, 0.02), np.random.uniform(0.03, 0.06))

    # Downsample
    sign = utils.downsample(sign, np.random.randint(300, 1500))

    # Add intensity and class
    i_sign = utils.get_intensity(sign, np.random.uniform(120, 180), np.random.uniform(20, 50))
    sign_class = np.full((sign.shape[0], 1), 99) # Temp class to separate from ground
    sign = np.c_[sign, i_sign, i_sign, i_sign, sign_class]

    return sign


def add_boxes_and_signs(scene, left_rail, right_rail, lower_road, upper_road, height_map):
    """Add cable boxes and signs, classified as "other", to the scene"""
    # Determine y coordinates
    min_y, max_y = scene[:,1].min(), scene[:,1].max()
    y_candidates = np.r_[np.arange(min_y, lower_road(left_rail) - 2, 2), np.arange(upper_road(left_rail) + 2, max_y, 2)]
    ys = np.random.choice(y_candidates, np.random.randint(3, 8))
    for y in ys:

        # Choose box or sign
        if np.random.choice([True, False]):
            obj = get_box()
        else:
            obj = get_sign()

        # Determine x coordinates
        dist_to_rail = np.random.uniform(2, 4)
        x = np.random.choice([left_rail - dist_to_rail, right_rail + dist_to_rail])

        # Determine height and place object
        _, i = height_map.query(np.array([x, y, 0]), k=1)
        z = scene[i][2]
        obj[:,:3] += np.array([x, y, z])
        scene = np.r_[scene, obj]

    return scene


def imitate_grass(p, left_rail, right_rail, lower_road, upper_road, stdev):
    "Changes elevation of ground points to imitate grass"
    if p[6] == 0:
        if p[0] < left_rail or p[0] > right_rail:
            if p[1] < lower_road(p[0]) or p[1] > upper_road(p[0]):
                p[2] += abs(np.random.normal(0, stdev))

    return p


def add_shrubs(scene, left_rail, right_rail, lower_road, upper_road, height_map, min_cand, max_cand, min_size, max_size):
    """Place shrubs in scene"""

    # Find x and y coordinates
    step_size = 0.5
    dist_to_rail = 1
    dist_to_road = 1
    min_x, max_x = scene[:,0].min(), scene[:,0].max()
    min_y, max_y = scene[:,1].min(), scene[:,1].max()
    x_candidates = np.r_[np.arange(min_x, left_rail - dist_to_rail, step_size), np.arange(right_rail + dist_to_rail, max_x, step_size)]
    xs = np.random.choice(x_candidates, min(np.random.randint(min_cand, max_cand), len(x_candidates)))
    for x in xs:
        y_candidates = np.r_[np.arange(min_y, lower_road(x) - dist_to_road, step_size), np.arange(upper_road(x) + dist_to_road, max_y, step_size)]
        if len(y_candidates) > 0:
            ys = np.random.choice(y_candidates, min(np.random.randint(min_cand, max_cand), len(y_candidates)))
            for y in ys:

                # Get shrub
                size = np.random.uniform(min_size, max_size)
                num_points = int(np.random.uniform(300, 500) * size**2)
                shrub = utils.generate_shrub(np.random.randint(50, 200), np.random.uniform(3, 6), size, size)

                # Center
                shrub[:,:2] -= shrub[:,:2].mean(axis=0)
                shrub[:,2] -= shrub[:,2].min()

                # Place
                x, y = (x, y) + np.random.uniform(-0.5, 0.5, 2)
                _, i = height_map.query(np.array([x, y, 0]), k=1)
                z = scene[i][2]
                shrub[:,:3] += np.array([x, y, z])

                scene = np.r_[scene, shrub]

    return scene


def add_trees(scene, left_rail, right_rail, lower_road, upper_road, height_map):
    trees = (utils.generate_tree() for _ in range(9))
    scene = place.trees(scene, left_rail, right_rail, lower_road, upper_road, height_map, trees)
    scene = crop_scene(scene)

    return scene


def add_vegetation(scene, left_rail, right_rail, lower_road, upper_road, height_map):
    """Add vegetation to scene. All vegetation is given a temp class of 99 to
    separate it from the ground points, whichi is necessary to imitate grass."""

    # Add shrubbery and trees
    scene = add_shrubs(scene, left_rail, right_rail, lower_road, upper_road, height_map, 5, 15, 0.5, 1)
    scene = add_shrubs(scene, left_rail, right_rail, lower_road, upper_road, height_map, 3, 8, 1, 2)
    scene = add_trees(scene, left_rail, right_rail, lower_road, upper_road, height_map)

    # Imitate grass
    stdev = np.random.uniform(0.02, 0.15)
    scene = np.apply_along_axis(imitate_grass, 1, scene, left_rail, right_rail, lower_road, upper_road, stdev)
    scene[:,6][scene[:,6] == 99] = 0 # Set temporary veg class to ground

    scene = crop_scene(scene)

    return scene


def crop_scene(scene):
    """Crops scene to its original extents"""
    scene = scene[scene[:,0] >= SCENE_X_MIN]
    scene = scene[scene[:,1] >= SCENE_Y_MIN]
    scene = scene[scene[:,0] <= SCENE_X_MAX]
    scene = scene[scene[:,1] <= SCENE_Y_MAX]

    return scene


def generate_scene(i):
    """Creates a single scene with index i"""

    # Set random seed - necessary for pool processing and repeatability
    np.random.seed(i)

    # Create scene
    scene, left_rail, right_rail, lower_road, upper_road = create_ground()

    # Set ground extents, used to crop edges of scene
    global SCENE_X_MIN, SCENE_Y_MIN, SCENE_X_MAX, SCENE_Y_MAX
    SCENE_X_MIN, SCENE_Y_MIN = scene[:,:2].min(axis=0)
    SCENE_X_MAX, SCENE_Y_MAX = scene[:,:2].max(axis=0)

    # Add objects to scene
    scene = add_objects(scene, left_rail, right_rail, lower_road, upper_road)

    # Mirror
    if np.random.choice([True, False]):
        scene[:,0] = -scene[:,0]
    if np.random.choice([True, False]):
        scene[:,1] = -scene[:,1]

    # Center
    scene[:,:2] -= scene[:,:2].mean(axis=0)

    # Save scene
    if SAVETXT:
        np.savetxt(os.path.join(TXT_DIR, f'scene{i}.txt'), scene)
    else:
        utils.create_blocks(scene, f'{i}', H5_DIR, 1024)


if __name__ == '__main__':
    # Object classes
    CLASS_DICT = {'ground': 0,
                  'cross': 1,
                  'portal': 2,
                  'barrier': 3,
                  'wire': 4,
                  'pole': 5}

    # Object samples
    SAMPLE_DIR = 'data/mesh_geometries'
    SAMPLES = get_samples()

    # Parameters
    SAVETXT = True
    TXT_DIR = ''
    H5_DIR = ''
    n_cpu = 4
    n_scenes = 480

    # Create scenes
    start_time = time.time()
    p = Pool(n_cpu)
    p.map(generate_scene, range(n_scenes))
    p.close()
    print(f'Created {n_scenes} scenes in {time.time() - start_time} seconds')
