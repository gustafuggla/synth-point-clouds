import numpy as np
import utils
import os


# Intensity distributions
def get_int_mean():
    return np.random.uniform(30, 220)


def get_int_stdev():
    return np.random.uniform(10, 50)


# Point density
def get_num_points(pc):
    xyz_min = pc[:,:3].min(axis=0)
    xyz_max = pc[:,:3].max(axis=0)
    xyz = xyz_max - xyz_min
    xyz.sort()
    a, b = xyz[1:]
    a, b = max(0.1, a), max(0.1, b)
    num_points = int(a*b*np.random.uniform(50, 100))

    return num_points


# Probabilities of altering a geometry
def shape_fbm_roll():
    return np.random.uniform(0,1) < 0.25


def point_fbm_roll():
    return np.random.uniform(0,1) < 0.25


def intensity_fbm_roll():
    return np.random.uniform(0,1) < 0.75


def scanline_roll():
    return np.random.uniform(0,1) < 0.25


def ghost_roll():
    return np.random.uniform(0,1) < 0.25


def noise_roll():
    return np.random.uniform(0,1) < 0.25


# Functions for getting object geometries
def barrier(samples, class_dict):
    barrier = np.random.choice(samples['barrier']).copy()

    num_points = get_num_points(barrier)*2

    if shape_fbm_roll:
        barrier = utils.alter_shape_fbm(barrier, 1, 0, np.random.uniform(0.1, 0.2))

    if shape_fbm_roll:
        barrier = utils.alter_shape_fbm(barrier, 1, 2, np.random.uniform(0.1, 0.2))

    if point_fbm_roll:
        barrier = utils.fbm_point_removal(barrier, 1)

    barrier[:,:3] *= np.random.uniform(0.8, 1.2, 3)
    barrier = utils.set_anchor_point(barrier)
    barrier = utils.rotate_pc(barrier, np.random.normal(0, 2), 0)
    barrier = utils.rotate_pc(barrier, np.random.normal(0, 5), 1)

    if scanline_roll:
        barrier = utils.create_scan_lines(barrier, np.random.uniform(0.01, 0.02), np.random.uniform(0.04, 0.08))

    barrier = utils.downsample(barrier, num_points)

    i_barrier = utils.get_intensity(barrier, get_int_mean(), get_int_stdev())
    barrier = np.c_[barrier, i_barrier, i_barrier, i_barrier]

    if intensity_fbm_roll:
        barrier = utils.alter_intensity_fbm(barrier, 1)

    barrier[:,:3] += np.random.normal(0, np.random.uniform(0, 0.01), (barrier.shape[0], 3))

    barrier_class = np.full((barrier.shape[0], 1), class_dict['barrier'])
    barrier = np.c_[barrier, barrier_class]

    if ghost_roll:
        barrier = utils.create_ghost_copy(barrier)

    if noise_roll:
        barrier[:,:3] += np.random.normal(0, np.random.uniform(0.005, 0.02), (barrier.shape[0], 3))

    return barrier


def cross_sign(theta, samples, class_dict):
    cross = np.random.choice(samples['cross']).copy()
    cross = utils.set_anchor_point(cross)

    num_points = int(get_num_points(cross)*2)

    cross[:,:3] *= np.random.uniform(0.8, 1.2, 3)

    if np.random.choice([True, False]):
        geom = np.genfromtxt(os.path.join('data/mesh_geometries/geom', np.random.choice(['circle.txt', 'triangle.txt'])))
        geom = utils.downsample(geom, np.random.randint(3000, 4000))
        geom -= geom.mean(axis=0)
        geom = utils.scale_pc(geom, np.random.uniform(0.6, 0.8), 2)
        geom[:,2] += np.random.uniform(2, 2.4)
        cross = np.r_[cross, geom]

    if shape_fbm_roll:
        cross = utils.alter_shape_fbm(cross, np.random.choice([1,2]), 0, np.random.uniform(0.05, 0.1))

    if shape_fbm_roll:
        cross = utils.alter_shape_fbm(cross, 1, 2, np.random.uniform(0.05, 0.1))

    if point_fbm_roll:
        cross = utils.fbm_point_removal(cross, 1)

    cross = utils.rotate_pc(cross, np.random.normal(0, 3), 0)
    cross = utils.rotate_pc(cross, np.random.normal(0, 3), 1)
    cross = utils.rotate_pc(cross, theta + np.random.uniform(-5, 5), 2)
    cross = cross[cross[:,2]>=np.random.uniform(0, 0.1)]

    if scanline_roll:
        cross = utils.create_scan_lines(cross, np.random.uniform(0.01, 0.02), np.random.uniform(0.03, 0.06))

    cross = utils.downsample(cross, num_points)
    cross[:,:3] += np.random.normal(0, np.random.uniform(0, 0.01), (cross.shape[0], 3))

    i_cross = utils.get_intensity(cross, get_int_mean(), get_int_stdev())
    cross_class = np.full((cross.shape[0], 1), class_dict['cross'])
    cross = np.c_[cross, i_cross, i_cross, i_cross, cross_class]

    if intensity_fbm_roll:
        cross = utils.alter_intensity_fbm(cross, 2)

    if ghost_roll:
        cross = utils.create_ghost_copy(cross)

    if noise_roll:
        cross[:,:3] += np.random.normal(0, np.random.uniform(0.005, 0.02), (cross.shape[0], 3))

    return cross


def portal(theta, samples, class_dict):
    portal = np.random.choice(samples['portal']).copy()

    num_points = int(get_num_points(portal)*0.67)

    if shape_fbm_roll:
        portal = utils.alter_shape_fbm(portal, np.random.choice([1,2]), 0, np.random.uniform(0.1, 0.3))

    if shape_fbm_roll:
        portal = utils.alter_shape_fbm(portal, 1, 2, np.random.uniform(0.1, 0.3))

    if point_fbm_roll:
        portal = utils.fbm_point_removal(portal, 1)

    portal[:,:3] *= np.random.uniform(0.8, 1.2, 3)

    portal[:,:2] -= portal[:,:2].mean(axis=0)
    portal[:,2] -= portal[:,2].min()
    portal = utils.rotate_pc(portal, np.random.normal(0, 2), 0)
    portal = utils.rotate_pc(portal, np.random.normal(0, 2), 1)
    portal = utils.rotate_pc(portal, theta + np.random.uniform(-5, 5), 2)

    if scanline_roll:
        portal = utils.create_scan_lines(portal, np.random.uniform(0.01, 0.02), np.random.uniform(0.04, 0.08))

    portal = utils.downsample(portal, num_points)
    portal[:,:3] += np.random.normal(0, np.random.uniform(0, 0.01), (portal.shape[0], 3))

    i_portal = utils.get_intensity(portal, get_int_mean(), get_int_stdev())
    portal_class = np.full((portal.shape[0], 1), class_dict['portal'])
    portal = np.c_[portal, i_portal, i_portal, i_portal, portal_class]

    if intensity_fbm_roll:
        portal = utils.alter_intensity_fbm(portal, np.random.choice([1,2]))

    if ghost_roll:
        portal = utils.create_ghost_copy(portal)

    if noise_roll:
        portal[:,:3] += np.random.normal(0, np.random.uniform(0.005, 0.02), (portal.shape[0], 3))

    return portal


def pole(samples, class_dict):
    pole = np.random.choice(samples['pole']).copy()

    pole *= np.random.uniform(0.8, 1.2)
    polearm = np.random.choice(samples['polearm']).copy()
    polearm = utils.downsample(polearm, np.random.randint(5000, polearm.shape[0]))
    polearm *= np.random.uniform(0.8, 1.2)
    if np.random.choice([True, False]):
        polearm2 = polearm.copy() + np.array([np.random.uniform(0.7, 1), 0, 0])
        polearm = np.r_[polearm, polearm2]

    pole[:,:2] -= pole[:,:2].mean(axis=0)
    pole[:,2] -= pole[:,2].min()
    pole = utils.rotate_pc(pole, np.random.choice([0, 90, 180, 270]), 2)

    polearm[:,0] -= polearm[:,0].mean()
    polearm[:,1] -= polearm[:,1].min()
    polearm[:,2] -= polearm[:,2].max()

    polearm[:,2] += pole[:,2].max() - np.random.uniform(0, 2)
    polearm[:,1] -= np.random.uniform(0, 2)
    pole = np.r_[pole, polearm]

    num_points = get_num_points(pole)

    if shape_fbm_roll:
        pole = utils.alter_shape_fbm(pole, np.random.choice([1,2]), 0, np.random.uniform(0.1, 0.3))

    if shape_fbm_roll:
        pole = utils.alter_shape_fbm(pole, 2, 1, np.random.uniform(0.05, 0.1))

    if shape_fbm_roll:
        pole = utils.alter_shape_fbm(pole, 1, 2, np.random.uniform(0.05, 0.1))

    if point_fbm_roll:
        pole = utils.fbm_point_removal(pole, 1)

    if np.random.choice([True, False]):
        pole[:,0] = -pole[:,0]

    if scanline_roll:
        pole = utils.rotate_pc(pole, 90, 2)
        pole = utils.create_scan_lines(pole, np.random.uniform(0.01, 0.02), np.random.uniform(0.03, 0.06))
        pole = utils.rotate_pc(pole, -90, 2)

    pole = utils.downsample(pole, num_points)
    pole[:,:3] += np.random.normal(0, np.random.uniform(0, 0.01), (pole.shape[0], 3))

    i_pole = utils.get_intensity(pole, get_int_mean(), get_int_stdev())
    pole_class = np.full((pole.shape[0], 1), class_dict['pole'])
    pole = np.c_[pole, i_pole, i_pole, i_pole, pole_class]

    if intensity_fbm_roll:
        pole = utils.alter_intensity_fbm(pole, 2)

    if ghost_roll:
        pole = utils.create_ghost_copy(pole)

    if noise_roll:
        pole[:,:3] += np.random.normal(0, np.random.uniform(0.005, 0.02), (pole.shape[0], 3))

    return pole


def single_wire(samples, x, z, y_min, y_max, class_dict):
    wire = np.random.choice(samples['wire'].copy())

    num_points = get_num_points(wire)

    wire *= np.random.uniform(0.8, 1.2)

    if scanline_roll:
        wire = utils.create_scan_lines(wire, np.random.uniform(0.01, 0.02), np.random.uniform(0.03, 0.06))

    wire -= wire.mean(axis=0)
    wire[:,0] += x + np.random.uniform(-0.2, 0.2)
    wire[:,2] += z + np.random.uniform(-0.2, 0.2)

    wire = wire[wire[:,1] > y_min - 1]
    wire = wire[wire[:,1] < y_max + 1]
    wire = utils.downsample(wire, num_points)

    i_wire = utils.get_intensity(wire, get_int_mean(), get_int_stdev())
    wire_class = np.full((wire.shape[0], 1), class_dict['wire'])
    wire = np.c_[wire, i_wire, i_wire, i_wire, wire_class]

    if intensity_fbm_roll:
        wire = utils.alter_intensity_fbm(wire, 1)

    wire[:,:3] += np.random.normal(0, np.random.uniform(0.01, 0.03), (wire.shape[0], 3))

    if shape_fbm_roll:
        wire = utils.alter_shape_fbm(wire, 1, 0, np.random.uniform(0.05, 0.1))

    if shape_fbm_roll:
        wire = utils.alter_shape_fbm(wire, 1, 2, np.random.uniform(0.05, 0.1))

    if point_fbm_roll:
        wire = utils.fbm_point_removal(wire, 1)

    return wire


def wire(samples, scene_y_min, scene_y_max, class_dict):
    # Create grid in xz plane and add wires
    xs = np.linspace(0, np.random.uniform(2, 6), np.random.randint(2, 4))
    wires = []
    for x in xs:
        zs = np.linspace(0, np.random.uniform(1.5,2.5), np.random.randint(1, 4))
        for z in zs:
            wires.append(single_wire(samples, x, z, scene_y_min, scene_y_max, class_dict))

    # Combine wires to single array
    wire = wires[0]
    for next_wire in wires[1:]:
        wire = np.r_[wire, next_wire]

    # Alter height with second degree curve
    a = np.random.uniform(0, 0.002)
    z = lambda y: a * y**2
    wire = wire[wire[:,1].argsort()]
    for p in wire:
        p[2] += z(p[1])

    # Center
    wire[:,:2] -= wire[:,:2].mean(axis=0)
    wire[:,2] -= wire[:,2].min()

    if noise_roll:
        wire[:,:3] += np.random.normal(0, np.random.uniform(0.005, 0.02), (wire.shape[0], 3))

    if ghost_roll:
        wire = utils.create_ghost_copy(wire)

    return wire
