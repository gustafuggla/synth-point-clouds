import numpy as np
import utils


def barriers(crossing, left_rail, right_rail, lower_road, upper_road, height_map, theta, left_barrier, right_barrier):
    dist_to_rail = np.random.uniform(3, 6)

    left_x = left_rail - dist_to_rail
    left_y = lower_road(left_x) + np.random.uniform(0, 1)
    _, i = height_map.query(np.array([left_x, left_y + 2, 0]), k=1)
    left_z = crossing[i][2]
    left_barrier = utils.rotate_pc(left_barrier, theta + np.random.uniform(-5, 5), 2)
    left_barrier[:,:3] += np.array([left_x, left_y, left_z])

    right_x = right_rail + dist_to_rail
    right_y = upper_road(right_x) + np.random.uniform(0, 1)
    _, i = height_map.query(np.array([right_x, right_y - 2, 0]), k=1)
    right_z = crossing[i][2]
    right_barrier = utils.rotate_pc(right_barrier, 180 + theta + np.random.uniform(-5, 5), 2)
    right_barrier[:,:3] += np.array([right_x, right_y, right_z])

    crossing = np.r_[crossing, left_barrier, right_barrier]

    return crossing


def cross_signs(crossing, left_rail, right_rail, lower_road, upper_road, height_map, theta, cross_signs):
    tie_offset = np.random.uniform(4, 8)
    road_offset = np.random.uniform(1, 2)
    road_center = lambda x: (lower_road(x) + upper_road(x)) / 2
    for center_x in [left_rail - tie_offset, right_rail + tie_offset]:
        lower_y, upper_y = lower_road(center_x) - road_offset, upper_road(center_x) + road_offset
        center_y = road_center(center_x)
        lower_p, upper_p = np.array([[0, lower_y - center_y, 0]]), np.array([[0, upper_y - center_y, 0]])
        lower_p, upper_p = utils.rotate_pc(lower_p, theta, 2), utils.rotate_pc(upper_p, theta, 2)
        for p in [lower_p, upper_p]:
            cross = next(cross_signs)
            cross_height = cross[:,2].max() - cross[:,2].min()
            x, y = p[0,:2] + np.array([center_x, center_y])
            _, i = height_map.query(np.array([x, y, 0]), k=1)
            z = crossing[i][2] + np.random.uniform(3.5, 4) - cross_height
            cross[:,:3] += np.array([x, y, z])
            crossing = np.r_[crossing, cross]

    return crossing


def portals(crossing, left_rail, right_rail, lower_road, upper_road, height_map, portals):
    dist_to_rail = np.random.uniform(8, 12)
    left_x = left_rail - dist_to_rail
    left_y = (lower_road(left_x) + upper_road(left_x)) / 2
    right_x = right_rail + dist_to_rail
    right_y = (lower_road(right_x) + upper_road(right_x)) / 2

    for x, y in zip([left_x, right_x], [left_y, right_y]):
        portal = next(portals)
        portal[:,:2] += np.array([x, y])
        x_min, y_min = portal[:,:2].min(axis=0)
        _, i = height_map.query(np.array([x, y, 0]), k=1)
        z = crossing[i][2]
        portal[:,2] += z
        crossing = np.r_[crossing, portal]

    return crossing


def poles(crossing, left_rail, right_rail, height_map, pole_1, pole_2):
    dist_to_rail = 2 + np.random.uniform(-0.5, 0.5)
    roll = np.random.uniform(0, 1)
    if roll < 0.25:
        x_1 = x_2 = left_rail - dist_to_rail
        pole_1 = utils.rotate_pc(pole_1, 270 + np.random.uniform(-5, 5), 2)
        pole_2 = utils.rotate_pc(pole_2, 270 + np.random.uniform(-5, 5), 2)
    elif roll < 0.5:
        x_1 = left_rail - dist_to_rail
        x_2 = right_rail + dist_to_rail
        pole_1 = utils.rotate_pc(pole_1, 270 + np.random.uniform(-5, 5), 2)
        pole_2 = utils.rotate_pc(pole_2, 90 + np.random.uniform(-5, 5), 2)
    elif roll < 0.75:
        x_1 = right_rail + dist_to_rail
        x_2 = left_rail - dist_to_rail
        pole_1 = utils.rotate_pc(pole_1, 90 + np.random.uniform(-5, 5), 2)
        pole_2 = utils.rotate_pc(pole_2, 270 + np.random.uniform(-5, 5), 2)
    else:
        x_1 = x_2 = right_rail + dist_to_rail
        pole_1 = utils.rotate_pc(pole_1, 90 + np.random.uniform(-5, 5), 2)
        pole_2 = utils.rotate_pc(pole_2, 90 + np.random.uniform(-5, 5), 2)

    y_1 = crossing[:,1].min() + np.random.uniform(1, 4)
    y_2 = crossing[:,1].max() - np.random.uniform(1, 4)
    _, i = height_map.query(np.array([x_1, y_1, 0]), k=1)
    z_1 = crossing[i][2]
    _, i = height_map.query(np.array([x_2, y_2, 0]), k=1)
    z_2 = crossing[i][2]

    offset = 0.5
    pole_1[:,:3] += np.array([x_1 + np.random.uniform(-offset, offset), y_1, z_1])
    pole_2[:,:3] += np.array([x_2 + np.random.uniform(-offset, offset), y_2, z_2])

    crossing = np.r_[crossing, pole_1, pole_2]

    return crossing


def wires(crossing, height_map, wire):
    wire = utils.rotate_pc(wire, np.random.uniform(-5, 5), 2)
    x, y = crossing[:,:2].mean(axis=0)
    x += np.random.uniform(-1, 1)
    _, i = height_map.query(np.array([x, y, 0]), k=1)
    z = crossing[i][2] + np.random.uniform(4, 6)
    wire[:,:3] += np.array([x, y, z])

    crossing = np.r_[crossing, wire]

    return crossing


def trees(crossing, left_rail, right_rail, lower_road, upper_road, height_map, trees):
    step_size = 2
    dist_to_road = 4
    dist_to_rail = 4
    dist_to_edge = 0
    min_x, max_x = crossing[:,0].min(), crossing[:,0].max()
    min_y, max_y = crossing[:,1].min(), crossing[:,1].max()
    x_candidates = np.r_[np.arange(min_x, left_rail - dist_to_rail, step_size), np.arange(right_rail + dist_to_rail, max_x, step_size)]
    xs = np.random.choice(x_candidates, min(np.random.randint(1, 4), len(x_candidates)))
    for x in xs:
        y_candidates = np.r_[np.arange(min_y + dist_to_edge, lower_road(x) - dist_to_road, step_size), np.arange(upper_road(x) + dist_to_road, max_y - dist_to_edge, step_size)]
        if len(y_candidates) > 0:
            ys = np.random.choice(y_candidates, min(np.random.randint(1, 4), len(y_candidates)))
            for y in ys:
                x, y = (x, y) + np.random.uniform(-0.3, 0.3, 2)
                _, i = height_map.query(np.array([x, y, 0]), k=1)
                z = crossing[i][2]
                tree = next(trees)
                tree = utils.scale_pc(tree, np.random.uniform(10, 15), 2)
                tree = utils.rotate_pc(tree, np.random.uniform(0, 360), 2)
                tree[:,:3] += np.array([x, y, z])
                crossing = np.r_[crossing, tree]

    return crossing
