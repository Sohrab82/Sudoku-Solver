import numpy as np


def reorder(points):
    points = points.reshape((4, 2))
    points_ordered = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    points_ordered[0] = points[np.argmin(add)]
    points_ordered[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_ordered[1] = points[np.argmin(diff)]
    points_ordered[2] = points[np.argmax(diff)]
    return points_ordered


def center_of_contour(c):
    # calculate the center of mass of a contour
    x = 0
    y = 0
    for pt in c:
        x += pt[0][0]
        y += pt[0][1]
    return (x / float(len(c)), y / float(len(c)))


def calc_distance(pt1, pt2):
    # calculates distance between two points
    return np.math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def is_around_point(cnt, pt0, r):
    # returns true if any of the points in cnt are in radius of r to pt
    for pt in cnt:
        if calc_distance(pt[0], pt0) < r:
            return True
    return False
