"""Estimated string lanes derived from a current, tracked neck quadrilateral."""

import cv2
import numpy as np


def estimate_strings(corners):
    """Divide current image width into six equal lanes between tracked edges.

    Indices follow ROI side 0->1 through side 3->2, not musical string numbers.
    Centers are lane estimates, not measured strings or physical dimensions.
    """
    try:
        quad = np.float32([[p['x'], p['y']] for p in corners])
    except (TypeError, KeyError, ValueError):
        return None
    if (quad.shape != (4, 2) or not np.isfinite(quad).all() or
            (quad < 0).any() or (quad > 1).any() or
            not cv2.isContourConvex(quad) or cv2.contourArea(quad) < .002):
        return None
    centers = (np.arange(6) + .5) / 6.
    boundaries = np.arange(7) / 6.
    fractions = np.r_[centers, boundaries][:, None, None]
    upper, lower = quad[[0, 1]], quad[[3, 2]]
    mapped = upper[None] + fractions * (lower - upper)[None]
    if not np.isfinite(mapped).all():
        return None

    def line(index, y, endpoints):
        return dict(index=index, board_y=float(y), observed=False,
                    start=dict(x=float(endpoints[0, 0]), y=float(endpoints[0, 1])),
                    end=dict(x=float(endpoints[1, 0]), y=float(endpoints[1, 1])))

    return dict(status='estimated', method='live_neck_width_six_equal_lanes',
                observed=False,
                margin_fraction=None,
                edge_margins=[float(centers[0]), float(1.-centers[-1])],
                margin_source='equal_lane_centers',
                width_source='current_tracked_image_edges',
                physical_width_validated=False,
                coordinate_space='normalized_image', orientation='roi_order',
                string_numbers_known=False,
                corners=[dict(x=float(x), y=float(y)) for x, y in quad],
                strings=[line(i, y, ends) for i, (y, ends) in enumerate(zip(centers, mapped[:6]))],
                boundaries=[line(i, y, ends) for i, (y, ends) in enumerate(zip(boundaries, mapped[6:]))])


def locate_string(model, point):
    """Return a geometric lane candidate, never a pressed/sounding-string claim."""
    if not model or model.get('status') != 'estimated':
        return None
    try:
        p = np.float32([point['x'], point['y']])
        quad = np.float32([[q['x'], q['y']] for q in model['corners']])
    except (TypeError, KeyError, ValueError):
        return None
    if not np.isfinite(p).all():
        return None
    transform = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    x, projective_y = cv2.perspectiveTransform(p.reshape(1, 1, 2), transform)[0, 0]
    if not (0 <= x <= 1 and 0 <= projective_y <= 1):
        return None
    # A projective y is not a fraction of visible width on a tapered board.
    inverse = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), quad)
    ends = cv2.perspectiveTransform(np.float32([[[x, 0], [x, 1]]]), inverse)[0]
    across = ends[1] - ends[0]
    y = float(np.dot(p - ends[0], across) / np.dot(across, across))
    centers = np.array([line['board_y'] for line in model['strings']])
    boundaries = np.array([line['board_y'] for line in model['boundaries']])
    index = min(5, int(np.searchsorted(boundaries[1:-1], y, side='right')))
    pitch = float(np.median(np.diff(centers)))
    ambiguous = bool(np.min(np.abs(boundaries[1:-1]-y)) <= pitch*.15)
    return dict(string_line=index, estimated=True, ambiguous=ambiguous,
                board_x=float(x), board_y=float(y), pressed=None, sounding=None)
