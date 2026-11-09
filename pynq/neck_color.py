"""Automatic colour segmentation and fitted-border neck proposals."""

import ctypes
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


HEADSTOCK_END_RAISE = .006


@lru_cache(maxsize=1)
def native_backend():
    path = Path(__file__).with_name('libneck_color.so')
    if not path.is_file():
        return None
    native = ctypes.CDLL(str(path))
    native.neck_color_borders.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    native.neck_color_borders.restype = ctypes.c_int
    native.neck_color_ridges.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    native.neck_color_ridges.restype = ctypes.c_int
    native.neck_color_candidates.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    native.neck_color_candidates.restype = ctypes.c_int
    return native


def find_color_neck_regions(bgr, limit=3, require_crossbars=True):
    """Segment colours, fit contour borders, and validate elongated strips.

    No manual ROI, calibrated colour, Hough search or DNN is used. A proposal
    is not semantic guitar recognition or a numbered string/fret lattice.
    """
    _validate(bgr, limit, require_crossbars)
    height, width = bgr.shape[:2]
    scale = min(1., 240. / width, 180. / height)
    small = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    native = native_backend()
    if native is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        output = np.empty((limit, 64), np.float32)
        count = native.neck_color_candidates(lab.ctypes.data, gray_small.ctypes.data,
            small.shape[1], small.shape[0], gray.ctypes.data, width, height,
            limit, int(require_crossbars), output.ctypes.data)
        if not 0 <= count <= limit or not np.isfinite(output[:count]).all():
            raise RuntimeError('Invalid native colour candidates')
        proposals = []
        for row in output[:count]:
            points = row[:8].reshape(4, 2)
            crossbar_count = int(row[13])
            if (points < 0).any() or (points > 1).any() or not 0 <= crossbar_count <= 42:
                raise RuntimeError('Invalid native colour coordinates')
            proposals.append(dict(corners=[dict(x=float(x), y=float(y)) for x, y in points],
                score=float(row[8]), border_color_support=float(row[9]),
                interior_color_support=float(row[10]), color_delta_lab8=float(row[11]),
                crossbars=dict(supported=bool(row[12]), count=crossbar_count,
                    run_count=int(row[14]), span=float(row[15]),
                    positions=row[16:16 + crossbar_count].astype(float).tolist()),
                acquisition='color_border_candidate', algorithm='color_region_contour_fit',
                lattice_confirmed=False))
        return proposals
    lines = _region_borders(lab)
    if len(lines) < 2:
        return []
    return _pair_borders(lines, lab, small, bgr, limit, require_crossbars)


def _region_borders(lab):
    native = native_backend()
    if native is not None:
        lab = np.ascontiguousarray(lab)
        output = np.empty((64, 2, 2), np.float32)
        count = native.neck_color_borders(lab.ctypes.data, lab.shape[1], lab.shape[0], output.ctypes.data)
        if not 0 <= count <= 64:
            raise RuntimeError('Invalid native colour border result')
        return output[:count]
    smooth = cv2.GaussianBlur(lab, (3, 3), .6)
    # Histogram arithmetic must not wrap in uint8: L alone needs 11 bins.
    pixels = smooth.reshape(-1, 3).astype(np.int32)
    codes = (pixels[:, 0] // 24) * 169 + (pixels[:, 1] // 20) * 13 + pixels[:, 2] // 20
    counts = np.bincount(codes, minlength=1859)
    centers, segments = [], []
    tolerance = np.array([26, 16, 16], np.int16)
    side = max(lab.shape[:2])
    minimum = side * .08
    for code in np.argsort(-counts, kind='stable'):
        if counts[code] < max(40, len(pixels) * .004):
            break
        lightness, remainder = divmod(int(code), 169)
        a, b = divmod(remainder, 13)
        center = np.array([lightness * 24 + 12, a * 20 + 10, b * 20 + 10], np.int16)
        if any(np.all(np.abs(center - old) < tolerance) for old in centers):
            continue
        centers.append(center)
        mask = cv2.inRange(smooth, np.clip(center - tolerance, 0, 255).astype(np.uint8),
                           np.clip(center + tolerance, 0, 255).astype(np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.arcLength(contour, True) < side * .35:
                continue
            polygon = cv2.approxPolyDP(contour, 2.5, True)[:, 0, :]
            vectors = np.roll(polygon, -1, axis=0) - polygon
            eligible = np.sum(vectors * vectors, axis=1) >= minimum * minimum
            if eligible.any():
                segments.extend(np.stack((polygon[eligible],
                                          polygon[eligible] + vectors[eligible]), axis=1))
        if len(centers) == 10:
            break
    if not segments:
        return np.empty((0, 2, 2), np.float32)
    return _fit_borders(np.asarray(segments, np.float32), side)


def _fit_borders(lines, side):
    lengths = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
    lines = lines[np.argsort(-lengths, kind='stable')[:128]]
    vectors = lines[:, 1] - lines[:, 0]
    axes = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    centers = lines.mean(axis=1)
    normals = np.column_stack((-axes[:, 1], axes[:, 0]))
    agreement = np.abs(axes @ axes.T)
    distances = np.abs(np.einsum('ijk,ik->ij', centers[None] - centers[:, None], normals))
    projections = np.einsum('jpk,ik->ijp', lines, axes)
    own = np.einsum('ipk,ik->ip', lines, axes)
    eligible = ((agreement > .993) & (distances <= side * .01) &
                (projections.min(axis=2) <= own.max(axis=1)[:, None] + side * .2) &
                (projections.max(axis=2) >= own.min(axis=1)[:, None] - side * .2))
    fitted = []
    for index in range(len(lines)):
        points = lines[eligible[index]].reshape(-1, 2)
        result = cv2.fitLine(points, cv2.DIST_HUBER, 0, .01, .01).ravel()
        axis, center = result[:2], result[2:]
        extent = (points - center) @ axis
        fitted.append([center + extent.min() * axis, center + extent.max() * axis])
    combined = np.concatenate((np.asarray(fitted), lines))
    lengths = np.linalg.norm(combined[:, 1] - combined[:, 0], axis=1)
    combined = combined[np.argsort(-lengths, kind='stable')]
    axes = combined[:, 1] - combined[:, 0]
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    normals = np.column_stack((-axes[:, 1], axes[:, 0]))
    centers = combined.mean(axis=1)
    agreement = np.abs(axes @ axes.T)
    distances = np.abs(np.einsum('ijk,ik->ij', centers[None] - centers[:, None], normals))
    duplicate = (agreement >= .999) & (distances <= 1.5)
    projections = np.einsum('jpk,ik->ijp', combined, axes)
    own = np.einsum('ipk,ik->ip', combined, axes)
    duplicate &= ((projections.max(axis=2) >= own.min(axis=1)[:, None]) &
                  (projections.min(axis=2) <= own.max(axis=1)[:, None]))
    selected, removed = [], np.zeros(len(combined), bool)
    for index in range(len(combined)):
        if removed[index]:
            continue
        selected.append(index)
        removed |= duplicate[index]
        if len(selected) == 64:
            break
    return combined[selected]


def _pair_borders(lines, lab, small, bgr, limit, require_crossbars):
    sh, sw = small.shape[:2]
    vectors = lines[:, 1] - lines[:, 0]
    axes = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    axes[(axes[:, 0] < 0) | ((axes[:, 0] == 0) & (axes[:, 1] < 0))] *= -1
    first, second = np.triu_indices(len(lines), 1)
    agreement = np.sum(axes[first] * axes[second], axis=1)
    aligned = np.abs(agreement) > .992
    first, second = first[aligned], second[aligned]
    if not len(first):
        return []
    axis = axes[first] + np.where(agreement[aligned] >= 0, 1., -1.)[:, None] * axes[second]
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    normal = np.column_stack((-axis[:, 1], axis[:, 0]))
    pair = lines[np.column_stack((first, second))]
    u = np.sum(pair * axis[:, None, None], axis=-1)
    v = np.sum(pair * normal[:, None, None], axis=-1)
    low, high = u.min(axis=(1, 2)), u.max(axis=(1, 2))
    overlap = u.max(axis=2).min(axis=1) - u.min(axis=2).max(axis=1)
    mid = (low + high) * .5
    slopes = (v[:, :, 1] - v[:, :, 0]) / (u[:, :, 1] - u[:, :, 0])
    offsets = v[:, :, 0] + slopes * (mid[:, None] - u[:, :, 0])
    swap = offsets[:, 0] > offsets[:, 1]
    slopes[swap] = slopes[swap, ::-1]
    offsets[swap] = offsets[swap, ::-1]
    separation = offsets[:, 1] - offsets[:, 0]
    ends_width = separation[:, None] + (slopes[:, 1] - slopes[:, 0])[:, None] * (np.column_stack((low, high)) - mid[:, None])
    keep = ((separation >= sw * .018) & (separation <= sw * .19) & (high - low >= sw * .28) &
            ((high - low) / np.maximum(separation, 1.) >= 4) & (overlap >= sw * .09) &
            (ends_width.min(axis=1) >= sw * .015) & (ends_width.max(axis=1) <= sw * .22))
    if not keep.any():
        return []
    axis, normal, low, high, mid, slopes, offsets, separation = (
        value[keep] for value in (axis, normal, low, high, mid, slopes, offsets, separation))
    ranked = np.argsort(-(high - low), kind='stable')[:96]
    axis, normal, low, high, mid, slopes, offsets, separation = (
        value[ranked] for value in (axis, normal, low, high, mid, slopes, offsets, separation))
    samples_u = low[:, None] + (high - low)[:, None] * np.linspace(.04, .96, 40)[None]
    border_v = offsets[:, :, None] + slopes[:, :, None] * (samples_u[:, None] - mid[:, None, None])
    distance = np.clip(separation * .14, 1.5, 4.)
    samples_v = np.stack((border_v[:, 0] - distance[:, None],
                          border_v[:, 0] + distance[:, None],
                          (border_v[:, 0] + border_v[:, 1]) * .5,
                          border_v[:, 1] - distance[:, None],
                          border_v[:, 1] + distance[:, None]), axis=1)
    locations = samples_u[:, None, :, None] * axis[:, None, None] + samples_v[:, :, :, None] * normal[:, None, None]
    valid = ((locations[..., 0] >= 1) & (locations[..., 0] < sw - 1) &
             (locations[..., 1] >= 1) & (locations[..., 1] < sh - 1))
    valid = valid.all(axis=1)
    colors = cv2.remap(lab, locations[..., 0].reshape(-1, 40).astype(np.float32),
                       locations[..., 1].reshape(-1, 40).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    colors = colors.reshape(len(axis), 5, 40, 3).astype(np.float32)
    delta0 = np.linalg.norm(colors[:, 0] - colors[:, 1], axis=-1)
    delta1 = np.linalg.norm(colors[:, 4] - colors[:, 3], axis=-1)
    interior = np.linalg.norm(colors[:, 1] - colors[:, 3], axis=-1)
    support0 = ((delta0 >= 18) & valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    support1 = ((delta1 >= 18) & valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    inside_support = ((interior <= 45) & valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    balance = np.minimum(support0, support1)
    supported = ((valid.sum(axis=1) >= 20) & (balance >= .30) & (inside_support >= .30))
    color_strength = np.minimum(np.median(delta0, axis=1), np.median(delta1, axis=1))
    scores = (high - low) * balance * inside_support * np.minimum(color_strength, 70)
    proposals = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    source_scale = np.float32([bgr.shape[1] / sw, bgr.shape[0] / sh])
    pixel_scale = np.float32([bgr.shape[1] - 1, bgr.shape[0] - 1])
    ordered = np.flatnonzero(supported)
    ordered = ordered[np.argsort(-scores[ordered], kind='stable')][:32]
    examined = []
    for index in ordered:
        start, end = low[index], high[index]
        for side in (0, 1):
            direction = axis[index] + slopes[index, side] * normal[index]
            origin = (offsets[index, side] - slopes[index, side] * mid[index]) * normal[index]
            for dimension, maximum in enumerate((sw - 1, sh - 1)):
                if abs(direction[dimension]) < 1e-8:
                    continue
                bounds = (np.array([0., maximum]) - origin[dimension]) / direction[dimension]
                start, end = max(start, bounds.min()), min(end, bounds.max())
        if end - start < sw * .28:
            continue
        ends_u = np.array([start, end, end, start])
        sides = np.array([0, 0, 1, 1])
        ends_v = offsets[index, sides] + slopes[index, sides] * (ends_u - mid[index])
        quad = ends_u[:, None] * axis[index] + ends_v[:, None] * normal[index]
        if (quad < -1e-4).any() or (quad[:, 0] > sw - 1 + 1e-4).any() or (quad[:, 1] > sh - 1 + 1e-4).any():
            continue
        quad = np.clip(quad, 0, [sw - 1, sh - 1])
        if any(_overlap(quad, prior) > .97 for prior in examined):
            continue
        examined.append(quad.astype(np.float32))
        if len(examined) > 16:
            break
        quad = _refine_quad(lab, quad)
        quad = _extend_quad(lab, quad)
        full_quad = (quad + .5) * source_scale - .5
        normalized = full_quad / pixel_scale
        if (normalized < 0).any() or (normalized > 1).any():
            continue
        if any(_overlap(normalized, np.float32([[point['x'], point['y']] for point in prior['corners']])) > .8
               for prior in proposals):
            continue
        crossbars = _crossbar_support(gray, full_quad, shear=True)
        if not crossbars['supported']:
            crossbars = _crossbar_support(gray_small, quad, shear=True)
        if require_crossbars and not crossbars["supported"]:
            continue
        if crossbars['supported']:
            normalized = _clip_quad_to_crossbars(
                normalized, crossbars['positions'],
                crossbars.get('ridge_support'), crossbars.get('ridge_strength'))
        tracking_quad = (normalized * [319, 199]).astype(np.float32)
        lengths = np.linalg.norm(tracking_quad - np.roll(tracking_quad, 1, axis=0), axis=1)
        area = cv2.contourArea(tracking_quad)
        if (not cv2.isContourConvex(tracking_quad) or lengths.min() < 8 or
                area < 319 * 199 * .002 or area / lengths.max() < 8):
            continue
        proposals.append({"corners": [dict(x=float(x), y=float(y)) for x, y in normalized],
                          "score": float(scores[index]), "border_color_support": float(balance[index]),
                          "interior_color_support": float(inside_support[index]),
                          "color_delta_lab8": float(color_strength[index]),
                          "crossbars": crossbars,
                          "acquisition": "color_border_candidate", "algorithm": "color_region_contour_fit",
                          "lattice_confirmed": False})
        if len(proposals) == limit:
            break
    return proposals


def _refine_quad(lab, quad):
    refined = quad.astype(np.float32).copy()
    for first, last in ((0, 1), (3, 2)):
        start, end = quad[first], quad[last]
        axis = end - start
        length = np.linalg.norm(axis)
        normal = np.float32([-axis[1], axis[0]]) / length
        samples = start + np.linspace(.03, .97, 48)[:, None] * axis
        offsets = np.arange(-2., 2.1, .5, dtype=np.float32)
        locations = samples[:, None] + offsets[None, :, None] * normal
        colors = []
        for shift in (-1.5, 1.5):
            points = locations + shift * normal
            colors.append(cv2.remap(lab, points[..., 0].astype(np.float32),
                points[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE).astype(np.float32))
        strength = np.linalg.norm(colors[1] - colors[0], axis=-1) - abs(offsets) * 3
        best = strength.argmax(axis=1)
        supported = strength[np.arange(48), best] >= 18
        if supported.sum() < 24:
            continue
        points = locations[np.arange(48), best][supported].astype(np.float32)
        fitted = cv2.fitLine(points, cv2.DIST_HUBER, 0, .01, .01).ravel()
        direction, center = fitted[:2], fitted[2:]
        extent = (np.float32([start, end]) - center) @ direction
        moved = center + extent[:, None] * direction
        if (np.linalg.norm(moved - [start, end], axis=1) <= 2.5).all():
            refined[first], refined[last] = moved
    return refined


def _extend_quad(lab, quad):
    axis = (quad[1] + quad[2] - quad[0] - quad[3]) * .5
    length = np.linalg.norm(axis)
    axis /= length
    normal = np.float32([-axis[1], axis[0]])
    parameters = np.linspace(-.8, 1.8, 160, dtype=np.float32)
    upper = quad[0] + parameters[:, None] * (quad[1] - quad[0])
    lower = quad[3] + parameters[:, None] * (quad[2] - quad[3])
    gap = np.linalg.norm(lower - upper, axis=1)
    distance = np.clip(gap * .14, 1.5, 4.)
    points = np.stack((upper - distance[:, None] * normal,
                       upper + distance[:, None] * normal,
                       lower - distance[:, None] * normal,
                       lower + distance[:, None] * normal))
    height, width = lab.shape[:2]
    valid = ((points[..., 0] >= 1) & (points[..., 0] < width - 1) &
             (points[..., 1] >= 1) & (points[..., 1] < height - 1)).all(axis=0)
    colors = cv2.remap(lab, points[..., 0].astype(np.float32), points[..., 1].astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    support = ((np.linalg.norm(colors[0] - colors[1], axis=-1) >= 18) &
               (np.linalg.norm(colors[3] - colors[2], axis=-1) >= 18) &
               (np.linalg.norm(colors[1] - colors[2], axis=-1) <= 55) & valid)
    bounds = [0., 1.]
    for direction, initial in ((-1, np.searchsorted(parameters, 0.)),
                                (1, np.searchsorted(parameters, 1.) - 1)):
        misses = 0
        selected = initial
        for index in range(initial, -1 if direction < 0 else len(parameters), direction):
            if not valid[index]:
                break
            misses = 0 if support[index] else misses + 1
            if misses >= 6:
                break
            if support[index]:
                selected = index
        if direction < 0:
            bounds[0] = min(0., float(parameters[selected]))
        else:
            bounds[1] = max(1., float(parameters[selected]))
    first, last = bounds
    return np.float32([quad[0] + first * (quad[1] - quad[0]),
                       quad[0] + last * (quad[1] - quad[0]),
                       quad[3] + last * (quad[2] - quad[3]),
                       quad[3] + first * (quad[2] - quad[3])])


def _regular_crossbar_run(positions):
    """Return the longest physically plausible contiguous fret sequence."""
    positions = np.asarray(positions, np.float32)
    if (positions.ndim != 1 or len(positions) < 5
            or not np.isfinite(positions).all()
            or (np.diff(positions) <= 0).any()):
        return None
    best = None
    for start in range(len(positions) - 4):
        previous_gap = None
        smallest_gap = float("inf")
        largest_gap = 0.
        for end in range(start + 1, len(positions)):
            gap = float(positions[end] - positions[end - 1])
            if previous_gap is not None:
                ratio = gap / previous_gap
                if ratio < .60 or ratio > 1.60:
                    break
            previous_gap = gap
            smallest_gap = min(smallest_gap, gap)
            largest_gap = max(largest_gap, gap)
            if largest_gap > smallest_gap * 2.2:
                break
            count = end - start + 1
            span = float(positions[end] - positions[start])
            if count < 5 or span < .25:
                continue
            candidate = (count, span, -start)
            if best is None or candidate > best[0]:
                best = (candidate, positions[start:end + 1].copy())
    return None if best is None else best[1]


def _headstock_side(positions):
    """Return the crossbar end that should carry the nut-side hard limit."""
    regular = _regular_crossbar_run(positions)
    if regular is None:
        return None
    gaps = np.diff(regular)
    if len(gaps) < 5 or not np.isfinite(gaps).all() or (gaps <= 0).any():
        return None
    # Projective fret spacing changes gradually. Fit the whole regular run
    # instead of comparing only the edge gaps, which is too noisy for a short
    # neck and can hide a real perspective trend.
    indexes = np.arange(len(gaps), dtype=np.float32)
    slope = float(np.polyfit(indexes, gaps, 1)[0] / np.mean(gaps))
    # Fret spacing shrinks towards the nut under perspective projection.
    if slope < -.0075:
        return 1
    if slope > .0075:
        return 0
    return None


def _nut_ridge_boundary(positions, supports, strengths, headstock):
    """Find the nut as the terminal full-width ridge beyond the fret run."""
    if headstock not in (0, 1):
        return None
    positions = np.asarray(positions, np.float32)
    supports = np.asarray(supports, np.float32)
    strengths = np.asarray(strengths, np.float32)
    if (positions.ndim != 1 or supports.shape != positions.shape or
            strengths.shape != positions.shape or not len(positions)):
        return None
    regular = _regular_crossbar_run(positions)
    if regular is None:
        return None
    # A nut is a broad, high-contrast crossbar. Headstock studs and tuners
    # can be detected as narrow ridges, but do not cover most of the width.
    strong = (supports >= .52) & (strengths >= 18.)
    if headstock == 1:
        eligible = np.flatnonzero(strong & (positions >= regular[-1] - 1e-4))
        return None if not len(eligible) else float(positions[eligible[-1]])
    eligible = np.flatnonzero(strong & (positions <= regular[0] + 1e-4))
    return None if not len(eligible) else float(positions[eligible[0]])


def _clip_quad_to_crossbars(normalized, positions, ridge_support=None, ridge_strength=None):
    """Clip a neck quad without allowing the nut-side line to reach the headstock."""
    observed = np.asarray(positions, np.float32)
    positions = observed
    regular = _regular_crossbar_run(observed)
    reliable = regular is not None
    if reliable:
        positions = regular
    headstock = _headstock_side(positions) if reliable else None
    gaps = np.diff(positions) if len(positions) > 1 else np.empty(0, np.float32)
    edge_count = min(3, len(gaps))
    edge_gap = (float(np.median(gaps[-edge_count:]))
                if edge_count else .0)
    leading_gap = (float(np.median(gaps[:edge_count]))
                   if edge_count else .0)
    nut_boundary = _nut_ridge_boundary(
        observed, ridge_support, ridge_strength, headstock) if (
            ridge_support is not None and ridge_strength is not None) else None
    if headstock == 1:
        start = max(0., float(observed[0]))
        end = float(positions[-1])
        if nut_boundary is not None and nut_boundary >= end:
            # Keep the fitted boundary itself; the tracker may use the nut as
            # its endpoint, while headstock hardware stays outside the ROI.
            end = min(1., nut_boundary)
        elif edge_gap > 0:
            nut_limit = min(1., end + edge_gap * 1.5)
            candidates = observed[(observed > end + 1e-4) &
                                  (observed <= nut_limit + 1e-4)]
            if len(candidates):
                end = float(candidates[-1])
            else:
                candidates = observed[observed > end + 1e-4]
                if len(candidates) and candidates[0] - end <= edge_gap * 2.8:
                    # A hand can hide one fret. Cross that single large gap,
                    # then admit only the next two observed boundaries so the
                    # headstock tuners and logo remain outside the ROI.
                    previous = end
                    for candidate in candidates[:2]:
                        if candidate - previous > edge_gap * 2.8:
                            break
                        end = float(candidate)
                        previous = end
    elif headstock == 0:
        start = float(positions[0])
        end = min(1., float(observed[-1]))
        if nut_boundary is not None and nut_boundary <= start:
            start = max(0., nut_boundary)
        elif leading_gap > 0:
            nut_limit = max(0., start - leading_gap * 1.5)
            candidates = observed[(observed < start - 1e-4) &
                                  (observed >= nut_limit - 1e-4)]
            if len(candidates):
                start = float(candidates[0])
            else:
                candidates = observed[observed < start - 1e-4][::-1]
                if len(candidates) and start - candidates[0] <= leading_gap * 2.8:
                    previous = start
                    for candidate in candidates[:2]:
                        if previous - candidate > leading_gap * 2.8:
                            break
                        start = float(candidate)
                        previous = start
    else:
        # Without a reliable orientation, never extrapolate at either end.
        # A slightly short proposal is safer than crossing a nut.
        start = float(positions[0])
        end = float(positions[-1])
    result = np.array([
        normalized[0] * (1 - start) + normalized[1] * start,
        normalized[0] * (1 - end) + normalized[1] * end,
        normalized[3] * (1 - end) + normalized[2] * end,
        normalized[3] * (1 - start) + normalized[2] * start])
    if end - start < .20:
        return np.asarray(normalized).copy()
    # The nut-side colour boundary is easily pulled down by the darker
    # headstock joint. Raise only that cross-section by a small image-relative
    # amount; the opposite endpoint and the full-length taper remain intact.
    headstock = _headstock_side(positions) if reliable else None
    if headstock == 1:
        result[[1, 2], 1] = np.maximum(
            0., result[[1, 2], 1] - HEADSTOCK_END_RAISE)
    elif headstock == 0:
        result[[0, 3], 1] = np.maximum(
            0., result[[0, 3], 1] - HEADSTOCK_END_RAISE)
    return result


def _validate(bgr, limit, require_crossbars):
    """Find elongated strips with observed colour changes at both long edges.

    No colour is calibrated or learned from a supplied ROI. Coordinates use
    the original frame's first/last pixel centres, as neck_region does.
    A proposal is not a semantic guitar detection or measured wire lattice.
    Crossbar support rejects plain strips without requiring six strings or
    assigning fret numbers. Disable it only for offline candidate inspection.
    """
    if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or
            bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 24):
        raise ValueError("Expected a nonempty HWC uint8 BGR image")
    if type(limit) is not int or not 1 <= limit <= 6:
        raise ValueError("Candidate limit must be from 1 through 6")
    if type(require_crossbars) is not bool:
        raise ValueError("Expected boolean require_crossbars")


def _crossbar_support(gray, quad, shear=False):
    native = native_backend()
    if native is not None:
        gray = np.ascontiguousarray(gray)
        points = np.ascontiguousarray(quad, np.float32)
        output = np.empty(259, np.float32)
        supported = native.neck_color_ridges(gray.ctypes.data, gray.shape[1], gray.shape[0],
            points.ctypes.data, int(shear), output.ctypes.data)
        count = int(output[0]) if supported >= 0 else -1
        if supported not in (0, 1) or not 0 <= count <= 42:
            raise RuntimeError('Invalid native crossbar result')
        return dict(supported=bool(supported), count=count, span=float(output[2]),
                    run_count=int(output[1]), positions=output[3:3 + count].astype(float).tolist(),
                    ridge_support=output[64:64 + count].astype(float).tolist(),
                    ridge_strength=output[106:106 + count].astype(float).tolist())
    destination = np.float32([[0, 0], [255, 0], [255, 39], [0, 39]])
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    patch = cv2.warpPerspective(gray, transform, (256, 40), flags=cv2.INTER_LINEAR)
    best = _ridge_support(patch)
    if shear and not best['supported']:
        for amount in (-.5, .5, -1., 1.):
            # Oblique frets need not be vertical in a strip-aligned crop.
            matrix = np.float32([[1, amount, -19.5 * amount], [0, 1, 0]])
            adjusted = cv2.warpAffine(patch, matrix, (256, 40), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
            result = _ridge_support(adjusted)
            if (result['supported'], result['run_count'], result['count']) > (
                    best['supported'], best['run_count'], best['count']):
                best = result
            if best['supported']:
                break
    return best


def _ridge_support(patch):
    image = patch[5:-5].astype(np.float32)
    # A fret is a narrow bright or dark ridge extending across the strip.
    # The row vote suppresses isolated texture and partial finger edges.
    best = {"supported": False, "count": 0, "span": 0., "run_count": 0,
            "positions": [], "ridge_support": [], "ridge_strength": []}
    for sign in (1., -1.):
        values = image * sign
        ridge = np.zeros_like(values)
        ridge[:, 2:-2] = np.minimum(values[:, 2:-2] - values[:, :-4],
                                   values[:, 2:-2] - values[:, 4:])
        support = (cv2.dilate(ridge, np.ones((1, 3), np.uint8)) > 8).mean(axis=0)
        profile = np.maximum(ridge, 0).mean(axis=0)
        baseline = np.median(profile)
        threshold = max(4., baseline + 2 * np.median(abs(profile - baseline)))
        peaks = np.flatnonzero((profile[1:-1] >= profile[:-2]) &
                               (profile[1:-1] > profile[2:]) &
                               (profile[1:-1] >= threshold) &
                               (support[1:-1] >= .45)) + 1
        selected = []
        for index in sorted(peaks, key=lambda x: -profile[x]):
            if 5 <= index <= 250 and all(abs(index - old) >= 6 for old in selected):
                selected.append(int(index))
        selected.sort()
        span = (selected[-1] - selected[0]) / 255. if selected else 0.
        run_count = 0
        # Require a contiguous, smoothly spaced run. A few unrelated chair or
        # garment edges separated by one large gap cannot establish a neck.
        positions = np.array(selected, np.float32)
        for count in range(len(selected), 4, -1):
            windows = positions[np.arange(len(selected) - count + 1)[:, None] + np.arange(count)]
            gaps = np.diff(windows, axis=1)
            ratios = gaps[:, 1:] / gaps[:, :-1]
            valid = ((gaps.max(axis=1) <= gaps.min(axis=1) * 2.2) &
                     (ratios >= .60).all(axis=1) & (ratios <= 1.60).all(axis=1) &
                     (windows[:, -1] - windows[:, 0] >= 64))
            if valid.any():
                run_count = count
                break
        supported = run_count >= 5
        if (supported, run_count, len(selected)) > (best["supported"], best["run_count"], best["count"]):
            best = dict(supported=bool(supported), count=len(selected), span=float(span),
                        run_count=run_count,
                        positions=[float(x / 255.) for x in selected],
                        ridge_support=[float(support[x]) for x in selected],
                        ridge_strength=[float(profile[x]) for x in selected])
    return best


def _overlap(first, second):
    intersection = cv2.intersectConvexConvex(first.astype(np.float32), second.astype(np.float32))[0]
    return intersection / max(min(cv2.contourArea(first.astype(np.float32)), cv2.contourArea(second)), 1e-9)
