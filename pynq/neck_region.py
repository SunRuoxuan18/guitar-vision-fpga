"""Propose a neck region from long borders and repeated transverse wires.

This is a region proposal, never evidence for six strings or absolute frets.
"""
import cv2
import numpy as np
import ctypes
from pathlib import Path

_NATIVE = None
_NATIVE_TRIED = False


def native_backend():
    global _NATIVE, _NATIVE_TRIED
    if not _NATIVE_TRIED:
        _NATIVE_TRIED = True
        path = Path(__file__).with_name('libneck_region.so')
        if path.is_file():
            try:
                library = ctypes.CDLL(str(path))
                library.neck_border_candidates.argtypes = [ctypes.c_void_p] + [ctypes.c_int]*5 + [ctypes.c_void_p]
                library.neck_border_candidates.restype = ctypes.c_int
                if hasattr(library, 'tracking_well_spread'):
                    library.tracking_well_spread.argtypes = [ctypes.c_void_p, ctypes.c_int]
                    library.tracking_well_spread.restype = ctypes.c_int
                if hasattr(library, 'tracking_skin_component'):
                    library.tracking_skin_component.argtypes = ([ctypes.c_void_p]*5 +
                        [ctypes.c_int]*3 + [ctypes.c_float]*2 + [ctypes.c_int]*2 + [ctypes.c_void_p])
                    library.tracking_skin_component.restype = ctypes.c_int
                _NATIVE = library
            except (OSError, AttributeError):
                pass
    return _NATIVE


def find_neck_region(gray):
    candidates = find_neck_regions(gray,limit=1)
    return candidates[0] if candidates else None


def find_neck_regions(gray, limit=6):
    """Return bounded border proposals; downstream wire checks choose geometry."""
    if type(limit) is not int or not 1 <= limit <= 6:
        raise ValueError('Neck candidate limit must be from 1 through 6')
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), .6), 35, 90)
    candidates = _find_neck_regions(gray, 6, edges)
    # Two fixed edge passes handle low-contrast fragments behind a hand.
    # Compare observed length, independent of neck position or direction.
    # This runs only during background acquisition, never on the track path.
    alternate = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), 30, 80)
    recovered = _find_neck_regions(gray, 6, alternate)
    def extent(quad):
        points = np.float32([[p['x']*(gray.shape[1]-1), p['y']*(gray.shape[0]-1)]
                             for p in quad])
        return np.linalg.norm(points[1]-points[0])+np.linalg.norm(points[2]-points[3])
    if recovered and (not candidates or max(map(extent,recovered)) > max(map(extent,candidates))):
        candidates = recovered
    return candidates[:limit]


def find_neck_regions_fast(gray, edge=None, limit=6, alternate=False, require_frets=True):
    """Bound acquisition cost using one edge pass and a capped segment count.

    All border/fret geometry tests still operate in original pixel units.
    The gray path needs only one Canny pass. An optional PL edge image can
    be reused without recalculating gradients.
    """
    if type(limit) is not int or not 1 <= limit <= 6:
        raise ValueError('Neck candidate limit must be from 1 through 6')
    height, width = gray.shape
    scale = min(1., 448. / width)
    shape = (round(width * scale), round(height * scale))
    if edge is not None:
        reduced = cv2.resize(edge, shape, interpolation=cv2.INTER_AREA)
        edges = (reduced >= 40).astype(np.uint8) * 255
    else:
        reduced = cv2.resize(gray, shape, interpolation=cv2.INTER_LINEAR)
        low, high = (25, 70) if alternate else (30, 80)
        edges = cv2.Canny(cv2.GaussianBlur(reduced, (5, 5), 1.0), low, high)
    return _find_neck_regions(gray, limit, edges, scale, 384, require_frets=require_frets,
                                    fine_accumulator=True, quick_lines=True)


def _find_neck_regions(gray, limit, edges, scale=1., max_segments=None, require_frets=False,
                       fine_accumulator=False, quick_lines=False):
    # Acquisition uses a coarser accumulator; final borders use observed endpoints.
    rho, theta = ((3, np.pi/180) if max_segments is not None and not fine_accumulator
                  else (1, np.pi/360))
    if quick_lines:
        lines = cv2.HoughLinesP(edges,1,np.pi/180,50,
            minLineLength=max(10,round(22*scale)),maxLineGap=max(3,round(9*scale)))
    else:
        lines = cv2.HoughLinesP(edges, rho, theta, max(15, round(30*scale)),
                                minLineLength=max(10, round(22*scale)),
                                maxLineGap=max(3, round(9*scale)))
    if lines is None:
        return []
    segments = lines[:, 0].reshape(-1, 2, 2).astype(float) / scale
    if max_segments is not None and len(segments) > max_segments:
        order = np.argsort(-np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1))
        half = max_segments // 2
        # Keep long borders and a deterministic spread of shorter fret wires.
        rest = order[half:]
        selected = np.concatenate((order[:half], rest[np.linspace(
            0, len(rest)-1, max_segments-half).astype(int)]))
        segments = segments[selected]
    native = native_backend() if max_segments is not None else None
    if native is not None:
        packed = np.ascontiguousarray(segments, dtype=np.float32)
        output = np.empty((limit, 4, 2), np.float32)
        count = native.neck_border_candidates(packed.ctypes.data, len(packed),
            gray.shape[1], gray.shape[0], limit, int(require_frets), output.ctypes.data)
        if not 0 <= count <= limit:
            raise RuntimeError('Invalid native neck candidate count')
        return [[dict(x=float(x), y=float(y)) for x,y in quad] for quad in output[:count]]
    vectors = segments[:, 1]-segments[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    long = np.argsort(-lengths)[:12]
    best = None
    proposals = []
    for i in long:
        if lengths[i] < 110:
            continue
        axis = vectors[i]/lengths[i]
        if axis[0] < 0:
            axis = -axis
        normal = np.array([-axis[1], axis[0]])
        uv = np.stack((segments@axis, segments@normal), axis=-1)
        slopes = (vectors@normal)/np.maximum(abs(vectors@axis), 1)
        for j in long:
            if j <= i or lengths[j] < 110 or abs(slopes[j]) > .12:
                continue
            # Fingers split opposite borders into disjoint visible segments.
            # Their union proposes a region; the transverse-wire and downstream
            # six-string checks still have to establish a physical lattice.
            lo = min(uv[i,:,0].min(), uv[j,:,0].min())
            hi = max(uv[i,:,0].max(), uv[j,:,0].max())
            if hi-lo < 130:
                continue
            center = (lo+hi)/2
            borders = []
            for k in (i,j):
                a,b = uv[k]
                slope = (b[1]-a[1])/(b[0]-a[0])
                borders.append((slope, a[1]+slope*(center-a[0])))
            borders.sort(key=lambda b:b[1])
            width = borders[1][1]-borders[0][1]
            if not 18 <= width <= 110 or (hi-lo)/width < 3:
                continue
            middle = uv.mean(axis=1)
            cross = (abs(vectors@axis) < abs(vectors@normal)*.5) & (lengths >= width*.45)
            lower = borders[0][1]+borders[0][0]*(middle[:,0]-center)
            upper = borders[1][1]+borders[1][0]*(middle[:,0]-center)
            cross &= (uv[:,:,1].min(axis=1) <= lower+width*.3)
            cross &= (uv[:,:,1].max(axis=1) >= upper-width*.3)
            available_cross = cross.copy()
            cross &= (middle[:,0] >= lo-4) & (middle[:,0] <= hi+4)
            xs = []
            for x in sorted(middle[cross,0]):
                if not xs or x-xs[-1] > 5:
                    xs.append(float(x))
            if require_frets and len(xs) < 3:
                continue
            if xs and xs[-1]-xs[0] < max(110,width*2.5):
                continue
            lattice_confirmed = False
            if len(xs) >= 5:
                gaps = np.diff(xs)
                # Regular grids are not enough to establish a guitar neck.
                scale_gaps = 2.**(np.arange(len(gaps))/12.)
                progression = float(np.std(scale_gaps)/np.mean(scale_gaps))
                if (np.std(gaps)/np.mean(gaps) < progression*.45 or
                        np.max(gaps) > np.median(gaps)*3):
                    continue
                lattice_confirmed = True
            elif len(xs) > 0:
                # Border geometry can seed ROI LK before enough fret evidence
                # exists. This is deliberately a proposal, never a lattice.
                pass
            # Continue only through nearby observed transverse wires; a hand
            # may break the long border before the end of the visible neck.
            all_xs = []
            for x in sorted(middle[available_cross,0]):
                if not all_xs or x-all_xs[-1]>5:
                    all_xs.append(float(x))
            first,last = (xs[0],xs[-1]) if xs else (lo,hi)
            gap_limit = (float(np.median(np.diff(xs)))*1.8
                         if len(xs) > 1 else (hi-lo))
            for x in [x for x in all_xs if x>last+5]:
                if x-last>gap_limit:
                    break
                gap_limit = (x-last)*1.5
                last = x
            gap_limit = (float(np.median(np.diff(xs)))*1.8
                         if len(xs) > 1 else (hi-lo))
            for x in reversed([x for x in all_xs if x<first-5]):
                if first-x>gap_limit:
                    break
                gap_limit = (first-x)*1.5
                first = x
            # Occluded frets must not crop independently observed long borders.
            ends = (min(lo, first-4), max(hi, last+4))
            # Long segments may be two inner strings. Repeated transverse
            # wires supply an independent envelope, rather than cropping the
            # outer strings according to that inner pair's narrow spacing.
            slope = (borders[0][0]+borders[1][0])*.5
            transverse = uv[available_cross,:,1]-slope*(uv[available_cross,:,0]-center)
            if len(transverse):
                low = float(np.quantile(transverse.min(axis=1),.20))-2
                high = float(np.quantile(transverse.max(axis=1),.80))+2
            else:
                low, high = borders[0][1], borders[1][1]
            tight_low = min(borders[0][1]-2,low)
            tight_high = max(borders[1][1]+2,high)
            low = min(borders[0][1]-width*.6,low)
            high = max(borders[1][1]+width*.6,high)
            if high-low > min(180,width*5):
                continue
            borders = [(borders[0][0],low),(borders[1][0],high)]
            quad = np.array([[u,b[1]+b[0]*(u-center)] for b,u in
                ((borders[0],ends[0]),(borders[0],ends[1]),
                 (borders[1],ends[1]),(borders[1],ends[0]))])
            quad = quad[:,:1]*axis+quad[:,1:]*normal
            if (quad < 0).any() or (quad > np.array([gray.shape[1]-1,gray.shape[0]-1])).any():
                continue
            score = max(1, len(xs))*(ends[1]-ends[0])*min(width,60)
            if not lattice_confirmed:
                score *= .35
            if best is None or score > best[0]:
                best = score, quad
            if limit == 1:
                continue
            proposals.append((score,quad))
            # Inner strings need the wider envelope; actual outer borders do
            # not. Keep both until six observed wires can distinguish them.
            tight = np.array([[u,b[1]+b[0]*(u-center)] for b,u in
                (((borders[0][0],tight_low),ends[0]),((borders[0][0],tight_low),ends[1]),
                 ((borders[1][0],tight_high),ends[1]),((borders[1][0],tight_high),ends[0]))])
            tight = tight[:,:1]*axis+tight[:,1:]*normal
            if (tight >= 0).all() and (tight <= np.array([gray.shape[1]-1,gray.shape[0]-1])).all():
                proposals.append((score,tight))
    if best is None:
        return []
    selected = [best[1]]
    for _,quad in sorted(proposals,key=lambda item:-item[0]):
        if len(selected) >= limit:
            break
        if all(np.linalg.norm(quad-old,axis=1).mean() > 7 for old in selected):
            selected.append(quad)
    return [[dict(x=float(x/(gray.shape[1]-1)),y=float(y/(gray.shape[0]-1))) for x,y in quad]
            for quad in selected]
