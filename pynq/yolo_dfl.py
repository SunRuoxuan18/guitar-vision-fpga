"""NumPy decoder for a batch-one YOLOv8 DFL detection head.

The input is the concatenation of the stride-8, stride-16, and stride-32
raw heads before Ultralytics' DFL, anchor, and sigmoid operations.
"""


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def _validated_strides(strides):
    if not isinstance(strides, (tuple, list)) or not strides:
        raise ValueError("strides must be a nonempty sequence of positive integers")
    result = tuple(_positive_integer(stride, "stride") for stride in strides)
    if len(set(result)) != len(result):
        raise ValueError("strides must not contain duplicates")
    return result


def _input_dimensions(input_size):
    if isinstance(input_size, bool):
        raise ValueError("input_size must be a positive integer or [width, height]")
    if isinstance(input_size, int):
        width = height = input_size
    elif isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        width, height = input_size
    else:
        raise ValueError("input_size must be a positive integer or [width, height]")
    width = _positive_integer(width, "input_size width")
    height = _positive_integer(height, "input_size height")
    return width, height


def _sigmoid(values, np):
    """Return sigmoid(values) without overflowing for finite input logits."""
    result = np.empty(values.shape, dtype=values.dtype)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    negative_values = values[~nonnegative]
    negative_exp = np.exp(negative_values)
    result[~nonnegative] = negative_exp / (1.0 + negative_exp)
    return result


def decode_dfl(output, input_size, classes_count, reg_max=16, strides=(8, 16, 32)):
    """Decode raw YOLOv8 DFL logits to ``[1, 4 + C, N]`` cxcywh proposals.

    ``input_size`` is either a square integer or ``[width, height]`` in
    pixels. The candidate order is row-major within each stride scale, with
    the scales in the supplied ``strides`` order.
    """
    import numpy as np

    input_width, input_height = _input_dimensions(input_size)
    classes_count = _positive_integer(classes_count, "classes_count")
    reg_max = _positive_integer(reg_max, "reg_max")
    strides = _validated_strides(strides)
    if any(input_width % stride or input_height % stride for stride in strides):
        raise ValueError("input_size must be divisible by every stride")

    expected_candidates = sum((input_width // stride) * (input_height // stride)
                              for stride in strides)
    channels = 4 * reg_max + classes_count
    try:
        raw = np.asarray(output)
    except (TypeError, ValueError) as error:
        raise ValueError("output must be a numeric tensor") from error
    if raw.dtype.kind not in "fiu":
        raise ValueError("output tensor must contain real numeric values")
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError("output must have exact shape [1, 4*reg_max+C, N]")
    if raw.shape[1] != channels:
        raise ValueError("output channel count must equal 4*reg_max + classes_count")
    if raw.shape[2] != expected_candidates:
        raise ValueError("output candidate count does not match input_size and strides")
    if not np.isfinite(raw).all():
        raise ValueError("output tensor contains nonfinite values")

    # The shifted logits make the DFL softmax safe for all finite values while
    # float32 keeps the post-processing practical on ARM/PYNQ.
    raw = raw.astype(np.float32, copy=False)
    candidates = raw.shape[2]
    logits = raw[:, :4 * reg_max, :].reshape(1, 4, reg_max, candidates)
    shifted = logits - logits.max(axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    bins = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1)
    distances = (probabilities * bins).sum(axis=2)

    anchor_parts = []
    stride_parts = []
    for stride in strides:
        grid_width = input_width // stride
        grid_height = input_height // stride
        y, x = np.meshgrid(np.arange(grid_height, dtype=np.float32),
                           np.arange(grid_width, dtype=np.float32), indexing="ij")
        anchor_parts.append(np.stack((x, y), axis=-1).reshape(-1, 2) + 0.5)
        stride_parts.append(np.full(grid_height * grid_width, stride, dtype=np.float32))
    anchors = np.concatenate(anchor_parts, axis=0).T
    stride_per_anchor = np.concatenate(stride_parts).reshape(1, 1, candidates)

    centers = anchors.reshape(1, 2, candidates) * stride_per_anchor
    distances *= stride_per_anchor
    left_top = centers - distances[:, :2, :]
    right_bottom = centers + distances[:, 2:, :]
    cxcywh = np.concatenate(((left_top + right_bottom) / 2.0,
                              right_bottom - left_top), axis=1)
    classes = _sigmoid(raw[:, 4 * reg_max:, :], np)
    return np.concatenate((cxcywh, classes), axis=1)
