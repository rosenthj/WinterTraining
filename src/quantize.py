"""Fixed point export of the relative nets Winter evaluates.

Winter evaluates the net in 16 bit fixed point. Each accumulator dimension gets
its own scale, chosen so that the accumulator provably cannot leave the 16 bit
range for any legal position, and the output weights absorb the reciprocal of
that scale so per dimension scaling costs nothing at run time.

The scales are derived here rather than in the engine so that the net file
carries them, but they are a pure function of the weights: no positions and no
calibration data are involved. The engine recomputes the same bounds in its
tests and asserts that the stored scales are safe.

The one exception is the pooled head of NetRelHDP (format version 2). Its
inputs are the per lane sums of the piece output head, and the bound provable
from the weights is about a hundred times larger than anything seen in play,
which would waste most of their 16 bits. Their scales are therefore calibrated
on positions, with a margin, and the engine saturates rather than wraps when a
lane exceeds its range. Everything that could overflow, including the 32 bit
accumulation of the head's hidden layer, is still bounded from the weights.
"""

import numpy as np


MAGIC = b"WNET"
FORMAT_VERSION = 1
POOLED_FORMAT_VERSION = 2
FLAG_QUANTIZED = 1

MAX_INT16 = 32767
MAX_INT32 = 2 ** 31 - 1
RELU_BOUND = 8.0
NUM_PIECE_TYPES = 12
GRID = 15
NUM_OUTPUTS = 3
MAX_PIECES = 32

# A side has at most 15 non king pieces, at most 8 of which are pawns.
# Promotions allow up to 10 of any other type. Index order is Winter's:
# pawn, knight, bishop, rook, queen, king.
MAX_PIECE_COUNT = (8, 10, 10, 10, 10, 1)
MAX_NON_KING = 15
MAX_CANDIDATES = 10

# Headroom of the pooled head's input range over the largest lane seen in calibration.
CALIBRATION_MARGIN = 4.0


def _as_array(x):
    """Accept torch tensors as well as anything numpy understands."""
    detach = getattr(x, "detach", None)
    if detach is not None:
        x = detach().to("cpu")
    return np.asarray(x, np.float64)


def _mirror_square(sq):
    return (sq % 8) + 8 * (7 - sq // 8)


def engine_tensors(c1_weight, b1, out_weight, f1_weight, f1_bias, fout_weight, d, fd):
    """Rebuild the weights in the layout Winter evaluates in.

    Winter stores a piece relation as one vector of ``2 * d`` values: the first
    half is the net as trained, the second half is the same net seen from the
    other side. Both halves come from the tensors below, the engine builds them
    at load time and so do we, because the bounds have to be taken over what the
    engine actually accumulates.

    Returns ``(W, B, OW, MOW, FL, FLB, FO, MFO)``:
      W[src piece][dst piece][dy][dx][channel]  relation weights
      B[dst piece][square][channel]             bias, self relation folded in
      OW[piece][square][outcome][channel]       piece output head, and its mirror
      FL[piece][square][channel], FLB[channel]  global accumulator and its bias
      FO[outcome][channel]                      global output head, and its mirror
    """
    block, full_block = 2 * d, 2 * fd
    mirror_pt = np.array([(p + 6) % NUM_PIECE_TYPES for p in range(NUM_PIECE_TYPES)])
    mirror_dy = np.arange(GRID - 1, -1, -1)
    mirror_rank = np.arange(7, -1, -1)
    mirror_sq = np.array([_mirror_square(sq) for sq in range(64)])

    c1 = _as_array(c1_weight).reshape(NUM_PIECE_TYPES, d, NUM_PIECE_TYPES, GRID, GRID)
    bias = _as_array(b1).reshape(NUM_PIECE_TYPES, d, 8, 8)
    out = _as_array(out_weight).reshape(NUM_OUTPUTS, 2, NUM_PIECE_TYPES, d, 8, 8)
    f1w = _as_array(f1_weight).reshape(fd, NUM_PIECE_TYPES, 64)
    f1b = _as_array(f1_bias).reshape(fd)
    fow = _as_array(fout_weight).reshape(NUM_OUTPUTS, full_block)

    # Relation weights. The conv is indexed by (dst piece, channel, src piece).
    unmirrored = c1.transpose(2, 0, 3, 4, 1)
    W = np.zeros((NUM_PIECE_TYPES, NUM_PIECE_TYPES, GRID, GRID, block))
    W[..., :d] = unmirrored
    W[..., d:] = unmirrored[np.ix_(mirror_pt, mirror_pt, mirror_dy)]

    # Bias. The relation of a piece to itself is constant, so Winter folds it in
    # here instead of visiting it during accumulation.
    self_relation = W[np.arange(NUM_PIECE_TYPES), np.arange(NUM_PIECE_TYPES), 7, 7]
    B = np.zeros((NUM_PIECE_TYPES, 8, 8, block))
    B[..., :d] = bias.transpose(0, 2, 3, 1) + self_relation[:, None, None, :d]
    B[..., d:] = bias[mirror_pt].transpose(0, 2, 3, 1)[:, mirror_rank] \
        + self_relation[mirror_pt][:, None, None, :d]
    B = B.reshape(NUM_PIECE_TYPES, 64, block)

    OW = np.zeros((NUM_PIECE_TYPES, 8, 8, NUM_OUTPUTS, block))
    OW[..., :d] = out[:, 0].transpose(1, 3, 4, 0, 2)
    OW[..., d:] = out[:, 1][:, mirror_pt].transpose(1, 3, 4, 0, 2)[:, mirror_rank]
    OW = OW.reshape(NUM_PIECE_TYPES, 64, NUM_OUTPUTS, block)
    MOW = np.concatenate([OW[np.ix_(mirror_pt, mirror_sq)][..., d:],
                          OW[np.ix_(mirror_pt, mirror_sq)][..., :d]], axis=-1)

    FL = np.zeros((NUM_PIECE_TYPES, 64, full_block))
    FL[..., :fd] = f1w.transpose(1, 2, 0)
    FL[..., fd:] = f1w.transpose(1, 2, 0)[np.ix_(mirror_pt, mirror_sq)]
    FLB = np.concatenate([f1b, f1b])
    FO = fow
    MFO = np.concatenate([fow[:, fd:], fow[:, :fd]], axis=1)
    return W, B, OW, MOW, FL, FLB, FO, MFO


def _bound_side(candidates):
    """Largest total one side can contribute to an accumulator dimension.

    ``candidates`` has shape ``(..., 6, k)`` and holds, per piece type, the
    contributions that type could make, sorted with the most extreme first. The
    king is always on the board, every other piece is optional and is only
    counted while it pushes the accumulator further out. Every source piece
    stands on its own square, so no two of them can claim the same candidate.
    """
    per_type = [candidates[..., t, :MAX_PIECE_COUNT[t]] for t in range(5)]
    pool = np.maximum(np.concatenate(per_type, axis=-1), 0.0)
    pool = np.sort(pool, axis=-1)[..., ::-1][..., :MAX_NON_KING]
    return pool.sum(-1) + candidates[..., 5, 0]


def _extremes(values):
    """The MAX_CANDIDATES largest values descending, and the smallest negated."""
    ordered = np.sort(values, axis=-1)[..., ::-1]
    return ordered[..., :MAX_CANDIDATES], -ordered[..., -MAX_CANDIDATES:][..., ::-1]


def _scale_from_bound(bound):
    return np.maximum((MAX_INT16 / np.maximum(bound, RELU_BOUND)).astype(np.int64), 1)


def piece_scales(W, B):
    """Scales for the piece accumulator, from a bound on how far it can travel.

    A piece receives one relation weight per other piece on the board, taken
    from the offset grid of that pair of piece types. Two source pieces are
    always on different squares and therefore at different offsets, so bounding
    a side by the most extreme distinct entries of the grid is valid. Doing so
    over the whole grid rather than per destination square costs a few percent
    of tightness and a great deal of time. The grid also holds the offset of the
    destination square itself, which no source piece can occupy, so the bound is
    slightly conservative.
    """
    block = W.shape[-1]
    grid = W.reshape(NUM_PIECE_TYPES, NUM_PIECE_TYPES, GRID * GRID, block)
    bound = np.zeros(block)
    for dst in range(NUM_PIECE_TYPES):
        # (channel, src piece, offset) -> per type extremes
        up, down = _extremes(grid[:, dst].transpose(2, 0, 1))
        hi = B[dst].max(0) + _bound_side(up[:, :6]) + _bound_side(up[:, 6:])
        lo = B[dst].min(0) - _bound_side(down[:, :6]) - _bound_side(down[:, 6:])
        bound = np.maximum(bound, np.maximum(hi, -lo))
    return _scale_from_bound(bound)


def full_scales(FL, FLB):
    """Scales for the global accumulator, where a piece contributes one weight
    per square it can stand on."""
    up, down = _extremes(FL.transpose(2, 0, 1))
    hi = FLB + _bound_side(up[:, :6]) + _bound_side(up[:, 6:])
    lo = FLB - _bound_side(down[:, :6]) - _bound_side(down[:, 6:])
    return _scale_from_bound(np.maximum(hi, -lo))


def output_scale(OW, MOW, FO, MFO, piece_scale, full_scale):
    """Largest common scale for the two 32 bit output heads.

    Both heads accumulate ``act[c] * weight_q[c]``, where ``act[c]`` is at most
    ``RELU_BOUND * scale[c]`` and ``weight_q[c]`` is the weight quantized with
    ``output_scale / scale[c]``. A term is therefore bounded by
    ``RELU_BOUND * |weight[c]| * output_scale`` whichever scale the dimension
    uses, so bounding the sum of ``|weight|`` over one output gives the largest
    scale which cannot overflow. On top of that every quantized weight has to
    fit into 16 bits. A single evaluation reads either a table or its mirror,
    never both, so the two are bounded separately.
    """
    piece_sum = 0.0
    for table in (OW, MOW):
        # At most one piece per square, and at most MAX_PIECES on the board.
        per_square = np.abs(table).sum(-1).max(0)                  # (square, outcome)
        largest = np.sort(per_square, axis=0)[::-1][:MAX_PIECES]
        piece_sum = max(piece_sum, largest.sum(0).max())
    full_sum = max(np.abs(FO).sum(-1).max(), np.abs(MFO).sum(-1).max())
    scale = MAX_INT32 / (RELU_BOUND * max(piece_sum, full_sum))

    piece_largest = np.maximum(np.abs(OW).reshape(-1, OW.shape[-1]).max(0),
                               np.abs(MOW).reshape(-1, MOW.shape[-1]).max(0))
    full_largest = np.maximum(np.abs(FO).max(0), np.abs(MFO).max(0))
    for largest, scales in ((piece_largest, piece_scale), (full_largest, full_scale)):
        usable = largest > 0
        scale = min(scale, (MAX_INT16 * scales[usable] / largest[usable]).min())
    # A power of two keeps the conversion back to float in the engine exact.
    return float(2.0 ** np.floor(np.log2(scale)))


def _quantize(values, scales, name):
    """Round to 16 bit, broadcasting ``scales`` over the last axis of ``values``."""
    rounded = np.rint(np.asarray(values, np.float64) * scales)
    largest = np.abs(rounded).max() if rounded.size else 0
    if largest > MAX_INT16:
        raise ValueError(f"{name} does not fit into 16 bits: max |value| is {largest:.0f}")
    return rounded.astype("<i2")


def pack(c1_weight, b1, out_weight, out_bias, f1_weight, f1_bias, fout_weight,
         d, fd, num_inputs, head=None):
    """Serialize a net to Winter's quantized format.

    The layout follows the float format, except that the bias of the piece
    accumulator arrives with the self relation already folded in, and the output
    bias moves into the header because it is the one value the engine still
    applies in float. Only the unmirrored half of every tensor is stored, the
    engine builds the mirror at load time.

    ``head`` adds NetRelHDP's pooled head and makes this a version 2 file (see
    ``pooled_head``): a dict of ``p1_weight``, ``p1_bias``, ``pout_weight`` and
    ``z_max``, the largest |lane| per (outcome, lane) seen in calibration, in
    logits. Without it the file is version 1, byte for byte as before.
    """
    W, B, OW, MOW, FL, FLB, FO, MFO = engine_tensors(
        c1_weight, b1, out_weight, f1_weight, f1_bias, fout_weight, d, fd)
    piece_scale = piece_scales(W, B)
    full_scale = full_scales(FL, FLB)
    scale = output_scale(OW, MOW, FO, MFO, piece_scale, full_scale)

    # Winter derives one scale per dimension and applies it to both halves of an
    # accumulator, which is what keeps its evaluation exactly symmetric under a
    # colour swap. The two halves are mirror images of the same weights, so the
    # bounds agree and only the first half is stored.
    if not np.array_equal(piece_scale[:d], piece_scale[d:]):
        raise ValueError("piece scales differ between the mirrored halves")
    if not np.array_equal(full_scale[:fd], full_scale[fd:]):
        raise ValueError("full scales differ between the mirrored halves")
    piece_scale, full_scale = piece_scale[:d], full_scale[:fd]

    # Per (piece, channel) for the conv tensors, per output channel for the
    # linear ones. The output heads absorb the reciprocal accumulator scale.
    conv_scale = np.tile(piece_scale, NUM_PIECE_TYPES)[:, None, None, None]
    self_relation = _as_array(c1_weight).reshape(
        NUM_PIECE_TYPES, d, NUM_PIECE_TYPES, GRID, GRID)[
            np.arange(NUM_PIECE_TYPES), :, np.arange(NUM_PIECE_TYPES), 7, 7]
    folded_bias = _as_array(b1).reshape(NUM_PIECE_TYPES, d, 8, 8) \
        + self_relation[:, :, None, None]

    payload = [
        _quantize(_as_array(c1_weight).reshape(NUM_PIECE_TYPES * d, NUM_PIECE_TYPES, GRID, GRID),
                  conv_scale, "c1.weight"),
        _quantize(folded_bias.reshape(NUM_PIECE_TYPES * d, 8, 8),
                  np.tile(piece_scale, NUM_PIECE_TYPES)[:, None, None], "b1"),
        _quantize(_as_array(out_weight).reshape(NUM_OUTPUTS, 2 * NUM_PIECE_TYPES * d, 8, 8),
                  (scale / np.tile(piece_scale, 2 * NUM_PIECE_TYPES))[None, :, None, None], "out.weight"),
        _quantize(_as_array(f1_weight).reshape(fd, num_inputs),
                  full_scale[:, None], "f1.weight"),
        _quantize(_as_array(f1_bias).reshape(fd), full_scale, "f1.bias"),
        _quantize(_as_array(fout_weight).reshape(NUM_OUTPUTS, 2 * fd),
                  (scale / np.tile(full_scale, 2))[None, :], "fout.weight"),
    ]

    version = FORMAT_VERSION
    if head is not None:
        version = POOLED_FORMAT_VERSION
        shifts = z_shifts(head["z_max"], scale)
        p1_q, p1_b, p1_exp = pooled_head(head["p1_weight"], head["p1_bias"], shifts, scale,
                                         full_scale, d)
        payload.append(p1_q)

    header = bytearray(MAGIC)
    header += np.array([version, 0, FLAG_QUANTIZED, d, fd, num_inputs,
                        NUM_PIECE_TYPES, GRID, NUM_OUTPUTS], "<u4").tobytes()
    header += np.array([scale, RELU_BOUND], "<f4").tobytes()
    header += piece_scale.astype("<i4").tobytes()
    header += full_scale.astype("<i4").tobytes()
    header += _as_array(out_bias).reshape(NUM_OUTPUTS).astype("<f4").tobytes()
    if head is not None:
        # Version 2 appends: pd, the lane shifts of the first half (the mirror half
        # uses the same), the hidden layer's row exponents and bias, and the float
        # output layer. The p1 weights follow the other tensors in the payload.
        header += np.array([p1_q.shape[0]], "<u4").tobytes()
        header += shifts.astype("<i4").tobytes()
        header += p1_exp.astype("<i4").tobytes()
        header += p1_b.astype("<i4").tobytes()
        header += _as_array(head["pout_weight"]).astype("<f4").tobytes()
    header += b"\0" * (-len(header) % 16)
    # header_bytes is the second word, and tells the engine where the payload starts.
    header[8:12] = np.array([len(header)], "<u4").tobytes()

    return bytes(header) + b"".join(t.tobytes() for t in payload)


def z_shifts(z_max, scale):
    """Right shifts that bring the pooled head's int32 lanes into 16 bits.

    ``z_max`` is (outcome, lane) in logits, and a lane holds ``z * scale`` in the
    engine. Lane ``j`` of one half and lane ``j`` of the other trade places when
    the engine evaluates the mirrored position, so both use the larger of their
    two ranges. Returns (outcome, lane of the first half).
    """
    z_max = _as_array(z_max)
    half = z_max.shape[1] // 2
    lane = CALIBRATION_MARGIN * np.maximum(z_max[:, :half], z_max[:, half:]) * scale
    return np.maximum(0, np.ceil(np.log2(np.maximum(lane, 1) / MAX_INT16))).astype(np.int64)


def pooled_head(p1_weight, p1_bias, shifts, scale, full_scale, d):
    """Quantize the pooled head's hidden layer for 16 bit inputs and 32 bit sums.

    Its inputs are the lanes, shifted by ``shifts`` and saturated to 16 bits, then
    the clipped global accumulator as the engine already holds it. The weights
    absorb both input scales, and each row gets the largest power of two scale
    under which the 32 bit sum provably cannot overflow for any inputs in range.

    Returns ``(weights, bias, exponents)``: row ``r`` evaluates to
    ``(weights[r] . inputs + bias[r]) * 2 ** -exponents[r]``.
    """
    lanes = np.concatenate([shifts, shifts], axis=1)
    column = np.concatenate([(scale / 2.0 ** lanes).ravel(), np.tile(full_scale, 2)])
    w = _as_array(p1_weight) / column
    b = _as_array(p1_bias)
    # A saturated lane can reach -32768.
    x_max = np.concatenate([np.full(NUM_OUTPUTS * d, MAX_INT16 + 1.0),
                            RELU_BOUND * np.tile(full_scale, 2)])
    # Rounding moves every weight and the bias by at most half a unit.
    room = MAX_INT32 - 0.5 * x_max.sum() - 0.5
    row = np.minimum(room / (np.abs(w) @ x_max + np.abs(b)), MAX_INT16 / np.abs(w).max(1))
    exponents = np.floor(np.log2(row)).astype(np.int64)
    weights = _quantize(w, (2.0 ** exponents)[:, None], "p1.weight")
    bias = np.rint(b * 2.0 ** exponents)
    worst = np.abs(weights.astype(np.float64)) @ x_max + np.abs(bias)
    if worst.max() > MAX_INT32:
        raise ValueError(f"p1 can reach {worst.max():.0f}, beyond 32 bits")
    return weights, bias.astype(np.int64), exponents


def _shape(state, key):
    if key not in state:
        raise ValueError(f"checkpoint has no '{key}'; is this a NetRelHD net?")
    return tuple(state[key].shape)


def unwrap_state(state):
    """The weights of a checkpoint that may wrap them in a dict."""
    for wrapper in ("model", "state_dict", "model_state_dict"):
        if "c1.weight" not in state and wrapper in state:
            state = state[wrapper]
    return state


def from_state_dict(state, positions=None):
    """Pack a NetRelHD or NetRelHDP checkpoint, taking the dimensions from the tensor
    shapes.

    The shapes also identify the architecture: NetRelHD feeds its output heads the
    net and its mirror concatenated, so ``out.weight`` and ``fout.weight`` are
    twice as wide on their input axis as they are in NetRelH. NetRelHDP adds the
    pooled head, whose input scales are calibrated on ``positions`` (a (N, 772)
    tensor), so for it they are required.
    """
    state = unwrap_state(state)
    pooled = "p1.weight" in state
    if pooled and positions is None:
        raise ValueError("a NetRelHDP net needs positions to calibrate its pooled head")
    c1 = _shape(state, "c1.weight")
    if len(c1) != 4 or c1[1] != NUM_PIECE_TYPES or c1[2:] != (GRID, GRID) \
            or c1[0] % NUM_PIECE_TYPES != 0:
        raise ValueError(f"c1.weight has shape {c1}, expected (12*d, 12, {GRID}, {GRID})")
    d = c1[0] // NUM_PIECE_TYPES
    fd, num_inputs = _shape(state, "f1.weight")

    expected = {
        "b1": (NUM_PIECE_TYPES * d, 8, 8),
        "out.weight": (NUM_OUTPUTS, 2 * NUM_PIECE_TYPES * d, 8, 8),
        "out.bias": (NUM_OUTPUTS,),
        "f1.bias": (fd,),
        "fout.weight": (NUM_OUTPUTS, 2 * fd),
    }
    for key, want in expected.items():
        got = _shape(state, key)
        if got != want:
            hint = ""
            if key in ("out.weight", "fout.weight") and got[1] * 2 == want[1]:
                hint = " -- this looks like NetRelH, which Winter does not load"
            raise ValueError(f"{key} has shape {got}, expected {want}{hint}")
    if num_inputs != NUM_PIECE_TYPES * 64:
        raise ValueError(f"f1 takes {num_inputs} inputs, but the mirrored path can only "
                         f"supply {NUM_PIECE_TYPES * 64}")

    head = None
    if pooled:
        pd = _shape(state, "p1.weight")[0]
        for key, want in {"p1.weight": (pd, NUM_OUTPUTS * d + 2 * fd), "p1.bias": (pd,),
                          "pout.weight": (NUM_OUTPUTS, pd)}.items():
            if _shape(state, key) != want:
                raise ValueError(f"{key} has shape {_shape(state, key)}, expected {want}")
        head = {"p1_weight": state["p1.weight"], "p1_bias": state["p1.bias"],
                "pout_weight": state["pout.weight"],
                "z_max": observed_lanes(float_model(state), positions)}

    blob = pack(state["c1.weight"], state["b1"], state["out.weight"], state["out.bias"],
                state["f1.weight"], state["f1.bias"], state["fout.weight"],
                d=d, fd=fd, num_inputs=num_inputs, head=head)
    return blob, d, fd, num_inputs


def infer_float_dims(num_values, num_inputs=NUM_PIECE_TYPES * 64):
    """Recover (d, fd) from the value count of a float .bin written by serialize()."""
    per_d = (NUM_PIECE_TYPES * NUM_PIECE_TYPES * GRID * GRID
             + NUM_PIECE_TYPES * 64 + NUM_OUTPUTS * 2 * NUM_PIECE_TYPES * 64)
    per_fd = num_inputs + 1 + NUM_OUTPUTS * 2
    found = [(d, (num_values - NUM_OUTPUTS - d * per_d) // per_fd)
             for d in range(8, 129, 8)
             if 0 < num_values - NUM_OUTPUTS - d * per_d
             and (num_values - NUM_OUTPUTS - d * per_d) % per_fd == 0]
    found = [(d, fd) for d, fd in found if fd in (16, 32, 64, 128, 256, 512)]
    if len(found) != 1:
        raise ValueError(f"cannot infer d and fd from {num_values} values "
                         f"(candidates: {found or 'none'}); pass --d and --fd")
    return found[0]


def from_float_file(path, d=None, fd=None, num_inputs=NUM_PIECE_TYPES * 64):
    """Pack a float .bin written by NetRelHD.serialize()."""
    raw = np.fromfile(path, dtype=np.float32)
    if d is None or fd is None:
        d, fd = infer_float_dims(raw.size, num_inputs)
    offset = 0

    def take(count):
        nonlocal offset
        chunk = raw[offset:offset + count]
        offset += count
        return chunk

    tensors = [take(NUM_PIECE_TYPES * d * NUM_PIECE_TYPES * GRID * GRID),
               take(NUM_PIECE_TYPES * d * 64),
               take(NUM_OUTPUTS * 2 * NUM_PIECE_TYPES * d * 64),
               take(NUM_OUTPUTS),
               take(fd * num_inputs), take(fd), take(NUM_OUTPUTS * 2 * fd)]
    if offset != raw.size:
        raise ValueError(f"{path} holds {raw.size} values, d={d} fd={fd} needs {offset}")
    return pack(*tensors, d=d, fd=fd, num_inputs=num_inputs), d, fd, num_inputs


# Everything below evaluates nets on positions and needs torch, which the packing
# above does not, so it is imported where it is used.

def float_model(state, activation_bound=RELU_BOUND):
    """The checkpoint as a float64 NetRelHD or NetRelHDP in eval mode."""
    import torch
    import torch.nn as nn
    import model

    state = unwrap_state(state)
    d = _shape(state, "c1.weight")[0] // NUM_PIECE_TYPES
    fd, num_inputs = _shape(state, "f1.weight")
    kwargs = dict(d=d, fd=fd, num_inputs=num_inputs,
                  activation=nn.Hardtanh(min_val=0, max_val=activation_bound))
    if "p1.weight" in state:
        net = model.NetRelHDP(pd=_shape(state, "p1.weight")[0], **kwargs)
    else:
        net = model.NetRelHD(**kwargs)
    net.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
    return net.double().eval()


def _batches(positions, batch_size=1024):
    import torch
    for i in range(0, positions.shape[0], batch_size):
        yield torch.as_tensor(positions[i:i + batch_size], dtype=torch.float64)


def _pooled_lanes(net, x):
    """The lanes of NetRelHDP.forward: per outcome, one per int32 lane of Winter's
    output accumulators. Also returns the global features."""
    import torch
    features, f = net.features(x)
    z = torch.einsum('bcs,kcs->bkc', features.flatten(2), net.out.weight.flatten(2))
    z = z.view(-1, NUM_OUTPUTS, 2, NUM_PIECE_TYPES, net.d).sum(3)
    return z.reshape(-1, NUM_OUTPUTS, net.d, 2).sum(-1), f


def observed_lanes(net, positions):
    """Largest |lane| per (outcome, lane) over ``positions``, in logits."""
    import torch
    largest = None
    with torch.no_grad():
        for x in _batches(positions):
            z = _pooled_lanes(net, x)[0].abs().amax(0)
            largest = z if largest is None else torch.maximum(largest, z)
    return largest.numpy()


def unpack(blob):
    """Read a quantized net back into its integer tensors and scales."""
    if blob[:4] != MAGIC:
        raise ValueError("bad magic, this is not a quantized net")
    version, header_bytes, flags, d, fd, num_inputs = (
        int(v) for v in np.frombuffer(blob, "<u4", 6, 4))
    scale, relu_bound = (float(v) for v in np.frombuffer(blob, "<f4", 2, 40))
    offset = 48

    def take(dtype, count, start):
        return np.frombuffer(blob, dtype, count, start), start + count * np.dtype(dtype).itemsize

    net = dict(version=version, d=d, fd=fd, num_inputs=num_inputs, scale=scale,
               relu_bound=relu_bound)
    net["piece_scale"], offset = take("<i4", d, offset)
    net["full_scale"], offset = take("<i4", fd, offset)
    net["out_bias"], offset = take("<f4", NUM_OUTPUTS, offset)
    if version == POOLED_FORMAT_VERSION:
        (pd,), offset = take("<u4", 1, offset)
        net["pd"] = pd = int(pd)
        shifts, offset = take("<i4", NUM_OUTPUTS * (d // 2), offset)
        net["shifts"] = shifts.reshape(NUM_OUTPUTS, d // 2)
        net["p1_exp"], offset = take("<i4", pd, offset)
        net["p1_bias"], offset = take("<i4", pd, offset)
        pout, offset = take("<f4", NUM_OUTPUTS * pd, offset)
        net["pout"] = pout.reshape(NUM_OUTPUTS, pd)
    elif version != FORMAT_VERSION:
        raise ValueError(f"unknown format version {version}")

    offset = header_bytes
    shapes = [("c1", (NUM_PIECE_TYPES * d, NUM_PIECE_TYPES, GRID, GRID)),
              ("b1", (NUM_PIECE_TYPES * d, 8, 8)),
              ("out", (NUM_OUTPUTS, 2 * NUM_PIECE_TYPES * d, 8, 8)),
              ("f1", (fd, num_inputs)), ("f1_bias", (fd,)), ("fout", (NUM_OUTPUTS, 2 * fd))]
    if version == POOLED_FORMAT_VERSION:
        shapes.append(("p1", (net["pd"], NUM_OUTPUTS * d + 2 * fd)))
    for name, shape in shapes:
        values, offset = take("<i2", int(np.prod(shape)), offset)
        net[name] = values.reshape(shape)
    if offset != len(blob):
        raise ValueError(f"{len(blob) - offset} unexpected bytes after the payload")
    return net


def simulate(net, positions):
    """Logits Winter computes for ``positions`` with the unpacked net ``net``.

    Every step before the pooled head is an integer sum or a clip, so running the
    float net on the dequantized weights reproduces it exactly, and none of those
    sums can overflow. The pooled head is evaluated as the engine does: lanes
    shifted and saturated, a 32 bit sum, then float. Returns (logits, fraction of
    saturated lane values).
    """
    import torch
    import torch.nn as nn
    import model

    d, fd = net["d"], net["fd"]
    ps = net["piece_scale"].astype(np.float64)
    fs = net["full_scale"].astype(np.float64)
    scale = net["scale"]
    t = lambda a: torch.as_tensor(np.asarray(a, np.float64))

    c1 = net["c1"] / np.tile(ps, NUM_PIECE_TYPES)[:, None, None, None]
    # The stored bias has the self relation folded in, which the conv adds again.
    types = np.arange(NUM_PIECE_TYPES)
    self_relation = c1.reshape(NUM_PIECE_TYPES, d, NUM_PIECE_TYPES, GRID, GRID)[
        types, :, types, 7, 7]
    b1 = net["b1"].reshape(NUM_PIECE_TYPES, d, 8, 8) / ps[None, :, None, None] \
        - self_relation[:, :, None, None]
    state = {
        "c1.weight": t(c1), "b1": t(b1.reshape(NUM_PIECE_TYPES * d, 8, 8)),
        "out.weight": t(net["out"] / (scale / np.tile(ps, 2 * NUM_PIECE_TYPES))[None, :, None, None]),
        "out.bias": t(net["out_bias"]),
        "f1.weight": t(net["f1"] / fs[:, None]), "f1.bias": t(net["f1_bias"] / fs),
        "fout.weight": t(net["fout"] / (scale / np.tile(fs, 2))[None, :]),
    }
    base = model.NetRelHD(d=d, fd=fd, num_inputs=net["num_inputs"],
                          activation=nn.Hardtanh(min_val=0, max_val=net["relu_bound"]))
    base.load_state_dict(state)
    base = base.double().eval()

    pooled = net["version"] == POOLED_FORMAT_VERSION
    if pooled:
        shifts = t(2.0 ** np.concatenate([net["shifts"], net["shifts"]], axis=1))
        p1, p1_bias = t(net["p1"]), t(net["p1_bias"])
        p1_unscale = t(2.0 ** -net["p1_exp"].astype(np.float64))
        pout, full = t(net["pout"]), t(np.tile(fs, 2))

    logits, saturated, lanes = [], 0, 0
    with torch.no_grad():
        for x in _batches(positions):
            z, f = _pooled_lanes(base, x)
            out = z.sum(-1) + base.out.bias + base.fout(f)
            if pooled:
                z16 = torch.floor(torch.round(z * scale) / shifts)
                saturated += int((z16.abs() > MAX_INT16).sum())
                lanes += z16.numel()
                z16 = z16.clamp(-MAX_INT16 - 1, MAX_INT16)
                inputs = torch.cat([z16.flatten(1), torch.round(f * full)], dim=1)
                hidden = ((inputs @ p1.T + p1_bias) * p1_unscale).clamp(0, net["relu_bound"])
                out = out + hidden @ pout.T
            logits.append(out)
    return torch.cat(logits), (saturated / lanes if lanes else 0.0)


def check(state, blob, positions):
    """Compare the quantized net in ``blob`` with the float checkpoint on ``positions``.

    Returns a dict of the logit and expected score (win + draw / 2) errors, and the
    fraction of pooled lanes that saturated.
    """
    import torch
    net = float_model(state)
    with torch.no_grad():
        reference = torch.cat([net(x, activate=False) for x in _batches(positions)])
    logits, saturated = simulate(unpack(blob), positions)

    def expected_score(v):
        p = torch.softmax(v, dim=-1)
        return p[:, 0] + 0.5 * p[:, 1]

    score_error = (expected_score(logits) - expected_score(reference)).abs()
    return {"positions": positions.shape[0],
            "max_logit_error": (logits - reference).abs().max().item(),
            "mean_score_error": score_error.mean().item(),
            "max_score_error": score_error.max().item(),
            "saturated_lanes": saturated}


def load_positions(name, data_dir, count, seed=0):
    """Up to ``count`` positions of a dataset, as a dense (N, 772) float array."""
    import loader
    features, _ = loader.load_features_results(name, data_dir=data_dir)
    if count and features.shape[0] > count:
        rows = np.sort(np.random.default_rng(seed).choice(features.shape[0], count, replace=False))
        features = features[rows]
    return features.toarray().astype(np.float64)


def main():
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Convert a trained net to the quantized format Winter loads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help="A .pt checkpoint, or a float .bin from serialize()")
    parser.add_argument('output', nargs='?', default=None,
                        help="Destination .qbin (default: the input with a .qbin suffix)")
    parser.add_argument('--d', type=int, default=None,
                        help="Relative-conv block width; inferred when not given")
    parser.add_argument('--fd', type=int, default=None,
                        help="Full hidden-layer width; inferred when not given")
    parser.add_argument('--force', action='store_true', help="Overwrite the output if it exists")
    parser.add_argument('--check', action='store_true',
                        help="Simulate the written net on --positions and report its error "
                             "against the float checkpoint (.pt input only)")
    parser.add_argument('--positions', type=str, default="validation_games",
                        help="Dataset that calibrates a NetRelHDP head and that --check uses")
    parser.add_argument('--data-dir', type=str, default="../datasets/",
                        help="Directory holding features_<positions>.npz")
    parser.add_argument('--max-positions', type=int, default=20000,
                        help="Random subset of --positions to use; 0 for all")
    args = parser.parse_args()

    out = args.output or os.path.splitext(args.input)[0] + ".qbin"
    if os.path.exists(out) and not args.force:
        sys.exit(f"Refusing to overwrite {out} (pass --force to replace)")

    if args.input.endswith(".pt"):
        import torch
        try:
            state = torch.load(args.input, map_location="cpu", weights_only=True)
        except TypeError:      # torch older than 1.13
            state = torch.load(args.input, map_location="cpu")
        state = unwrap_state(state)
        positions = None
        if "p1.weight" in state or args.check:
            positions = load_positions(args.positions, args.data_dir, args.max_positions)
            print(f"using {positions.shape[0]} positions from {args.positions}")
        blob, d, fd, num_inputs = from_state_dict(state, positions)
    else:
        if args.check:
            sys.exit("--check needs a .pt checkpoint to compare against")
        blob, d, fd, num_inputs = from_float_file(args.input, args.d, args.fd)

    with open(out, "wb") as f:
        f.write(blob)

    scales = np.frombuffer(blob[48:48 + 4 * d], "<i4")
    scale = np.frombuffer(blob[40:44], "<f4")[0]
    print(f"wrote {out}  ({len(blob):,} bytes, d={d} fd={fd} inputs={num_inputs})")
    print(f"  accumulator scales: min {scales.min()} median {int(np.median(scales))} "
          f"max {scales.max()}")
    print(f"  output scale: 2^{int(np.log2(scale))}")
    print(f"  set block_size = 2 * {d} and full_block_size = 2 * {fd} in Winter's net_types.h")
    net = unpack(blob)
    if net["version"] == POOLED_FORMAT_VERSION:
        print(f"  pooled head: pd={net['pd']}, lane shifts {net['shifts'].min()}..{net['shifts'].max()}, "
              f"row exponents {net['p1_exp'].min()}..{net['p1_exp'].max()} "
              f"(format version {POOLED_FORMAT_VERSION})")
    if args.check:
        report = check(state, blob, positions)
        print(f"  check on {report['positions']} positions: expected score error mean "
              f"{report['mean_score_error']:.2e} max {report['max_score_error']:.2e}, "
              f"logit error max {report['max_logit_error']:.4f}, "
              f"saturated lanes {report['saturated_lanes']:.1e}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
