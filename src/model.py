import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def tensor_to_bytes(x):
    return x.detach().to('cpu').flatten().numpy().tobytes()


class Net(nn.Module):
    def __init__(self, d=16, d2=None, num_inputs=772, activation=F.relu):
        super().__init__()

        self.activation = activation
        self.l1 = nn.Linear(num_inputs, d)
        if d2:
            self.l2 = nn.Linear(d, d2)
        else:
            self.l2 = None
            d2 = d
        self.out = nn.Linear(d2, 3)

    def forward(self, x_in, activate=True):
        x = self.l1(x_in)
        x = self.activation(x)
        if self.l2:
            x = self.activation(self.l2(x))
        x = self.out(x)
        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        buffer = bytearray()
        self._s(buffer, self.l1, "l1", verbose=verbose)
        self._s(buffer, self.l2, "l2", verbose=verbose)
        self._s(buffer, self.out, "out", verbose=verbose)
        with open(filename, "wb") as f:
            f.write(buffer)


class NetRel(nn.Module):
    def __init__(self, d=8, num_inputs=772, activation=F.relu):
        super().__init__()
        self.d = d
        self.activation = activation
        self.c1 = nn.Conv2d(12, 12 * d, 15, padding=7, bias=False)
        self.b1 = nn.parameter.Parameter(data=torch.zeros((12 * d, 8, 8)))
        # conv, no bias, probably 15x15
        # linear for non-board visible, with bias
        # filter
        # out, 3 8x8 conv filters
        self.out = nn.Conv2d(12 * d, 3, 8, padding=0)

    def partial_load_blocks(self):
        # Conv output channels are grouped per piece-plane (d each), so growing d must be
        # seeded block-wise: 12 plane-blocks on c1/b1 (dim 0) and on out's input (dim 1).
        return {"c1.weight": (0, 12), "b1": (0, 12), "out.weight": (1, 12)}

    def forward(self, x_in, activate=True):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        x = x * mask
        x = self.activation(x)
        x = self.out(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def f(self, x_in):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        # x = self.b1
        return x * mask

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        # print(f"Skipping serialize call. Not yet implemented!")
        # return
        buffer = bytearray()
        self._s(buffer, self.c1, "conv layer", bias=False, verbose=verbose)
        # self._s(buffer, self.b1, "bias layer", bias=False, verbose=verbose)
        buffer.extend(tensor_to_bytes(self.b1.data))
        self._s(buffer, self.out, "out", verbose=verbose)
        with open(filename, "wb") as f:
            f.write(buffer)


class NetRelX(nn.Module):
    def __init__(self, d=8, num_inputs=772, activation=F.relu):
        super().__init__()
        self.d = d
        self.activation = activation
        self.c1 = nn.Conv2d(12, 12 * d, 15, padding=7, bias=False)
        self.b1 = nn.parameter.Parameter(data=torch.zeros((12 * d, 8, 8)))
        self.c2 = nn.Conv2d(12 * d, 12 * d, 15, groups=d, padding=7, bias=False)
        self.b2 = nn.parameter.Parameter(data=torch.zeros((12 * d, 8, 8)))
        # conv, no bias, probably 15x15
        # linear for non-board visible, with bias
        # filter
        # out, 3 8x8 conv filters
        self.out = nn.Conv2d(12 * d, 3, 8, padding=0)

    def forward(self, x_in, activate=True):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        x = x * mask
        x = self.activation(x)
        x = x.view(-1, 12, self.d, 8, 8).transpose(1, 2).reshape(-1, 12 * self.d, 8, 8)
        x = self.c2(x)
        x = x.view(-1, self.d, 12, 8, 8).transpose(1, 2).reshape(-1, 12 * self.d, 8, 8)
        x = x + self.b2
        x = x * mask
        x = self.activation(x)
        x = self.out(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def f(self, x_in):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        # x = self.b1
        return x * mask

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        # print(f"Skipping serialize call. Not yet implemented!")
        # return
        buffer = bytearray()
        self._s(buffer, self.c1, "conv layer", bias=False, verbose=verbose)
        # self._s(buffer, self.b1, "bias layer", bias=False, verbose=verbose)
        buffer.extend(tensor_to_bytes(self.b1.data))
        self._s(buffer, self.out, "out", verbose=verbose)
        with open(filename, "wb") as f:
            f.write(buffer)


class NetRelH(nn.Module):
    def __init__(self, d=8, fd=64, num_inputs=772, activation=F.relu):
        super().__init__()
        self.d = d
        self.activation = activation
        self.c1 = nn.Conv2d(12, 12 * d, 15, padding=7, bias=False)
        self.b1 = nn.parameter.Parameter(data=torch.zeros((12 * d, 8, 8)))
        self.out = nn.Conv2d(12 * d, 3, 8, padding=0)
        self.f_dim = num_inputs
        self.f1 = nn.Linear(self.f_dim, fd)
        self.fout = nn.Linear(fd, 3, bias=False)

    def partial_load_blocks(self):
        # Conv output channels grouped per piece-plane (d each): 12 plane-blocks on c1/b1
        # (dim 0) and on out's input (dim 1). out/fout are not mirror-doubled here, and f1's
        # fd units are not plane-grouped, so the full-layer path grows as a plain prefix.
        return {"c1.weight": (0, 12), "b1": (0, 12), "out.weight": (1, 12)}

    def forward(self, x_in, activate=True):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        x = x * mask
        x = self.activation(x)
        x = self.out(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        fx = x_in[:, :self.f_dim]
        fx = self.activation(self.f1(fx))
        x = x + self.fout(fx)
        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def f(self, x_in):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        # x = self.b1
        return x * mask

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        # print(f"Skipping serialize call. Not yet implemented!")
        # return
        buffer = bytearray()
        self._s(buffer, self.c1, "conv layer", bias=False, verbose=verbose)
        buffer.extend(tensor_to_bytes(self.b1.data))
        self._s(buffer, self.out, "out", verbose=verbose)
        self._s(buffer, self.f1, "f1 layer", verbose=verbose)
        self._s(buffer, self.fout, "f out", bias=False, verbose=verbose)
        with open(filename, "wb") as f:
            f.write(buffer)


class NetRelA(nn.Module):
    def __init__(self, d=8, num_inputs=772, activation=F.relu):
        super().__init__()
        self.d = d
        self.activation = activation
        self.c1 = nn.Conv2d(12, 12 * d, 15, padding=7, bias=False)
        self.b1 = nn.parameter.Parameter(data=torch.zeros((12 * d, 8, 8)))
        self.a1 = nn.Conv2d(12 * d, 1, 1)
        # conv, no bias, probably 15x15
        # linear for non-board visible, with bias
        # filter
        # out, 3 8x8 conv filters
        self.out = nn.Conv2d(12 * d, 3, 8, padding=0)

    def forward(self, x_in, activate=True):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        x = x * mask
        x = self.activation(x)
        a = self.a1(x)
        a = F.softmax(a.view(-1, 1, 8*8), dim=-1).view(-1, 1, 8, 8)
        x = a * x
        x = self.out(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def f(self, x_in):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        # x = self.b1
        return x * mask

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        # print(f"Skipping serialize call. Not yet implemented!")
        # return
        buffer = bytearray()
        self._s(buffer, self.c1, "conv layer", bias=False, verbose=verbose)
        # self._s(buffer, self.b1, "bias layer", bias=False, verbose=verbose)
        buffer.extend(tensor_to_bytes(self.b1.data))
        self._s(buffer, self.out, "out", verbose=verbose)
        self._s(buffer, self.a1, "att layer", bias=False, verbose=verbose)
        with open(filename, "wb") as f:
            f.write(buffer)


class NetRelR(nn.Module):
    def __init__(self, d=8, fd=64, rd=8, num_inputs=772, activation=F.relu):
        super().__init__()
        self.d = d
        self.activation = activation
        self.c1 = nn.Conv2d(12, 12 * d, 15, padding=7, bias=False)
        self.b1 = nn.parameter.Parameter(data=torch.zeros((12 * d, 8, 8)))
        self.out = nn.Conv2d(12 * d, 3, 8, padding=0)
        self.f_dim = num_inputs
        self.f1 = nn.Linear(self.f_dim, fd)
        self.fout = nn.Linear(fd, 3, bias=True)
        self.r1 = nn.Linear(768, rd)
        self.rout = nn.Linear(rd, 2, bias=True)

    def forward(self, x_in, activate=True):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)

        x = self.c1(x) + self.b1
        x = x * mask
        x = self.activation(x)
        x = self.out(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        fx = x_in[:, :self.f_dim]
        fx = self.activation(self.f1(fx))
        fx = self.fout(fx)

        rx = x_in[:, :768]
        rx = self.activation(self.r1(rx))
        rx = F.softmax(self.rout(rx), dim=-1)

        x = rx[:, 0:1] * x + rx[:, 1:] * fx
        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def f(self, x_in):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        # x = self.b1
        return x * mask

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        print(f"Skipping serialize call. Not yet implemented!")
        return
        buffer = bytearray()
        self._s(buffer, self.c1, "conv layer", bias=False, verbose=verbose)
        buffer.extend(tensor_to_bytes(self.b1.data))
        self._s(buffer, self.out, "out", verbose=verbose)
        self._s(buffer, self.f1, "f1 layer", verbose=verbose)
        self._s(buffer, self.fout, "f out", bias=False, verbose=verbose)
        with open(filename, "wb") as f:
            f.write(buffer)


class NetRelHC(nn.Module):
    def __init__(self, d=8, fd=64, cd=4, num_inputs=768, activation=F.relu):
        super().__init__()
        self.d = d
        self.activation = activation
        self.c1 = nn.Conv2d(12, 12 * d, 15, padding=7, bias=False)
        self.b1 = nn.parameter.Parameter(data=torch.zeros((12 * d, 8, 8)))
        self.out = nn.Conv2d(12 * d, 3, 8, padding=0)

        self.f_dim = num_inputs
        self.f1 = nn.Linear(self.f_dim, fd)
        self.fout = nn.Linear(fd, 3, bias=False)

        self.c2 = nn.Conv2d(12, cd, 15, padding=7, bias=False)
        self.b2 = nn.parameter.Parameter(data=torch.zeros((cd, 8, 8)))
        self.cout = nn.Conv2d(cd, 3, 8, padding=0, bias=False)

    def forward(self, x_in, activate=True):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        x = x * mask
        x = self.activation(x)
        x = self.out(x)

        cx = x_in[:, :768].view(-1, 12, 8, 8)
        cx = self.c2(cx) + self.b2
        cx = self.activation(cx)
        x = x + self.cout(cx)

        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        fx = x_in[:, :self.f_dim]
        fx = self.activation(self.f1(fx))
        x = x + self.fout(fx)

        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def f(self, x_in):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        # x = self.b1
        return x * mask

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        # print(f"Skipping serialize call. Not yet implemented!")
        # return
        buffer = bytearray()
        self._s(buffer, self.c1, "conv layer", bias=False, verbose=verbose)
        buffer.extend(tensor_to_bytes(self.b1.data))
        self._s(buffer, self.out, "out", verbose=verbose)
        self._s(buffer, self.f1, "f1 layer", verbose=verbose)
        self._s(buffer, self.fout, "f out", bias=False, verbose=verbose)
        with open(filename, "wb") as f:
            f.write(buffer)


class ConvNet(nn.Module):
    def __init__(self, d=8, kernel_size=15, padding=7, activation=F.relu):
        super().__init__()
        self.d = d
        self.activation = activation
        hidden_width = 8 - (kernel_size - 1) + (2 * padding)
        self.conv = nn.Conv2d(12, d, kernel_size=kernel_size, padding=padding, bias=False)
        self.bias = nn.parameter.Parameter(data=torch.zeros((d, hidden_width, hidden_width)))
        self.out = nn.Conv2d(d, 3, hidden_width, padding=0)

    def forward(self, x_in, activate=True):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        x = self.conv(x) + self.bias
        x = self.activation(x)
        x = self.out(x)

        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        print(f"Skipping serialize call. Not yet implemented!")
        return


class NetRelHD(nn.Module):
    def __init__(self, d=8, fd=64, num_inputs=772, activation=F.relu):
        super().__init__()
        self.d = d
        self.activation = activation
        self.c1 = nn.Conv2d(12, 12 * d, 15, padding=7, bias=False)
        self.b1 = nn.parameter.Parameter(data=torch.zeros((12 * d, 8, 8)))
        self.out = nn.Conv2d(2 * 12 * d, 3, 8, padding=0)
        self.f_dim = num_inputs
        self.f1 = nn.Linear(self.f_dim, fd)
        self.fout = nn.Linear(2 * fd, 3, bias=False)

    def partial_load_blocks(self):
        # Growing d/fd reindexes several concatenated axes, which must be seeded block-wise
        # rather than as one contiguous prefix (see load_partial_state_dict):
        #  - the conv output channels are grouped per input piece-plane, d channels each
        #    (mask = repeat_interleave(x, d)), so c1.weight / b1 are 12 blocks along dim 0;
        #  - out takes cat([real | mirror]) and each half is those same 12 plane-blocks,
        #    so out.weight is 12*2 = 24 blocks along its input axis (dim 1);
        #  - fout takes cat([fx | fxm]); f1's fd units are not plane-grouped, so it is just
        #    2 blocks (the two mirror halves) along dim 1.
        return {
            "c1.weight": (0, 12),
            "b1": (0, 12),
            "out.weight": (1, 24),
            "fout.weight": (1, 2),
        }

    def features(self, x_in):
        """Activated piece features (B, 2*12*d, 8, 8) and full layer features (B, 2*fd),
        each as [real | mirror]."""
        x = x_in[:, :768].view(-1, 12, 8, 8)
        x_mirror = torch.zeros_like(x)
        for i in range(8):
            x_mirror[:, :, i, :] = x[:, :, 7-i, :]
        x_mirror = torch.roll(x_mirror, 6, dims=1)

        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        x = x * mask

        xm = x_mirror.clone()
        mask_mirrored = torch.repeat_interleave(xm, self.d, dim=1)
        xm = self.c1(xm) + self.b1
        xm = xm * mask_mirrored

        x = self.activation(torch.cat([x, xm], dim=1))

        fx = x_in[:, :self.f_dim]
        fx = self.activation(self.f1(fx))

        fxm = x_mirror.view(-1, 12*8*8)
        fxm = self.activation(self.f1(fxm))

        return x, torch.cat([fx, fxm], dim=1)

    def forward(self, x_in, activate=True):
        x, f = self.features(x_in)
        x = self.out(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        x = x + self.fout(f)
        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        # print(f"Skipping serialize call. Not yet implemented!")
        # return
        buffer = bytearray()
        self._s(buffer, self.c1, "conv layer", bias=False, verbose=verbose)
        buffer.extend(tensor_to_bytes(self.b1.data))
        self._s(buffer, self.out, "out", verbose=verbose)
        self._s(buffer, self.f1, "f1 layer", verbose=verbose)
        self._s(buffer, self.fout, "f out", bias=False, verbose=verbose)
        with open(filename, "wb") as f:
            f.write(buffer)

    def serialize_quantized(self, filename, verbose=0):
        """Write the net in the 16 bit format Winter loads (see quantize.py).

        Half the size of serialize(), and it carries the per dimension scales,
        so the engine does not derive them at startup. Winter reads this format
        only; serialize() is kept for anything that wants the raw float weights.
        """
        import quantize
        blob = quantize.pack(
            self.c1.weight, self.b1.data, self.out.weight, self.out.bias,
            self.f1.weight, self.f1.bias, self.fout.weight,
            d=self.d, fd=self.f1.out_features, num_inputs=self.f_dim)
        if verbose >= 1:
            print(f"Buffering quantized net ({len(blob)} bytes)")
        with open(filename, "wb") as f:
            f.write(blob)


class NetRelHDP(NetRelHD):
    """NetRelHD with a nonlinear head on the pooled piece features.

    NetRelHD's piece head is linear, so each logit is a sum of per piece contributions
    and pieces only interact through their own clipped relus. Here the head is split
    before that final sum: z[k, j] pools the contributions of feature lane j to outcome k
    over all pieces, and a small hidden layer reads [z, full layer features]:

        logits = sum_j z[:, :, j] + out.bias + fout(f)    (exactly NetRelHD)
               + pout(act(p1([z, f])))                    (the pooled head)

    pout starts at zero, so the net initially computes the same function as NetRelHD,
    including when seeded from a NetRelHD checkpoint via --init-from.

    The lanes match Winter's output accumulators (output_helpers): a piece's 2*d
    features are stored as [real | mirror] and madd sums adjacent channel pairs into
    one int32 lane, so z has d lanes per outcome. With black to move Winter uses the
    mirrored output weights, which swaps the halves of z and of f, so an engine
    version needs a mirrored p1 as well, like fout.

    Winter cannot evaluate this head yet, so there is no quantized export.
    """
    # save() skips the quantized export when this is None.
    serialize_quantized = None

    def __init__(self, d=8, fd=64, pd=32, num_inputs=772, activation=F.relu):
        super().__init__(d=d, fd=fd, num_inputs=num_inputs, activation=activation)
        assert d % 2 == 0, "d must be even to pair channels the way madd does"
        self.p1 = nn.Linear(3 * d + 2 * fd, pd)
        # Its own instance so test() reports the head separately from the conv and fc layers.
        self.head_activation = copy.deepcopy(activation)
        self.pout = nn.Linear(pd, 3, bias=False)
        nn.init.zeros_(self.pout.weight)

    def forward(self, x_in, activate=True):
        x, f = self.features(x_in)
        # Per channel contributions of the out conv, summed over squares.
        z = torch.einsum('bcs,kcs->bkc', x.flatten(2), self.out.weight.flatten(2))
        # Pool over the 12 piece planes, then pair adjacent channels like madd.
        z = z.view(-1, 3, 2, 12, self.d).sum(3)
        z = z.reshape(-1, 3, self.d, 2).sum(-1)

        x = z.sum(-1) + self.out.bias + self.fout(f)
        h = self.head_activation(self.p1(torch.cat([z.flatten(1), f], dim=1)))
        x = x + self.pout(h)
        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def serialize(self, filename, verbose=0):
        print(f"Skipping serialize call. Not yet implemented!")
        return


class CRNet(nn.Module):
    def __init__(self, d=8, rec=3, kernel_size=15, padding=7, activation=F.relu):
        super().__init__()
        self.d = d
        self.rec = rec
        self.activation = activation
        hidden_width = 8 - (kernel_size - 1) + (2 * padding)
        self.conv = nn.Conv2d(12, d, kernel_size=kernel_size, padding=padding, bias=False)
        self.hidden_conv = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.out = nn.Conv2d(d // 2, 3, hidden_width, padding=0)

    def forward(self, x_in, activate=True, rec=None):
        if rec is None:
            rec=self.rec
        x = x_in[:, :768].view(-1, 12, 8, 8)
        x = self.activation(self.conv(x))
        for i in range(rec):
            x = self.activation(self.hidden_conv(x))
        x = self.out(x[:, ::2])

        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        print(f"Skipping serialize call. Not yet implemented!")
        return


class CRNetv2(nn.Module):
    def __init__(self, d=8, rec=3, kernel_size=15, padding=7, activation=F.relu):
        super().__init__()
        self.d = d
        self.rec = rec
        self.activation = activation
        hidden_width = 8 - (kernel_size - 1) + (2 * padding)
        self.conv = nn.Conv2d(12, d, kernel_size=kernel_size, padding=padding, bias=True)
        self.hc1 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.hc2 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.hc3 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.ho = nn.Conv2d(d, d // 2, kernel_size=1)
        self.out = nn.Conv2d(d // 2, 3, hidden_width)

    def forward(self, x_in, activate=True, rec=None):
        if rec is None:
            rec=self.rec
        x = x_in[:, :768].view(-1, 12, 8, 8)
        x = self.activation(self.conv(x))
        for i in range(rec):
            x = self.activation(self.hc1(x))
            x = self.activation(self.hc2(x))
            x = self.activation(self.hc3(x))
        x = self.activation(self.ho(x))
        x = self.out(x)

        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        print(f"Skipping serialize call. Not yet implemented!")
        return


class CRNetv3(nn.Module):
    def __init__(self, d=8, rec=3, activation=F.relu):
        super().__init__()
        self.d = d
        self.rec = rec
        self.activation = activation
        self.emb = nn.Conv2d(12, d, 1)
        self.hc1 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.hc2 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.hc3 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.hc4 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.hc5 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.hc6 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.hc7 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.ho = nn.Conv2d(d, d // 2, kernel_size=1)
        self.out = nn.Conv2d(d // 2, 3, 8)

    def forward(self, x_in, activate=True, rec=None):
        if rec is None:
            rec=self.rec
        x = x_in[:, :768].view(-1, 12, 8, 8)
        x = self.activation(self.emb(x))
        for i in range(rec):
            x = self.activation(self.hc1(x))
            x = self.activation(self.hc2(x))
            x = self.activation(self.hc3(x))
            x = self.activation(self.hc4(x))
            x = self.activation(self.hc5(x))
            x = self.activation(self.hc6(x))
            x = self.activation(self.hc7(x))
        x = self.activation(self.ho(x))
        x = self.out(x)

        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def serialize(self, filename, verbose=0):
        print(f"Skipping serialize call. Not yet implemented!")
        return


class NetRelHX(nn.Module):
    def __init__(self, d=8, fd=64, num_inputs=772, activation=F.relu):
        super().__init__()
        self.d = d
        self.activation = activation
        self.c1 = nn.Conv2d(12, 12 * d, 15, padding=7, bias=False)
        self.b1 = nn.parameter.Parameter(data=torch.zeros((12 * d, 8, 8)))
        self.c2 = nn.Conv2d(12 * d, 12 * d, 15, groups=d, padding=7, bias=False)
        self.b2 = nn.parameter.Parameter(data=torch.zeros((12 * d, 8, 8)))
        # conv, no bias, probably 15x15
        # linear for non-board visible, with bias
        # filter
        self.f_dim = num_inputs
        self.f1 = nn.Linear(self.f_dim, fd)
        self.fout = nn.Linear(fd, 3, bias=False)
        # out, 3 8x8 conv filters
        self.out = nn.Conv2d(12 * d, 3, 8, padding=0)

    def forward(self, x_in, activate=True):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        x = x * mask
        x = self.activation(x)
        x = x.view(-1, 12, self.d, 8, 8).transpose(1, 2).reshape(-1, 12 * self.d, 8, 8)
        x = self.c2(x)
        x = x.view(-1, self.d, 12, 8, 8).transpose(1, 2).reshape(-1, 12 * self.d, 8, 8)
        x = x + self.b2
        x = x * mask
        x = self.activation(x)
        x = self.out(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        fx = x_in[:, :self.f_dim]
        fx = self.activation(self.f1(fx))
        x = x + self.fout(fx)
        if not activate:
            return x
        return F.softmax(x, dim=-1)

    def f(self, x_in):
        x = x_in[:, :768].view(-1, 12, 8, 8)
        mask = torch.repeat_interleave(x, self.d, dim=1)
        x = self.c1(x) + self.b1
        # x = self.b1
        return x * mask

    def _s(self, buffer, l, name, bias=True, verbose=0):
        if l is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(tensor_to_bytes(l.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(tensor_to_bytes(l.bias))

    def serialize(self, filename, verbose=0):
        print(f"Skipping serialize call. Not yet implemented!")
        return
        # buffer = bytearray()
        # self._s(buffer, self.c1, "conv layer", bias=False, verbose=verbose)
        # # self._s(buffer, self.b1, "bias layer", bias=False, verbose=verbose)
        # buffer.extend(tensor_to_bytes(self.b1.data))
        # self._s(buffer, self.out, "out", verbose=verbose)
        # with open(filename, "wb") as f:
        #     f.write(buffer)
