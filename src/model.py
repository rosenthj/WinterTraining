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

    def forward(self, x_in, activate=True):
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
        x = self.out(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        fx = x_in[:, :self.f_dim]
        fx = self.activation(self.f1(fx))

        fxm = x_mirror.view(-1, 12*8*8)
        fxm = self.activation(self.f1(fxm))

        x = x + self.fout(torch.cat([fx, fxm], dim=1))
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
