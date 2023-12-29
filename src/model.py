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
        self.f1 = nn.Linear(768, fd)
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
        fx = x_in[:, :768]
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
