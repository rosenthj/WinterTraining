import torch.nn as nn
import torch.nn.functional as F


def _tensor_to_bytes(x):
    return x.detach().flatten().numpy().tobytes()


class Net(nn.Module):
    def __init__(self, d=16, d2=None, activation=F.relu):
        super().__init__()

        self.activation = activation
        self.l1 = nn.Linear(772, d)
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

    def _s(self, buffer, layer, name, bias=True, verbose=0):
        if layer is None:
            return
        if verbose >= 1:
            print(f"Buffering {name}")
        buffer.extend(_tensor_to_bytes(layer.weight))
        if bias:
            if verbose >= 2:
                print(f"{name} has bias")
            buffer.extend(_tensor_to_bytes(layer.bias))

    def serialize(self, filename, verbose=0):
        buffer = bytearray()
        self._s(buffer, self.l1, "l1", verbose=verbose)
        self._s(buffer, self.l2, "l2", verbose=verbose)
        self._s(buffer, self.out, "out", verbose=verbose)
        with open(filename, "wb") as f:
            f.write(buffer)
