import torch
import torch.nn as nn
import numpy as np


class VeryAdditiveCoupling(nn.Module):
    """
    Very Nice coupling layer
    """
    def __init__(self, in_out_dim, mid_dim, hidden, partitions, mask_config, device):
        """
        C'tor for a Very Nice coupling layer
        :param in_out_dim: input/output dimensions
        :param mid_dim: number of units in a hidden layer
        :param hidden: number of hidden layers
        :param partitions: number of partitions
        """
        super(VeryAdditiveCoupling, self).__init__()
        if in_out_dim % partitions:
            raise ValueError('input output dimension must be divisible by the number of partitions')

        self.device = device
        self.mask_config = mask_config
        self.ts = self._create_networks(in_out_dim, mid_dim, hidden, partitions)
        self.partitions = partitions
        self.D = self._degeneration_matrix(partitions)
        M = torch.randn_like(self.D, dtype=torch.float32)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, x, log_det_J):
        [B, D] = list(x.size())
        x_split = x.reshape((B, D//self.partitions, self.partitions))
        x_split = torch.roll(x_split, self.mask_config, -1)

        txs = []
        for i in range(self.partitions):
            txs.append(self.ts[i](x_split[:, :, i]))
        txs = torch.stack(txs, dim=1)
        xs = x_split.transpose(1, 2)

        if self.training:
            M_tag = self.M * self.D
        else:
            I = torch.eye(self.partitions, device=self.M.device)
            M_tag = torch.inverse(self.M * self.D + I) - I
        out = xs + M_tag @ txs

        out = torch.transpose(out, 1, 2)
        out = torch.roll(out, -self.mask_config, -1)
        out = out.reshape((B, D))

        return out, log_det_J

    @staticmethod
    def _create_networks(in_out_dim, mid_dim, hidden, partitions):
        nets = nn.ModuleList([])
        for _ in range(partitions):
            func = nn.ModuleList([])
            func.append(
                nn.Sequential(
                    nn.Linear(in_out_dim // partitions, mid_dim),
                    nn.ReLU()
                )
            )
            for _ in range(hidden):
                func.append(
                    nn.Sequential(
                        nn.Linear(mid_dim, mid_dim),
                        nn.ReLU(),
                    )
                )
            func.append(
                nn.Linear(mid_dim, in_out_dim // partitions)
            )
            nets.append(nn.Sequential(*func))
        return nets

    def _degeneration_matrix(self, partitions):
        D = np.tril(np.ones((partitions, partitions), dtype='float32'))
        np.fill_diagonal(D, 0)
        return torch.from_numpy(D).to(self.device)


class AdditiveCoupling(nn.Module):
    """
    Additive coupling layer as described in the original paper
    """
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """
        C'tor for additive coupling layer
        :param in_out_dim: input/output dimensions
        :param mid_dim: number of units in a hidden layer
        :param hidden: number of hidden layers
        :param mask_config: 1 if transform odd units, 0 if transform even units
        """
        super(AdditiveCoupling, self).__init__()
        self.mask_config = mask_config
        self.m = self._create_network(in_out_dim, mid_dim, hidden)

    def forward(self, x, log_det_J):
        """
        Forward pass
        :param x: input tensor
        :param log_det_J: log determinant of the Jacobian
        :return: transformed tensor and updated log-determinant of Jacobian
        """
        [B, D] = list(x.size())
        x = x.reshape((B, D//2, 2))

        if self.mask_config:
            x1, x2 = x[:, :, 1], x[:, :, 0]
        else:
            x1, x2 = x[:, :, 0], x[:, :, 1]

        if self.training:
            x2 = x2 + self.m(x1)
        else:
            x2 = x2 - self.m(x1)

        if self.mask_config:
            x = torch.stack((x2, x1), dim=2)
        else:
            x = torch.stack((x1, x2), dim=2)

        x = x.reshape((B, D))
        return x, log_det_J

    @staticmethod
    def _create_network(in_out_dim, mid_dim, hidden):
        """
        Create a fully-connected layer according to specified parameters
        :param in_out_dim: input/output dimensions
        :param mid_dim: number of units in a hidden layer
        :param hidden: number of hidden layers
        :return: Sequential model
        """
        func = nn.ModuleList([])
        func.append(
            nn.Linear(in_out_dim//2, mid_dim)
        )
        for _ in range(hidden):
            func.append(
                nn.Sequential(
                    nn.Linear(mid_dim, mid_dim),
                    nn.ReLU(),
                )
            )
        func.append(
            nn.Linear(mid_dim, in_out_dim // 2),
        )
        return nn.Sequential(*func)


class Scaling(nn.Module):
    """
    Scaling layer as described in paper
    """
    def __init__(self, dim):
        """
        C'tor for scaling layer
        :param dim: in/out dimension
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)),
            requires_grad=True
        )

    def forward(self, x, log_det_J):
        """
        Forward pass
        :param x: input tensor
        :param log_det_J: log determinant of the Jacobian
        :return: transformed tensor and updated log-determinant of Jacobian
        """
        log_det_J = log_det_J - self.scale.sum() + 2 * self.scale.sum() * self.training
        if self.training:
            x = x * torch.exp(self.scale)
        else:
            x = x * torch.exp(-self.scale)

        return x, log_det_J
