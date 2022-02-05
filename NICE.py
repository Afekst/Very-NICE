import torch
import torch.nn as nn


class AdditiveCoupling(nn.Module):
    """
    Additive coupling layer as described in the original paper
    """
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """
        C'tor to the additive coupling layer
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
            x2 += self.m(x1)
        else:
            x2 -= self.m(x1)

        if self.mask_config:
            x = torch.stack((x2, x1), dim=2)
        else:
            x = torch.stack((x1, x2), dim =2)

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
            nn.Sequential(
                nn.Linear(in_out_dim//2, mid_dim),
                nn.ReLU()
            )
        )
        for _ in range(hidden):
            func.append(
                nn.Sequential(
                    nn.Linear(mid_dim, mid_dim),
                    nn.ReLU()
                )
            )
        func.append(
            nn.Linear(mid_dim, in_out_dim//2)
        )
        return nn.Sequential(*func)
