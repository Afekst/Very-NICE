import torch
import torch.nn as nn
from torch.distributions.transforms import Transform, SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np


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
                nn.Linear(in_out_dim//2, mid_dim)
            )
        )
        for _ in range(hidden):
            func.append(
                nn.Sequential(
                    nn.Linear(mid_dim, mid_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(mid_dim)
                )
            )
        func.append(
            nn.Linear(mid_dim, in_out_dim//2)
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
        eps = 1e-5
        log_det_J = log_det_J - self.scale.sum() + 2 * self.scale.sum() * self.training
        if self.training:
            x = x * torch.exp(self.scale)
        else:
            x = x * torch.exp(-self.scale)

        return x, log_det_J


class NICE(nn.Module):
    """
    Complete NICE model as described in paper
    """
    def __init__(self, prior, coupling, in_out_dim, mid_dim, hidden, device):
        """
        C'tor for NICE model
        :param prior: 'logistic' or 'gaussian'
        :param coupling: number of coupling blocks
        :param in_out_dim: input/output dimension
        :param mid_dim: number of units in each hidden layer of the coupling block
        :param hidden: number of hidden layers in each coupling block
        """
        super(NICE, self).__init__()
        self.prior = self._define_prior(prior)
        self.in_out_dim = in_out_dim
        self.net = self._create_network(coupling, in_out_dim, mid_dim, hidden)
        self.scaling = Scaling(in_out_dim)
        self.device = device

    @staticmethod
    def _define_prior(prior):
        """
        Define the prior distribution to be used
        :param prior: 'logistic' or 'gaussian'
        :return: distribution function
        """
        if prior == 'gaussian':
            p = torch.distributions.Normal(
                torch.tensor(0.),
                torch.tensor(1.)
            )
        elif prior == 'logistic':
            p = TransformedDistribution(
                Uniform(0, 1),
                [SigmoidTransform().inv,
                 AffineTransform(loc=0., scale=1.)]
            )
        else:
            raise ValueError('Prior not implemented')
        return p

    @staticmethod
    def _create_network(coupling, in_out_dim, mid_dim, hidden):
        """
        Create NN consist of AdditiveCoupling blocks
        :param coupling: number of coupling blocks
        :param in_out_dim: input/output dimension
        :param mid_dim: number of units in each hidden layer of the coupling block
        :param hidden: number of hidden layers in each coupling block
        :return: list of coupling blocks
        """
        func = nn.ModuleList([
            AdditiveCoupling(in_out_dim=in_out_dim,
                             mid_dim=mid_dim,
                             hidden=hidden,
                             mask_config=(i+1) % 2)
            for i in range(coupling)
        ])
        return func

    def f_inverse(self, z):
        """
        Transformation g: Z -> X (inverse of f)
        :param z: tensor in latent space Z
        :return: transformed tensor in data space X
        """
        x, _ = self.scaling(z, 0)
        for cpl in reversed(self.net):
            x, _ = cpl(x, 0)
        return x

    def f(self, x):
        """
        Transformation f: X -> Z (inverse of g)
        :param x: tensor in data space X
        :return: transformed tensor in latent space Z and log determinant Jacobian
        """
        log_det_J = 0
        for cpl in self.net:
            x, log_det_J = cpl(x, log_det_J)
        z, log_det_J = self.scaling(x, log_det_J)
        return z, log_det_J

    def log_prob(self, x):
        """
        Computes data log-likelihood
        :param x: input minibatch
        :return: log-likelihood of input
        """
        z, log_det_J = self.f(x)
        log_det_J = log_det_J - torch.log(torch.tensor(256)) * self.in_out_dim
        log_ll = torch.sum(self.prior.log_prob(z.cpu()), dim=1).to(self.device)
        return log_ll + log_det_J

    def sample(self, size):
        """
        Generates samples
        :param size: number of samples to generate
        :return: samples from the data space X
        """
        z = self.prior.sample((size, self.in_out_dim))
        z = z.to(self.device)
        x = self.f_inverse(z)
        return x

    def forward(self, x):
        """
        Forward pass
        :param x: input minibatch
        :return: log-likelihood of input
        """
        return self.log_prob(x)
