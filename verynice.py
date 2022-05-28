import torch
import torch.nn as nn
from torch.distributions.transforms import Transform, SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np
from layers import VeryAdditiveCoupling, Scaling


class VeryNICE(nn.Module):
    """
    Complete VeryNICE model as described in paper
    """
    def __init__(self, prior, coupling, in_out_dim, partitions, hidden, device):
        """
        C'tor for VeryNICE model
        :param prior: 'logistic' or 'gaussian'
        :param coupling: number of coupling blocks
        :param in_out_dim: input/output dimension
        :param max_neurons: maximal number of hidden neurons
        :param hidden: number of hidden layers in each coupling block
        """
        super(VeryNICE, self).__init__()
        self.device = device
        self.prior = self._define_prior(prior)
        self.in_out_dim = in_out_dim
        self.net = self._create_network(coupling, in_out_dim, hidden, partitions)
        self.scaling = Scaling(in_out_dim)


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

    def _create_network(self, coupling, in_out_dim, hidden, partitions):
        """
        Create NN consist of AdditiveCoupling blocks
        :param coupling: number of coupling blocks
        :param in_out_dim: input/output dimension
        :param max_neurons: maximal number of hidden neurons
        :param hidden: number of hidden layers in each coupling block
        :return: list of coupling blocks
        """
        func = nn.ModuleList([
            VeryAdditiveCoupling(in_out_dim=in_out_dim,
                                 mid_dim=500,
                                 hidden=hidden,
                                 partitions=partitions,
                                 mask_config=(i+1) % 2,
                                 device=self.device)
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
