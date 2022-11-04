import torch
import torch.nn.functional as F
from survae.distributions import ConditionalDistribution

from .cond_surjection import ConditionalSurjection


class ConditionalDiscreteArgmaxSurjection(ConditionalSurjection):
    '''
    A generative argmax surjection using one-hot encoding. Argmax is performed over the final dimension.

    Note: This is a discrete version of the ConditionalBinaryProductArgmaxSurjection.

    Args:
        encoder: ConditionalDistribution, a distribution q(z|x) with support over z s.t. x=argmax z.

    Example:
        Input tensor x of shape (B, D, L) with discrete values {0,1,...,C-1}:
        encoder should be a distribution of shape (B, D, L, C).
    '''
    stochastic_forward = True

    def __init__(self, encoder, num_classes):
        super(ConditionalDiscreteArgmaxSurjection, self).__init__()
        assert isinstance(encoder, ConditionalDistribution)
        self.encoder = encoder
        self.num_classes = num_classes

    def forward(self, x, context):
        # Note: x is a discrete tensor, while context can be either discrete or continuous.

        # Transform
        z, log_qz = self.encoder.sample_with_log_prob(context_act=x, context_obs=context)
        ldj = -log_qz
        return z, ldj

    def inverse(self, z, context):
        # inverse transform does not require context as z is conditional on context already.
        idx = torch.argmax(z, dim=-1)
        return idx

    def inverse_soft(self, z, context):
        # inverse transform does not require context as z is conditional on context already.
        z_soft = F.softmax(z, dim=-1)
        return z_soft
