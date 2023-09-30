import torch
from torch import Tensor
from itertools import product
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.utils import parameters_to_vector
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np


def compute_feature_kernel(activation: Tensor, ) -> torch.Tensor:
    """
    Compute the feature kernel in the single testing batch, example: the first test batch.
    :param activation: activation of the last layer, shape: [batches, seq_size, embedding_dim]
    :return: matrix of feature kernel
    """
    activation = activation.view(activation.size(0), -1)
    kernel = torch.matmul(activation, activation.T)
    return kernel


def compute_feature_kernel_distance(kernel_history: Tensor):
    """
    The same as last statement but now consider the distance of features in two discrete time points
    :param kernel_history: the recordings of kernels, shape :[n_recordings, batches, batches], n_recordings
                        can be any time resolution, e.g., x epochs, x time steps.
    :return: matrix of kernel distance
    """
    distance = torch.zeros(kernel_history.size[0], kernel_history.size[0])
    for i, j in product(range(kernel_history.size[0]), range(kernel_history.size[0])):
        # Numerator and Denominator
        numerator = torch.matmul(kernel_history[i], kernel_history[j].T).trace()
        denominator = torch.sqrt(torch.matmul(kernel_history[i], kernel_history[i].T).trace()) \
                      * torch.sqrt(torch.matmul(kernel_history[j], kernel_history[j].T).trace())
        distance[i, j] = 1 - numerator / denominator
    return distance


def compute_feature_kernel_velocity(kernel_before, kernel_last):
    """
    compute the kernel velocity between two consecutive recordings.
    :param kernel_before:
    :param kernel_last:
    :return:
    """
    numerator = torch.matmul(kernel_before, kernel_last.T).trace()
    denominator = torch.sqrt(torch.matmul(kernel_before, kernel_before.T).trace()) \
                  * torch.sqrt(torch.matmul(kernel_last, kernel_last.T).trace())
    velocity = (1 - numerator / denominator) / 1  # because the same recording interval, so adopt 1 as 1 unit.
    return velocity


# The following is used to compute the metric: edge of stability
class SquaredLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return 0.5 * ((input - target) ** 2).sum()


class SquaredAccuracy(nn.Module):
    def __init__(self):
        super(SquaredAccuracy, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target.argmax(1)).float().sum()


class AccuracyCE(nn.Module):
    def __init__(self):
        super(AccuracyCE, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target).float().sum()


def _iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X.cuda(), batch_y.cuda()


def _get_loss_and_acc(loss: str):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()
    raise NotImplementedError(f"no such loss function: {loss}")


def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = 32):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    for (X, y) in _iterate_dataset(dataset, physical_batch_size):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    return hvp


def _lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
        torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                            neigs=6, physical_batch_size=1000):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = _lanczos(hvp_delta, nparams, neigs=neigs)
    return evals
