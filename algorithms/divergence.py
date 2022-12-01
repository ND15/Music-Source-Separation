import numpy as np

EPS = 1e-12


def add_eps(input, target, eps=EPS):
    _input = input + eps
    _target = target + eps
    return _input, _target


def kl_divergence(input, target, eps=EPS):
    _input, _target = add_eps(input, target, eps)
    ratio = _target / _input
    loss = _target * np.log(ratio)
    loss = loss.sum(dim=0)

    return loss


def is_divergence(input, target, eps=EPS):
    _input, _target = add_eps(input, target, eps)

    ratio = _target / _input
    loss = ratio - np.log(ratio) - 1

    return loss


def generalized_kl_divergence(input, target, eps=EPS):
    _input, _target = add_eps(input, target, eps)

    ratio = _input / _target
    loss = _target * np.log(ratio) + _input - _target

    return loss


def beta_divergence(input, target, beta=2):
    beta_minus1 = beta - 1

    assert beta != 0, "Use is_divergence() instead"
    assert beta_minus1 != 0, "Use generalized_kl_divergence() instead"

    loss = target * (target ** beta_minus1 - input ** beta_minus1) / beta_minus1 - (
            target ** beta - input ** beta) / beta

    return loss


def multichannel_is_divergence(input, target, eps=EPS):
    in_shape, tar_shape = input.shape, target.shape

    assert in_shape[-2] == in_shape[-1] and tar_shape[-2] == tar_shape[-1], "Invalid Input shape"
    n_channels = in_shape[-1]

    input, target = input + eps * np.eye(n_channels), target + eps * np.eye(n_channels)
    XX = target @ np.linalg.inv(input)

    loss = np.trace(XX, axis=-1, axis2=-1).real - np.log(np.linalg.det(XX).real) - n_channels

    return loss


def log_det_divergence(input, target, eps=EPS):
    shape_input, shape_target = input.shape, target.shape
    assert shape_input[-2] == shape_input[-1] and shape_target[-2] == shape_target[-1], "Invalid input shape"
    n_channels = shape_input[-1]

    XY = target @ np.linalg.inv(input)

    trace = np.trace(XY, axis1=-2, axis2=-1).real
    eigvals_X, eigvals_Y = np.linalg.eigvalsh(target).real, np.linalg.eigvalsh(input).real
    eigvals_X[eigvals_X < eps], eigvals_Y[eigvals_Y < eps] = eps, eps

    logdet = np.sum(np.log(eigvals_X), axis=-1) - np.sum(np.log(eigvals_Y), axis=-1)

    loss = trace - logdet - n_channels

    return
