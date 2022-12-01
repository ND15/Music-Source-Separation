import numpy as np

from algorithms.divergence import generalized_kl_divergence, is_divergence, multichannel_is_divergence

# from algorithms.linalg import solve_Riccati

__metrics__ = ['EUC', 'KL', 'IS']
EPS = 1e-12


class NMFBase:
    def __init__(self, n_basis=2, eps=EPS):
        self.n_basis = n_basis
        self.loss = []
        self.eps = eps

    def __call__(self, target, iteration=100, **kwargs):
        self.target = target
        self._reset(**kwargs)
        self.update()

        W, H = self.basis, self.activation

        return W.copy(), H.copy()

    def _reset(self, **kwargs):
        assert self.target is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        n_basis = self.n_basis
        n_bins, n_frames = self.target.shape

        self.basis = np.random.rand(n_bins, n_basis)
        self.activation = np.random.rand(n_basis, n_frames)

    def update(self, iterations=100):
        target = self.target

        for idx in range(iterations):
            self.update_once()

            WH = self.basis @ self.activation
            loss = self.criterion(WH, target)
            self.loss.append(loss.sum())

    def update_once(self):
        raise NotImplementedError("Implement update_once() in your class")


class ComplexNMFBase:
    def __init__(self, n_basis=2,
                 regularizer=0.1,
                 eps=EPS):
        self.n_beats = n_basis
        self.regularizer = regularizer
        self.loss = []

        self.eps = eps

    def __call__(self, target, iteration=100, **kwargs):
        self.target = target
        self._reset(**kwargs)
        print("iteration received: ", iteration)
        self.update(iteration=iteration)

        W, H = self.basis, self.activation
        Ph = self.phase

        return W.copy(), H.copy(), Ph.copy()

    def _reset(self, **kwargs):
        assert self.target is not None, "Specify data"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        n_basis = self.n_basis
        n_bins, n_frames = self.target_shape

        self.basis = np.random.rand(n_bins, n_basis)
        self.activation = np.random.rand(n_basis, n_frames)
        self.phase = 2 * np.pi * np.random.rand(n_bins, n_basis, n_frames)

    def init_phase(self):
        n_basis = self.n_basis
        target = self.target

        phase = np.angle(target)
        self.phase = np.tile(phase[:, np.newaxis, :], reps=(1, n_basis, 1))

    def update(self, iteration=100):
        target = self.target
        print("iteration received: ", iteration)
        for idx in range(iteration):
            self.update_once()
            WHPh = np.sum(self.basis[:, :, np.newaxis] * self.activation[:, np.newaxis, :] * self.phase, axis=1)
            loss = self.criterion(WHPh, target)
            self.loss.append(loss.sum())


class EucNMF(NMFBase):
    def __init__(self,
                 n_basis=2,
                 domain=2,
                 algorithm='mm',
                 eps=EPS):
        super(EucNMF, self).__init__(n_basis=n_basis,
                                     eps=eps)
        assert 1 <= domain <= 2, "1 <= domain <= 2 is not satisfied"
        assert algorithm == 'mm', "Algorithm must be mm"

        self.domain = domain
        self.algorithm = algorithm
        self.criterion = lambda input, target: (target - input) ** 2

    def update(self, iterations=1000):
        domain = self.domain
        target = self.target

        for idx in range(iterations):
            self.update_once()

            WH = (self.basis @ self.activation) ** (2 / domain)
            loss = self.criterion(WH, target)
            if idx % 5 == 0:
                print("*" * 20, "Iteration: ", idx, " ", np.mean(loss), "*" * 20)
            self.loss.append(loss.sum())

    def update_once(self):
        if self.algorithm == 'mm':
            self.update_once_mm()
        else:
            raise ValueError("{} algorithm is not supported".format(self.algorithm))

    def update_once_mm(self):
        target = self.target
        domain = self.domain
        eps = self.eps

        W, H = self.basis, self.activation

        # basis updation
        H_transpose = H.transpose(1, 0)

        WH = W @ H
        WH[WH < eps] = eps

        WHH = (WH ** ((4 - domain) / domain)) @ H_transpose
        WHH[WHH < eps] = eps

        numerator = (target * (WH ** ((2 - domain) / domain))) @ H_transpose
        W = W * (numerator / WHH) ** (domain / (4 - domain))

        # activations updation
        W_transpose = W.transpose(1, 0)

        WH = W @ H
        WH[WH < eps] = eps

        WWH = W_transpose @ (WH ** ((4 - domain) / domain))
        WWH[WWH < eps] = eps

        numerator = W_transpose @ (target * (WH ** ((2 - domain) / domain)))
        H = H * (numerator / WWH) ** (domain / (4 - domain))

        self.basis, self.activation = W, H


class KLNMF(NMFBase):
    def __init__(self,
                 n_basis=2,
                 domain=2,
                 algorithm='mm',
                 eps=EPS):
        super(KLNMF, self).__init__(n_basis=n_basis, eps=EPS)

        assert 1 <= domain <= 2, "1 <= 'domain' <= 2 is not satisfied."
        assert algorithm == 'mm', "Algorithm must be mm"

        self.domain = domain
        self.algorithm = algorithm
        self.criterion = generalized_kl_divergence

    def update(self, iteration=100):
        domain = self.domain
        target = self.target

        for idx in range(iteration):
            self.update_once()

            WH = (self.basis @ self.activation) ** (2 / domain)
            loss = self.criterion(WH, target)

            self.loss.append(loss.sum())

    def update_once(self):
        if self.algorithm == 'mm':
            self.update_once_mm()
        else:
            raise ValueError("No support for {} algorithm".format(self.algorithm))

    def update_once_mm(self):
        target = self.target
        domain = self.domain
        eps = self.eps

        W, H = self.basis, self.activation

        # update basis
        H_transpose = H.transpose(1, 0)
        WH = W @ H
        WH[WH < eps] = eps
        WHH = (WH ** ((2 - domain) / domain)) @ H_transpose
        WHH[WHH < eps] = eps
        division = target / WH
        W = W * (division @ H_transpose / WHH) ** (domain / 2)

        # update activations
        W_transpose = W.transpose(1, 0)
        WH = W @ H
        WH[WH < eps] = eps
        WWH = W_transpose @ (WH ** ((2 - domain) / domain))
        WWH[WWH < eps] = eps
        division = target / WH
        H = H * (W_transpose @ division / WWH) ** (domain / 2)

        self.basis, self.activation = W, H


class ISNMF(NMFBase):
    def __init__(self, n_basis=2, domain=2, algorithm='mm', eps=EPS):
        """
        Args:
            K: number of basis
            algorithm: 'mm': MM algorithm based update
        """
        super().__init__(n_basis=n_basis, eps=eps)

        assert 1 <= domain <= 2, "1 <= `domain` <= 2 is not satisfied."

        self.domain = domain
        self.algorithm = algorithm
        self.criterion = is_divergence

    def update(self, iteration=100):
        domain = self.domain
        target = self.target

        for idx in range(iteration):
            self.update_once()

            TV = (self.basis @ self.activation) ** (2 / domain)
            loss = self.criterion(TV, target)
            self.loss.append(loss.sum())

    def update_once(self):
        if self.algorithm == 'mm':
            self.update_once_mm()
        elif self.algorithm == 'me':
            self.update_once_me()
        else:
            raise ValueError("Not support {} based update.".format(self.algorithm))

    def update_once_mm(self):
        target = self.target
        domain = self.domain
        eps = self.eps

        T, V = self.basis, self.activation

        # Update basis
        V_transpose = V.transpose(1, 0)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / (TV ** ((domain + 2) / domain)), 1 / TV
        TVV = TV_inverse @ V_transpose
        TVV[TVV < eps] = eps
        T = T * (division @ V_transpose / TVV) ** (domain / (domain + 2))

        # Update activations
        T_transpose = T.transpose(1, 0)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / (TV ** ((domain + 2) / domain)), 1 / TV
        TTV = T_transpose @ TV_inverse
        TTV[TTV < eps] = eps
        V = V * (T_transpose @ division / TTV) ** (domain / (domain + 2))

        self.basis, self.activation = T, V

    def update_once_me(self):
        target = self.target
        domain = self.domain
        eps = self.eps

        assert domain == 2, "Only domain = 2 is supported."

        T, V = self.basis, self.activation

        # Update basis
        V_transpose = V.transpose(1, 0)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / (TV ** ((domain + 2) / domain)), 1 / TV
        TVV = TV_inverse @ V_transpose
        TVV[TVV < eps] = eps
        T = T * (division @ V_transpose / TVV)

        # Update activations
        T_transpose = T.transpose(1, 0)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / (TV ** ((domain + 2) / domain)), 1 / TV
        TTV = T_transpose @ TV_inverse
        TTV[TTV < eps] = eps
        V = V * (T_transpose @ division / TTV)

        self.basis, self.activation = T, V
