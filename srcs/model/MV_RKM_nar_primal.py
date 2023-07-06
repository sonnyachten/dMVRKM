import logging
from typing import Tuple
import hydra

import scipy.linalg as sl
import torch
from sklearn.kernel_approximation import RBFSampler

from srcs.model.pre_image_method import kernel_smoother
from srcs.utils.util import (Nystroem_RenyiSampling, conditional_trange,
                             convert_to_AR)

logger = logging.getLogger(__name__)


class Multiview_RKM_NAR_primal:
    r"""
    This implements C U = U \Lambda, where U = [Ux^T Uy^T]^T.
    """

    def __init__(self, **kwargs: dict):
        self.kwargs = kwargs
        self.s = self.kwargs["s"]  # Dimension of the latent space.
        self.lag = self.kwargs["lag"]  # time lag
        if "pre_image_method" in self.kwargs.keys():
            self.pre_image_method = self.kwargs["pre_image_method"]

        # Collect kernel parameters in separate dictionaries.
        self.kwargs_x, self.kwargs_y = {}, {}
        for k, v in self.kwargs.items():
            if k.endswith("_x"):
                self.kwargs_x[k.strip("_x")] = v
            elif k.endswith("_y"):
                self.kwargs_y[k.strip("_y")] = v
            else:
                self.kwargs_x[k] = v
                self.kwargs_y[k] = v

    def train(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Train the model using Recurrent RKM NAR Primal model.
        """
        assert X is not None, "Provide input data matrix"

        if X.ndim == 1:
            X = X.unsqueeze(1)  # Shape the data matrix as n x d
        if Y.ndim == 1:
            Y = Y.unsqueeze(1)  # Shape the data matrix as n x d
        self.d = X.shape[1]
        self.n = X.shape[0]

        self.X = X
        self.Y = Y

        # Get features ----------------------------------
        self.phi_X = feature_map(**self.kwargs_x)  # init class object
        self.phi_X_data = self.phi_X.features(data=self.X)  # centered data

        self.phi_Y = feature_map(**self.kwargs_y)  # init class object
        self.phi_Y_data = self.phi_Y.features(data=self.Y)  # centered data

        # Get Covariance matrices -------------
        C = self.Covariance_matrix(X=self.phi_X_data, Y=self.phi_Y_data)
        assert (
                self.s <= C.shape[0]
        ), "Number of components cannot be more than shape of matrix"

        # Matrix factorization -----------------
        lambdas, self.U = self.matrix_factorization(matrix=C)

        # resacle U to also satisfy stationarity conditions w.r.t. phi
        self.lambdas = torch.diag(lambdas)
        self.sqrt_lambdas = torch.diag(torch.sqrt(lambdas))
        self.U = self.U @ self.sqrt_lambdas
        self.Ux = self.U[: self.phi_X_data.shape[-1]]
        self.Uy = self.U[self.phi_X_data.shape[-1]:]

        # Pre-compute for later use --------------------
        self.coeff_h_phi_x = (
                torch.linalg.inv(self.lambdas - (self.Uy.T @ self.Uy)) @ self.Ux.T
        )

        # Get embeddings H
        self.H = torch.linalg.inv(self.lambdas) @ (
                self.Ux.T @ self.phi_X_data.T + self.Uy.T @ self.phi_Y_data.T
        )
        self.H = self.H.T

        # Save model
        torch.save({"rkm_state_dict": self.__dict__,
                    "kwargs": self.__dict__['kwargs'],
                    },
                   f"{hydra.utils.os.getcwd()}/model_{self.__str__().split('_')[-1].split(' ')[0]}.pt")
        logger.info(f"\nSaved model: {hydra.utils.os.getcwd()}/model_{self.__str__().split('_')[-1].split(' ')[0]}.pt")

    def matrix_factorization(
            self, matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements matrix factorization using SVD or Eigen decomposition
        and returns the eigenvalues and eigenvectors.
        """
        if self.kwargs["decomposition_method"] == "svd":
            U, S, _ = torch.linalg.svd(matrix, full_matrices=True)
            eigvals, eigvecs = S[: self.s], U[:, : self.s]

        elif self.kwargs["decomposition_method"] == "eigen":
            S, U = sl.eigh(matrix)  # EigVals are in ascending order
            eigvals = torch.flip(torch.tensor(S[-self.s:]), dims=[0])
            eigvecs = torch.flip(torch.tensor(U[:, -self.s:]), dims=[1])
        else:
            raise ValueError(
                "Invalid Decomposition method. Valid options are: svd, eigen."
            )

        # Check if decomposition is successful
        error = torch.norm(
            matrix - torch.tensor(U) @ torch.diag(torch.tensor(S)) @ torch.tensor(U).T
        )
        assert error < 1e-5, f"Decomposition error ({error}) more than tolerance!"
        return eigvals, eigvecs

    def predict(
            self,
            init_vec: torch.tensor = None,
            n_steps: int = 1,
            verbose=True,
    ):
        """
        Given X, predicts Y
        """
        if init_vec is None:
            init_vec = convert_to_AR(
                data=self.Y[-(self.lag + 1):],
                lag=self.lag,
                n_steps_ahead=self.kwargs["n_steps_ahead"],
            )
        if init_vec.ndim == 1:
            init_vec = init_vec.unsqueeze(0)

        self.H_pred = self.H.clone()
        self.Y_pred = self.Y.clone()

        for i in conditional_trange(verbose, n_steps, desc="Predicting"):
            # Get input
            if i == 0:
                x_star = init_vec
            else:
                x_star = convert_to_AR(
                    data=self.Y_pred[-(self.lag + 1):],
                    lag=self.lag,
                    n_steps_ahead=self.kwargs["n_steps_ahead"],
                )

            phi_x_star = self.phi_X.features(data=x_star)  # centered data

            # Predict next latent variable
            h_n_ahead = self.coeff_h_phi_x @ phi_x_star.T
            self.H_pred = torch.cat((self.H_pred, h_n_ahead.T), dim=0)

            # Predict next point in feature space
            phi_y_tilde = (self.Uy @ h_n_ahead).T + self.phi_Y.mean

            # Get next point in input space (pre-image)
            pre_image = self.pre_image(data=phi_y_tilde, verbose=verbose)

            self.Y_pred = torch.cat((self.Y_pred, pre_image), dim=0)
        return self.Y_pred[-n_steps:]

    def pre_image(self, data: torch.tensor, verbose=True):
        """
        Implements pre-image method
        """

        if self.kwargs_y["kernel"] == "rbf":
            if self.kwargs["pre_image_method"] == "kernel_smoother":
                similarities = self.phi_Y_data @ (data - self.phi_Y.mean).T
                ks = kernel_smoother(
                    nearest_neighbours=self.kwargs["nearest_neighbours"]
                )
                op = ks(x=self.Y[: self.n], kernel_vector=similarities)
            elif self.kwargs["pre_image_method"] == "ridge_regression":
                raise ValueError("ridge_regression not yet implemented.")
            else:
                raise ValueError(
                    f"pre_image_method {self.kwargs['pre_image_method']} not valid."
                )
        elif self.kwargs_y["kernel"] == "linear":
            op = data.T
        else:
            raise f"kernel_y: {self.kwargs_y['kernel']} not supported."

        if op.ndim == 1:  # Single dimensional data
            op = op.unsqueeze(-1)
        return op

    @staticmethod
    def Covariance_matrix(
            X: torch.Tensor = None, Y: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the centered covariance matrix in 2 view setting.
        """
        C_xx = X.T @ X
        C_xy = X.T @ Y
        C_yy = Y.T @ Y

        C = torch.cat(
            [torch.cat([C_xx, C_xy], dim=1), torch.cat([C_xy.T, C_yy], dim=1)], dim=0
        )
        return C


class feature_map:
    """
    This class implements the feature map for the kernel methods.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.kernel = kwargs["kernel"]
        self.mean = None

    def features(self, data: torch.Tensor = None) -> torch.Tensor:
        """
        Return the centered feature map.
        """

        if self.kernel == "rbf":
            gamma = 0.5 * 1 / self.kwargs["sigma"] ** 2
            if self.kwargs["approximator"] == "rff":
                self.kernel_x_sampler = RBFSampler(
                    n_components=self.kwargs["DIM_FEATURES"],
                    gamma=gamma,
                    random_state=42,
                )  # gamma=5e-4
                features = torch.tensor(self.kernel_x_sampler.fit_transform(data))

            elif self.kwargs["approximator"] == "nystroem":
                self.kernel_x_sampler = Nystroem_RenyiSampling(
                    n_components=self.kwargs["DIM_FEATURES"], gamma=gamma
                )  # gamma=5e-4
                features = torch.tensor(self.kernel_x_sampler.fit_transform(data))
            else:
                raise f"approximator: {self.kwargs['approximator']} is not supported."

        elif self.kernel == "linear":
            features = data

        else:
            raise f"kernel: {self.kernel} is not supported."

        # Center the features
        if self.kwargs["center_K"]:
            if self.mean is None:
                self.mean = features.mean(dim=0).view(1, -1)
            features = features - self.mean
        return features
