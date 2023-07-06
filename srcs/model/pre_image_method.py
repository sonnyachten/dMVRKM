import torch
import logging

logger = logging.getLogger(__name__)


class kernel_smoother:
    def __init__(self, nearest_neighbours: int = 10) -> None:
        self.nearest_neighbours = nearest_neighbours

    def __call__(
        self, x: torch.Tensor = None, kernel_vector: torch.Tensor = None
    ) -> torch.Tensor:

        sorted_kernel_vector, indices = torch.sort(kernel_vector, descending=True)

        normalized_weights = sorted_kernel_vector[
            : self.nearest_neighbours
        ] / torch.sum(sorted_kernel_vector[: self.nearest_neighbours])

        op = torch.sum(
            torch.diag(normalized_weights) @ x[indices[: self.nearest_neighbours]],
            dim=0,
        )
        return op


class ridge_regression:
    def __init__(
        self,
        ridge_regression_alpha: float = 1.0,
        ridge_regression_rbf_sigma: float = 1.0,
    ) -> None:
        self.ridge_regression_alpha = ridge_regression_alpha
        self.ridge_regression_rbf_sigma = ridge_regression_rbf_sigma
        self.pre_image_has_been_fit = False

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if not self.pre_image_has_been_fit:
            alpha = self.ridge_regression_alpha

            K2 = self.compute_kernel(
                X,
                kernel="rbf",
                sigma=self.ridge_regression_rbf_sigma,
            )
            # K2 = self.center(K2)
            K = K2 + alpha * torch.eye(K2.shape[0])

            # self.Kh = K
            self.ridge_regression_dual_coef_ = torch.linalg.solve(K, self.Y)
            self.pre_image_has_been_fit = True

        # map the next new point back to input space
        K = self.compute_kernel(
            self.H[-1, :].unsqueeze(0),
            x=X,
            kernel="rbf",
            sigma=self.ridge_regression_rbf_sigma,
        )
        # K = self.center(matrix=self.Kh, kernel_vector=K)

        if K.ndim <= 1:
            K = K.unsqueeze(0)
        K_reg = K  # + self.kwargs['ridge_regression_alpha'] * torch.eye(K.shape[0], K.shape[1])
        x_preimage = K_reg @ self.ridge_regression_dual_coef_
        if x_preimage.ndim <= 1:
            x_preimage = x_preimage.unsqueeze(0)

        return x_preimage
