import logging

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn

from srcs.model.nn import *
from srcs.utils import stiefel_optimizer
from srcs.utils.util import instantiate

logger = logging.getLogger(__name__)


class RKM_Stiefel(nn.Module):
    """Defines the Stiefel RKM model and its loss functions"""

    def __init__(self, args):
        super(RKM_Stiefel, self).__init__()
        self.args = args
        self.lag = self.args["lag"]
        self.s = self.args["s"]
        self.coeff_h_from_xy = None
        self.coeff_h_from_y = None
        self.coeff_h_from_x = None

        # Collect feature-map parameters in separate dictionaries.
        self.kwargs_x, self.kwargs_y = {}, {}
        for k, v in self.args.items():
            if k.endswith("_x"):
                self.kwargs_x[k.strip("_x")] = v
            elif k.endswith("_y"):
                self.kwargs_y[k.strip("_y")] = v
            else:
                self.kwargs_x[k] = v
                self.kwargs_y[k] = v

        self.encx = instantiate(self.args["encoderx"][0], args=self.kwargs_x)
        self.decx = instantiate(self.args["decoderx"][0], args=self.kwargs_x)
        self.ency = instantiate(self.args["encodery"][0], args=self.kwargs_y)
        self.decy = instantiate(self.args["decodery"][0], args=self.kwargs_y)

        if 'identity' in self.encx.__str__():
            self.args["DIM_FEATURES_x"] = self.args['dx']
        if 'identity' in self.ency.__str__():
            self.args["DIM_FEATURES_y"] = self.args['dy']

        # Init Stiefel Manifold_params
        if self.args["mode"] == "primal":
            if self.args["optimise_U"] == "split":
                self.Ux_tilde = nn.Parameter(
                    (0.5 ** -0.5)
                    * nn.init.orthogonal_(
                        torch.Tensor(self.args["s"], self.args["DIM_FEATURES_x"])
                    ),
                    requires_grad=True,
                )
                self.Uy_tilde = nn.Parameter(
                    (0.5 ** -0.5)
                    * nn.init.orthogonal_(
                        torch.Tensor(self.args["s"], self.args["DIM_FEATURES_y"])
                    ),
                    requires_grad=True,
                )
            elif self.args["optimise_U"] == "joint":
                self.U_tilde = nn.Parameter(
                    nn.init.orthogonal_(
                        torch.DoubleTensor(
                            self.args["s"],
                            self.args["DIM_FEATURES_x"] + self.args["DIM_FEATURES_y"],
                        )
                    ),
                    requires_grad=True,
                )
                self.Ux_tilde = self.U_tilde[:, : self.args["DIM_FEATURES_x"]]
                self.Uy_tilde = self.U_tilde[:, self.args["DIM_FEATURES_x"]:]
            else:
                raise ValueError("Valid arguments are `split` or `joint`")
        elif self.args["mode"] == "dual":
            self.H = nn.Parameter(
                nn.init.orthogonal_(
                    torch.DoubleTensor(
                        self.args["s"],
                        self.args["n"],
                    )
                ),
                requires_grad=True,
            )


    def forward(self, x, y):
        if self.args["optimise_U"] == "split":
            self.U_tilde = torch.cat((self.Ux_tilde, self.Uy_tilde), dim=1).to(
                self.args["device"]
            )

        # Encode
        phi_x = self.get_phi_x(x)
        phi_y = self.get_phi_y(y)

        if self.args["mode"] == "primal":
            C = self.covariance_matrix(phi_x=phi_x, phi_y=phi_y)
            self.eig_vals = self.U_tilde @ C @ self.U_tilde.T

            kpca_loss = self.primal_loss(C)  # KPCA Loss
            # Rescale U
            self.U = self.rescale_U(U_tilde=self.U_tilde, eig_vals=self.eig_vals)
            self.Ux = self.U[:, : self.args["DIM_FEATURES_x"]]
            self.Uy = self.U[:, self.args["DIM_FEATURES_x"]:]

        elif self.args["mode"] == "dual":
            K = self.kernel_matrix(phi_x=phi_x, phi_y=phi_y)
            self.eig_vals = self.H @ K @ self.H.T
            kpca_loss = self.dual_loss(K=K)

        assert kpca_loss > -1e-4, (
            "KPCA loss cannot be negative, something is wrong."
            f"\n Current value: {kpca_loss}"
            f"\n lambdas: {self.eig_vals}"
        )
        # If not trainable, then return 0
        if (len(self.decx._parameters) > 0) and (len(self.decy._parameters) > 0):
            recons_x = torch.tensor(0)
            recons_y = torch.tensor(0)
        else:
            if self.args["mode"] == "primal":
                H = self.get_h(phi_x=phi_x, phi_y=phi_y)  # latent space
                if len(self.decx._parameters) > 0:
                    phi_x_tilde = self.get_phi_x_tilde(H)
                    x_tilde = self.decx(phi_x_tilde + self.phi_x_mean)
                    recons_x = self.recon_loss(x_tilde, x)  # Recons_loss
                else:
                    recons_x = torch.tensor(0)

                if len(self.decy._parameters) > 0:
                    phi_y_tilde = self.get_phi_y_tilde(H)
                    y_tilde = self.decy(phi_y_tilde + self.phi_y_mean)
                    recons_y = self.recon_loss(y_tilde, y)  # Recons_loss
                else:
                    recons_y = torch.tensor(0)
            elif self.args["mode"] == "dual":
                if len(self.decx._parameters) > 0:
                    phi_x_tilde = self.get_phi_x_tilde(self.H)
                    x_tilde = self.decx(phi_x_tilde + self.phi_x_mean)
                    recons_x = self.recon_loss(x_tilde, x)  # Recons_loss
                else:
                    recons_x = torch.tensor(0)

                if len(self.decy._parameters) > 0:
                    phi_y_tilde = self.get_phi_y_tilde(self.H)
                    y_tilde = self.decy(phi_y_tilde + self.phi_y_mean)
                    recons_y = self.recon_loss(y_tilde, y)  # Recons_loss
                else:
                    recons_y = torch.tensor(0)
        return kpca_loss + recons_x + recons_y, kpca_loss, recons_x, recons_y

    @staticmethod
    def covariance_matrix(phi_x: torch.Tensor = None, phi_y: torch.Tensor = None):
        phi_ = torch.cat([phi_x, phi_y], dim=1)
        return phi_.T @ phi_  # Covariance matrix

    @staticmethod
    def kernel_matrix(phi_x: torch.Tensor = None, phi_y: torch.Tensor = None):
        phi_ = torch.cat([phi_x, phi_y], dim=1)
        return phi_ @ phi_.T  # kernel matrix

    def rescale_U(self, U_tilde: torch.Tensor = None, eig_vals: torch.Tensor = None):
        m = eig_vals.detach().cpu().numpy().astype(np.float_)
        sqrt_lambdas = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(self.eig_vals)
        return sqrt_lambdas.T @ U_tilde

    def get_phi_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns centered feature vector.
        """
        if x.ndimension() == 1:
            x = x.view(-1, 1)

        phi_x = self.encx(x)  # features

        if self.training:
            self.phi_x_mean = torch.mean(phi_x, dim=0)
        return phi_x - self.phi_x_mean  # feature centering

    def get_phi_y(self, y: torch.Tensor) -> torch.Tensor:
        """
        Returns centered feature vector.
        """
        if y.ndimension() == 1:
            y = y.view(-1, 1)

        phi_y = self.ency(y)  # features

        if self.training:
            self.phi_y_mean = torch.mean(phi_y, dim=0)
        return phi_y - self.phi_y_mean  # feature centering

    def get_phi_x_tilde(self, H: torch.Tensor):
        return H @ self.Ux

    def get_phi_y_tilde(self, H: torch.Tensor):
        return H @ self.Uy

    def get_h(self, **kwargs) -> torch.Tensor:
        """
        Get the latent vector H.
        """
        if "phi_x" in kwargs and "phi_y" in kwargs:
            if self.training is True or self.coeff_h_from_xy is None:
                self.coeff_h_from_xy = torch.linalg.inv(self.eig_vals)
            H = (
                        (kwargs["phi_x"] @ self.Ux.T) + (kwargs["phi_y"] @ self.Uy.T)
                ) @ self.coeff_h_from_xy

        elif "phi_x" in kwargs and "phi_y" not in kwargs:
            if self.training is True or self.coeff_h_from_x is None:
                self.coeff_h_from_x = (
                        torch.linalg.inv(self.eig_vals - (self.Uy @ self.Uy.T)) @ self.Ux
                )
            H = kwargs["phi_x"] @ self.coeff_h_from_x.T

        elif "phi_y" in kwargs and "phi_x" not in kwargs:
            if self.training is True or self.coeff_h_from_y is None:
                self.coeff_h_from_y = (
                        torch.linalg.inv(self.eig_vals - (self.Ux @ self.Ux.T)) @ self.Uy
                )
            H = kwargs["phi_y"] @ self.coeff_h_from_y.T
        else:
            raise ValueError("Input atleast one feature vector.")
        return H

    def get_U(self, **kwargs):
        """
        Returns either U, Ux, Uy
        """
        if "phi_x" in kwargs and "phi_y" in kwargs:
            U = torch.cat((kwargs["phi_x"].T @ self.H.T, kwargs["phi_y"].T @ self.H.T), dim=0)

        elif "phi_x" in kwargs and "phi_y" not in kwargs:
            U = kwargs["phi_x"].T @ self.H.T

        elif "phi_y" in kwargs and "phi_x" not in kwargs:
            U = kwargs["phi_y"].T @ self.H.T

        else:
            raise ValueError("Input atleast one feature vector.")
        return U

    def primal_loss(self, C: torch.Tensor) -> torch.Tensor:
        """
        Defines the PCA loss.
        """
        return torch.trace(C) - torch.trace((self.U_tilde @ C @ self.U_tilde.T))

    def dual_loss(self, K: torch.Tensor) -> torch.Tensor:
        """
        Defines the KPCA loss.
        """
        return 0.5 * (-torch.trace(self.H @ K @ self.H.T) + torch.trace(K))


def param_state(model):
    """
    Accumulate trainable parameters in 2 groups:
    1. Manifold_params (st)
    2. Network param (nn)
    """
    param_st, param_nn = [], []
    St_param_names = ["U_tilde", "H"]

    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in St_param_names:
                logger.info(f"Optimizing {name}")
                param_st.append(param)
            else:
                param_nn.append(param)
    return param_st, param_nn


def stiefel_opti(stief_param, lrg=1e-4):
    dict_st = {
        "params": stief_param,
        "lr": lrg,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "stiefel": True,
    }
    return stiefel_optimizer.AdamG([dict_st])  # CayleyAdam
