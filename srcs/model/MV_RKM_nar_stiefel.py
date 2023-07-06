import logging
from typing import Tuple

import hydra
import torch
from tqdm import tqdm

from srcs.data_loader.data_iterator import BatchIterator
from srcs.model.stiefel_rkm_model import RKM_Stiefel, param_state, stiefel_opti
from srcs.utils.util import conditional_trange, convert_to_AR, str_to_func
from srcs.model.pre_image_method import kernel_smoother

logger = logging.getLogger(__name__)


class checkpoint:
    def __init__(self):
        self.cp_dir = f"{hydra.utils.os.getcwd()}/cp/"
        self.cp_name = "checkpoint.pt"

        if not hydra.utils.os.path.exists(self.cp_dir):
            hydra.utils.os.makedirs(self.cp_dir)

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, f"{self.cp_dir}/{self.cp_name}")


class MV_RKM_nar_stiefel(checkpoint):
    def __init__(self, **kwargs: dict) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.lag = self.kwargs["lag"]
        self.s = self.kwargs["s"]
        self.iterator = BatchIterator(
            batch_size=self.kwargs["mb_size"], shuffle=self.kwargs["shuffle"]
        )

        # higher priority to user defined device
        # otherwise default to 'cuda'
        if "device" not in self.kwargs:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.kwargs["device"] = "cuda"
            else:
                self.kwargs["device"] = "cpu"
        self.device = self.kwargs["device"]

        if "recon_loss" not in self.kwargs:
            self.kwargs["recon_loss"] = torch.nn.MSELoss(reduction="sum")
        elif type(self.kwargs["recon_loss"]) is str:
            self.kwargs["recon_loss"] = str_to_func(self.kwargs["recon_loss"])

    def train(self, X: torch.Tensor, Y: torch.Tensor, verbose: bool = True) -> None:
        """
        Train NN and Stiefel params.
        """
        if X.ndim == 1:
            X = X.unsqueeze(1)  # Shape the data matrix as n x d
        if Y.ndim == 1:
            Y = Y.unsqueeze(1)  # Shape the data matrix as n x d

        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.kwargs["n"] = self.X.shape[0]
        self.kwargs["dx"] = self.X.shape[1]
        self.kwargs["dy"] = self.Y.shape[1]

        # Init Model  --------------------------------------------------
        self.rkm = RKM_Stiefel(args=self.kwargs).double().to(self.device)

        # Init Optimizers  --------------------------------------------------
        param_st, param_nn = param_state(self.rkm)
        self.optimizer1 = stiefel_opti(param_st, self.kwargs["lrg"])
        if len(param_nn) > 0:
            self.optimizer2 = torch.optim.Adam(param_nn, lr=self.kwargs["lr"])

        # Train loop -------------------------------------------------------
        Loss_stk = torch.empty(size=[0, 4])
        cost = torch.tensor([float("inf")])
        l_cost = torch.tensor([float("inf")])
        mini_batches = torch.ceil(torch.tensor(self.n / self.kwargs["mb_size"]))

        if verbose is True:
            pbar = tqdm(range(self.kwargs["max_epochs"]), desc="Description")
        else:
            pbar = range(self.kwargs["max_epochs"])

        for _ in pbar:  # Run until convergence or cut-off
            avg_loss, avg_kpca, avg_recons_x, avg_recons_t = 0, 0, 0, 0

            for batch in self.iterator(self.X, self.Y):
                loss, kpca, recons_x, recons_t = self.rkm(
                    batch.inputs.double().to(self.device),
                    batch.targets.double().to(self.device),
                )

                self.optimizer1.zero_grad()
                if len(param_nn) > 0:
                    self.optimizer2.zero_grad()
                loss.backward()
                if len(param_nn) > 0:
                    self.optimizer2.step()
                self.optimizer1.step()

                avg_loss += loss.item()
                avg_kpca += kpca.item()
                avg_recons_x += recons_x.item()
                avg_recons_t += recons_t.item()

            # Overall dataset statistics
            avg_loss = torch.floor(avg_loss / mini_batches)
            avg_kpca = torch.floor(avg_kpca / mini_batches)
            avg_recons_x = torch.floor(avg_recons_x / mini_batches)
            avg_recons_t = torch.floor(avg_recons_t / mini_batches)
            cost = avg_loss

            # Remember the lowest cost & save checkpoint
            state = {
                "rkm": self.rkm,
                "rkm_state_dict": self.rkm.state_dict(),
                "optimizer1": self.optimizer1.state_dict(),
                "Loss_stk": Loss_stk,
                "kwargs": self.kwargs,
            }
            if len(param_nn) > 0:
                state["optimizer2"] = self.optimizer2.state_dict()
            self.save_checkpoint(state, is_best=cost < l_cost)
            l_cost = min(cost, l_cost)

            if verbose is True:
                desc = (
                    f"Loss:{cost:.3f}|Kpca:{avg_kpca:.3f}|"
                    f"Rec_x:{avg_recons_x:.3f}|Rec_y:{avg_recons_t:.3f}"
                )
                pbar.set_description(desc=desc)

            Loss_stk = torch.cat(
                (
                    Loss_stk,
                    torch.Tensor([[cost, avg_kpca, avg_recons_x, avg_recons_t]]),
                ),
                dim=0,
            )

            # Terminate the loop when a threshold is reached
            if l_cost < 1e-8:
                break

        # Load best Checkpoint and save model ==================
        logger.info(
            f"Finished Training. Lowest cost: {l_cost}"
            f"\nLoading best checkpoint {self.cp_dir} & computing sub-space..."
        )

        sd_mdl = torch.load(f"{self.cp_dir}{self.cp_name}")
        self.rkm.load_state_dict(sd_mdl["rkm_state_dict"])

        # EVAL mode ================================
        self.rkm.eval()
        # Pre-compute certain things
        self.phi_y = self.rkm.get_phi_y(self.Y.double().to("cpu"))
        phi_x = self.rkm.get_phi_x(self.X.double().to("cpu"))
        self.rkm.args["DIM_FEATURES_x"] = phi_x.shape[1]
        if self.kwargs["mode"] == "primal":
            self.H = self.rkm.get_h(phi_x=phi_x, phi_y=self.phi_y).detach()  # latent points
            state["H"] = self.H
        elif self.kwargs["mode"] == "dual":
            state["H"] = self.rkm.H.T.detach()
            self.rkm.U = self.rkm.get_U(phi_x=phi_x, phi_y=self.phi_y).T
            self.rkm.Ux = self.rkm.U[:, : self.rkm.args["DIM_FEATURES_x"]]
            self.rkm.Uy = self.rkm.U[:, self.rkm.args["DIM_FEATURES_x"]:]

        state["new_H"], state["new_lambdas"] = self.scaling_with_rotation_matrix()
        state["Y"] = self.Y
        state["phi_y"] = self.phi_y
        state["n"] = self.X.shape[0]
        torch.save(
            state,
            f"{hydra.utils.os.getcwd()}/model_{self.__str__().split('_')[-1].split(' ')[0]}.pt",
        )
        logger.info(
            f"\nSaved model: {hydra.utils.os.getcwd()}/model_{self.__str__().split('_')[-1].split(' ')[0]}.pt"
        )

    def predict(
        self,
        init_vec: torch.tensor = None,
        n_steps: int = 1,
        verbose=True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the tensor (n_steps x d) of predicted values.
        """
        if init_vec is None:
            init_vec = convert_to_AR(
                data=self.Y[-(self.lag + 1) :],
                lag=self.lag,
                n_steps_ahead=self.kwargs["n_steps_ahead"],
            )
        if init_vec.ndim == 1:
            init_vec = init_vec.unsqueeze(0)

        self.H_pred = self.H.clone()
        self.Y_pred = self.Y.clone()

        self.rkm.eval()

        for i in conditional_trange(verbose, n_steps, desc="Predicting"):
            # Get input
            if i == 0:
                x_star = init_vec
            else:
                x_star = convert_to_AR(
                    data=self.Y_pred[-(self.lag + 1) :],
                    lag=self.lag,
                    n_steps_ahead=self.kwargs["n_steps_ahead"],
                ).detach()

            phi_x_star = self.rkm.get_phi_x(x_star.to(self.device))

            # Predict next latent variable
            h_n_ahead = self.rkm.get_h(phi_x=phi_x_star)
            self.H_pred = torch.cat((self.H_pred, h_n_ahead), dim=0)

            # get feature vector
            phi_y_tilde = self.rkm.get_phi_y_tilde(h_n_ahead)

            # if self.rkm.ency
            if "Random_Fourier_Encoder" in self.rkm.ency.__str__():
                y_tilde = self.pre_image(data=phi_y_tilde + self.rkm.phi_y_mean)
            else:
                y_tilde = self.rkm.decy(phi_y_tilde + self.rkm.phi_y_mean)  # pre-image

            if h_n_ahead.ndim == 1:
                h_n_ahead = h_n_ahead.unsqueeze(0)
            if y_tilde.ndim == 1:
                y_tilde = y_tilde.unsqueeze(0)

            self.Y_pred = torch.cat((self.Y_pred, y_tilde), dim=0)

        return self.Y_pred[-n_steps:].detach()

    def pre_image(self, data: torch.tensor, verbose=True):
        """
        Implements pre-image method
        """

        if self.kwargs["pre_image_method"] == "kernel_smoother":
            similarities = self.phi_y @ (data - self.rkm.phi_y_mean).T
            ks = kernel_smoother(nearest_neighbours=self.kwargs["nearest_neighbours"])
            op = ks(x=self.Y[: self.kwargs['n']], kernel_vector=similarities)
        elif self.kwargs["pre_image_method"] == "ridge_regression":
            raise ValueError("ridge_regression not yet implemented.")
        else:
            raise ValueError(
                f"pre_image_method {self.kwargs['pre_image_method']} not valid."
            )

        if op.ndim == 1:  # Single dimensional data
            op = op.unsqueeze(-1)
        return op

    def scaling_with_rotation_matrix(self):
        """
        returns new_H
        """
        O, lambdas, _ = torch.linalg.svd(self.rkm.eig_vals)
        if self.kwargs["mode"] == 'primal':
            new_U_tilde = O.T @ self.rkm.U_tilde
            new_U = torch.diag(torch.sqrt(lambdas)).T @ new_U_tilde

            phi_x = self.rkm.get_phi_x(self.X.double().to("cpu"))
            phi_y = self.rkm.get_phi_y(self.Y.double().to("cpu"))

            new_H = (
                (phi_x @ new_U[:, : self.rkm.args["DIM_FEATURES_x"]].T)
                + (phi_y @ new_U[:, self.rkm.args["DIM_FEATURES_x"] :].T)
            ) @ torch.diag(lambdas**-1)
            return new_H.detach().cpu().numpy(), torch.diag(lambdas).detach().cpu().numpy()
        elif self.kwargs["mode"] == 'dual':
            new_H = self.rkm.H.T @ O
            return new_H.detach().cpu().numpy(), torch.diag(lambdas).detach().cpu().numpy()
