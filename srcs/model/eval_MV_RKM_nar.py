import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import DictConfig

from srcs.utils.util import (
    instantiate,
    standardize,
    savepdf_tex,
    scatter_plot_with_histogram,
)

plt.style.use("seaborn-bright")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.colheader_justify", "center")
pd.set_option("display.precision", 3)


class Eval_MV_RKM_nar:
    def __init__(
        self,
        config: DictConfig = None,
        pre_trained_model_path: str = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.logger.info(f"Loading pre-trained model: {pre_trained_model_path}")
        sd_mdl = torch.load(
            f"{pre_trained_model_path}",
            map_location=torch.device(f"{self.config.hyperparameters.device}"),
        )

        self.model = instantiate(self.config.arch, **sd_mdl["kwargs"])  # Init model
        self.save_dir = str.split(pre_trained_model_path, "/model_")[0]

        if "stiefel" in self.model.__str__():
            self.model.rkm = sd_mdl["rkm"]
            self.model.rkm.load_state_dict(sd_mdl["rkm_state_dict"])
            self.model.device = "cpu"
            self.model.Y = sd_mdl["Y"]
            self.model.phi_y = sd_mdl["phi_y"]
            self.model.H = sd_mdl["H"]
            self.model.new_H = sd_mdl["new_H"]
            self.model.new_lambdas = sd_mdl["new_lambdas"]
            self.train_losses = sd_mdl["Loss_stk"]
        else:
            for k, v in sd_mdl["rkm_state_dict"].items():
                self.model.__dict__[f"{k}"] = v

        #  Load data
        self.train_data, self.test_data = instantiate(self.config.data)

        # Standardize data.
        std_train_data, self.train_data_mean, self.train_data_std = standardize(
            self.train_data.float()
        )

        # Test
        n_steps = self.test_data.shape[0]
        self.Y_pred = self.model.predict(n_steps=n_steps).cpu()
        self.Y_pred = (
            self.train_data_std * self.Y_pred
        ) + self.train_data_mean  # Un-standardization

    def eval_metrics(self):
        """eval_metrics. Computes metrics as defined in config like mse, mae etc"""

        # Init metrics
        metrics = [instantiate(met, is_func=True) for met in self.config["metrics"]]

        scores = []
        for met in metrics:
            scores.append(
                [
                    f"Test error {met.__name__}",
                    met(self.test_data, self.Y_pred.squeeze()).item(),
                ]
            )

        extra_metrics = [
            [
                "||H.T @ H - I_s||",
                torch.norm(
                    self.model.H.T @ self.model.H
                    - torch.eye(
                        self.model.H.shape[1], device=self.config.hyperparameters.device
                    )
                ).item(),
            ],
        ]

        df = pd.DataFrame(extra_metrics, columns=["Metrics", "Value"])
        df = pd.concat(
            [df, pd.DataFrame(scores, columns=["Metrics", "Value"])], ignore_index=True
        )
        self.logger.info(
            f"\nMetric for {self.model.__str__().split('_')[-1].split(' ')[0]} model: \n {df}"
        )

    def plot_full_data(self):
        """Plot train/test data."""

        _, ax = plt.subplots()
        ax.plot(
            np.arange(0, self.train_data.shape[0]),
            self.train_data,
            "b",
            label="Train",
            linewidth=1,
        )
        ax.plot(
            np.arange(
                self.train_data.shape[0],
                self.train_data.shape[0] + self.test_data.shape[0],
            ),
            self.test_data,
            "r",
            label="Test",
            linewidth=1,
        )
        ax.set_title("Time series (Full dataset)")
        ax.grid()
        ax.legend(loc="upper right")
        plt.savefig(f"{self.save_dir}/Dataset_full.svg", format="svg", dpi=800)
        plt.show()

    def plot_preds(self):
        # Plot test data
        _, ax = plt.subplots()
        ax.plot(
            np.arange(
                self.train_data.shape[0],
                self.train_data.shape[0] + self.test_data.shape[0],
            ),
            self.test_data.squeeze(),
            "b",
            label="Ground-truth",
            linewidth=1,
        )
        ax.plot(
            np.arange(
                self.train_data.shape[0],
                self.train_data.shape[0] + self.Y_pred.shape[0],
            ),
            self.Y_pred.squeeze(),
            "g",
            label="Prediction",
            linewidth=1,
        )
        # ax.set_title(f"Time series Prediction", wrap=True)
        ax.grid()
        ax.set_title(
            f"Predictions ({self.model.__str__().split('_')[-1].split(' ')[0]})"
        )
        plt.xlabel("Time steps")
        plt.tight_layout(pad=0.1)
        ax.legend(loc="upper right")
        plt.savefig(
            f"{self.save_dir}/prediction_{self.model.__str__().split('_')[-1].split(' ')[0]}.svg",
            format="svg",
            dpi=1200,
            transparent=True,
        )
        savepdf_tex(
            filename=f"{self.save_dir}/prediction_{self.model.__str__().split('_')[-1].split(' ')[0]}.svg"
        )
        plt.show()

        if self.test_data.ndim > 1:
            # 3D plot for multi-dimensional data
            ax = plt.figure().add_subplot(projection="3d")
            ax.plot(
                self.test_data[:, 0],
                self.test_data[:, 1],
                self.test_data[:, 2],
                "b",
                label="Ground-truth",
                linewidth=1,
            )
            ax.plot(
                self.Y_pred[:, 0],
                self.Y_pred[:, 1],
                self.Y_pred[:, 2],
                "g",
                label="Prediction",
                linewidth=1,
            )
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            plt.tight_layout(pad=0.2)
            ax.legend()
            ax.set_title(
                f"Predictions ({self.model.__str__().split('_')[-1].split(' ')[0]})"
            )
            plt.savefig(
                f"{self.save_dir}/prediction3D_{self.model.__str__().split('_')[-1].split(' ')[0]}.svg",
                format="svg",
                dpi=1200,
                transparent=True,
            )
            savepdf_tex(
                filename=f"{self.save_dir}/prediction3D_{self.model.__str__().split('_')[-1].split(' ')[0]}.svg"
            )
            plt.show()

    def plot_latents(self):

        scatter_plot_with_histogram(
            self.model.H_pred,
            histogram=False,
            save_path=self.save_dir,
            train_size=self.model.n,
            title="H of Train, Test set",
        )

    def plot_cov_H(self):
        cov = self.model.H.T @ self.model.H
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(r"$H^{\top} H$")
        ax.imshow(cov, interpolation="nearest")
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/cov_H_{self.model.__str__().split('_')[-1].split(' ')[0]}.svg",
            format="svg",
            dpi=800,
        )
        savepdf_tex(
            filename=f"{self.save_dir}/cov_H_{self.model.__str__().split('_')[-1].split(' ')[0]}.svg"
        )
        plt.show()

    def plot_eigenfunctions(
        self,
        matrix: torch.Tensor = None,
        nr_components: int = None,
        title: str = "Eigenfunctions",
    ):
        if matrix is None:
            matrix = self.model.H.cpu()
        if nr_components is None:
            nr_components = min(matrix.shape[1], 8)  # matrix.shape[1]

        import itertools

        colors = itertools.cycle(["r", "g", "b", "c", "y", "m", "k"])
        # markers = itertools.cycle(['.','*',',', '+', 'x', 'd'])

        fig, ax = plt.subplots(nr_components, 1, sharex=True)
        try:
            for i in range(nr_components):
                ax[i].plot(
                    matrix[:, i],
                    c=next(colors),
                    # marker=next(markers),
                    label=f"{i + 1}",
                    lw=1,
                )
                ax[i].grid()
                ax[i].legend(loc="upper right")
        except Exception as e:
            ax.plot(
                matrix,
                c=next(colors),
                # marker=next(markers),
                label=f"{1}",
                lw=1,
            )
            ax.grid()
            ax.legend(loc="upper right")

        plt.xlabel("Time steps")
        fig.suptitle(f"{title} ({self.model.__str__().split('_')[-1].split(' ')[0]})")
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{title}_{self.model.__str__().split('_')[-1].split(' ')[0]}.svg",
            format="svg",
            dpi=800,
        )
        savepdf_tex(
            filename=f"{self.save_dir}/{title}_{self.model.__str__().split('_')[-1].split(' ')[0]}.svg"
        )
        plt.show()

    def plot_eigenspectrum(self):
        """
        Plot normalized eigenvalues and show explained variance.
        """
        plt.figure()
        plt.bar(
            range(0, self.model.lambdas.shape[0]),
            torch.diag(self.model.lambdas),
            alpha=0.5,
            align="center",
            label=r"Eigvals",
        )
        plt.xlabel(
            f"Principal component index (Upto latent space dim = {self.model.lambdas.shape[0]})"
        )
        plt.legend(loc="best")
        plt.title(
            f"Eigenspectrum ({self.model.__str__().split('_')[-1].split(' ')[0]})"
        )
        plt.grid()
        plt.tight_layout(pad=0.2)
        plt.savefig(
            f"{self.save_dir}/eigenspectrum_{self.model.__str__().split('_')[-1].split(' ')[0]}.svg",
            format="svg",
            dpi=800,
        )
        savepdf_tex(
            filename=f"{self.save_dir}/eigenspectrum_{self.model.__str__().split('_')[-1].split(' ')[0]}.svg"
        )
        plt.show()

    def plot_feat_dimensions(
        self,
        matrix: torch.Tensor = None,
        nr_components: int = None,
    ):
        if matrix == None:
            matrix = self.model.Ux

        if nr_components is None:
            nr_components = 8  # matrix.shape[1]

        import itertools

        colors = itertools.cycle(["r", "g", "b", "c", "y", "m", "k"])
        # markers = itertools.cycle(['.','*',',', '+', 'x', 'd'])

        fig, ax = plt.subplots(nr_components, 1, sharex=True)
        for i in range(nr_components):
            ax[i].plot(
                matrix[:, i],
                c=next(colors),
                # marker=next(markers),
                label=f"{i + 1}",
                lw=1,
            )
            ax[i].grid()
            ax[i].legend(loc="upper right")
        # plt.title("Latent components")
        plt.xlabel("Feature space dimensions")
        plt.tight_layout(pad=0.2)
        plt.savefig(f"{self.save_dir}/latent_components.svg", format="svg", dpi=800)
        savepdf_tex(filename=f"{self.save_dir}/latent_components.svg")
        plt.show()

    def traversal(self):
        # # Interpolation along principal components ================
        for i in range(self.model.s):
            dim = i
            m = 35  # Number of steps
            mul_off = 0.0  # (for no-offset, set multiplier to 0)

            # Manually set the linspace range or get from Unit-Gaussian
            lambd = torch.linspace(-0.5, 0.5, steps=m)
            # lambd = torch.linspace(*utils._get_traversal_range(0.475), steps=m)

            uvec = torch.FloatTensor(torch.zeros(self.model.H_pred.shape[1]))
            uvec[dim] = 1  # unit vector
            yoff = mul_off * torch.ones(self.model.H_pred.shape[1]).float()
            yoff[dim] = 0

            # Traversal vectors
            yop = yoff.repeat(lambd.size(0), 1) + torch.mm(
                torch.diag(lambd), uvec.repeat(lambd.size(0), 1)
            )
            x_gen = (
                rkm.decoder(torch.mm(yop, U.t()).float())
                .detach()
                .numpy()
                .reshape(-1, nChannels, WH, WH)
            )

            # Save Images in the directory
            if not os.path.exists(
                "Traversal_imgs/{}/{}/{}".format(opt.dataset_name, opt.filename, dim)
            ):
                os.makedirs(
                    "Traversal_imgs/{}/{}/{}".format(
                        opt.dataset_name, opt.filename, dim
                    )
                )

            for j in range(x_gen.shape[0]):
                scipy.misc.imsave(
                    "Traversal_imgs/{}/{}/{}/{}im{}.png".format(
                        opt.dataset_name, opt.filename, dim, dim, j
                    ),
                    utils.convert_to_imshow_format(x_gen[j, :, :, :]),
                )

        print(
            "Traversal Images saved in: Traversal_imgs/{}/{}/".format(
                opt.dataset_name, opt.filename
            )
        )

    def animate_trajectory(self, title="Trajectory latent_space"):
        # if self.H is None:
        #     self.plot_latents()

        x = self.model.H_pred
        n = self.model.n

        # THE DATA POINTS
        dataSet = x[:, :3].T  # np.array([x, y, t])
        numDataPoints = dataSet.shape[1]

        # GET SOME MATPLOTLIB OBJECTS
        fig = plt.figure()
        ax = Axes3D(fig)

        # NOTE: Can't pass empty arrays into 3d version of plot()
        line_tr = ax.plot(
            dataSet[0][:n], dataSet[1][:n], dataSet[2][:n], "-", c="b", linewidth=1
        )[0]
        line_pred = ax.plot(
            dataSet[0][n], dataSet[1][n], dataSet[2][n], "-", c="g", linewidth=1
        )[0]

        dot = ax.plot(dataSet[0], dataSet[1], dataSet[2], "o", c="b")[0]

        # Axes Properties
        # ax.set_xlim3d([limit0, limit1])
        ax.set_xlabel(r"$H_{x(t)}$")
        ax.set_ylabel(r"$H_{y(t)}$")
        ax.set_zlabel(r"$H_{z(t)}$")
        ax.set_title(f"{title}")

        # ANIMATION FUNCTION
        def func(num):
            # NOTE: there is no .set_data() for 3 dim data...
            dot.set_data(dataSet[0:2, num])
            dot.set_3d_properties(dataSet[2, num])
            if num <= n:
                line_tr.set_data(dataSet[0:2, : num + 1])
                line_tr.set_3d_properties(dataSet[2, : num + 1])
                dot.set_color("b")
            else:
                line_tr.set_data(dataSet[0:2, :n])
                line_tr.set_3d_properties(dataSet[2, :n])
                line_pred.set_data(dataSet[0:2, n : num + 1])
                line_pred.set_3d_properties(dataSet[2, n : num + 1])
                dot.set_color("g")

            return line_tr, line_pred, dot

        # Creating the Animation object
        line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints)
        line_ani.save(f"{hydra.utils.os.getcwd()}/AnimationNew.mp4")

        plt.show()

    def plot_train_losses(self):
        _, ax = plt.subplots()
        ax.plot(torch.log(self.train_losses[:, 0]), label="Total loss", linewidth=1)
        ax.plot(torch.log(self.train_losses[:, 1]), label="kpca loss", linewidth=1)
        ax.plot(torch.log(self.train_losses[:, 2]), label="recon_x", linewidth=1)
        ax.plot(torch.log(self.train_losses[:, 3]), label="recon_y", linewidth=1)

        ax.set_title("Training losses")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Log loss")
        ax.grid()
        ax.legend(loc="upper right")
        plt.savefig(f"{self.save_dir}/train_losses.svg", format="svg", dpi=800)
        plt.show()
