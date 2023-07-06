import math

import torch
import torch.nn as nn
from sklearn.kernel_approximation import RBFSampler


class identity(torch.nn.Module):
    def __init__(self, args):
        super(identity, self).__init__()

    def forward(self, x):
        return x


class Encoder1(torch.nn.Module):
    def __init__(self, args):
        super(Encoder1, self).__init__()  # inheritance used here.
        self.args = args
        self.main = nn.Sequential(
            torch.nn.LazyLinear(self.args["dx"], 25),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(15),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(15),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(15),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(self.args["DIM_FEATURES_x"]),
            # torch.nn.Tanh()
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Decoder1(torch.nn.Module):
    def __init__(self, args):
        super(Decoder1, self).__init__()  # inheritance used here.
        self.args = args
        self.main = nn.Sequential(
            torch.nn.Linear(self.args["DIM_FEATURES_x"], 15),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(25),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(25),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(25),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(self.args["dx"]),
            # torch.nn.Tanh()
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Encoder2(torch.nn.Module):
    def __init__(self, args):
        super(Encoder2, self).__init__()  # inheritance used here.
        self.args = args
        self.main = nn.Sequential(
            torch.nn.Linear(self.args["dy"], 25),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(15),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(15),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(15),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.LazyLinear(self.args["DIM_FEATURES_y"]),
            # torch.nn.Tanh()
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Decoder2(torch.nn.Module):
    def __init__(self, args):
        super(Decoder2, self).__init__()  # inheritance used here.
        self.args = args
        self.main = nn.Sequential(
            torch.nn.Linear(self.args["DIM_FEATURES_y"], 15),
            torch.nn.LeakyReLU(negative_slope=0.2),
            # torch.nn.ReLU(),
            # torch.nn.Sigmoid(),
            torch.nn.LazyLinear(15),
            torch.nn.LeakyReLU(negative_slope=0.2),
            # torch.nn.ReLU(),
            # torch.nn.Sigmoid(),
            torch.nn.LazyLinear(15),
            torch.nn.LeakyReLU(negative_slope=0.2),
            # torch.nn.ReLU(),
            # torch.nn.Sigmoid(),
            torch.nn.LazyLinear(15),
            # torch.nn.LeakyReLU(negative_slope=0.2),
            # torch.nn.ReLU(),
            torch.nn.LazyLinear(self.args["dy"]),
            # torch.nn.ReLU(),
            #  torch.nn.Tanh()
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Random_Fourier_Encoder(torch.nn.Module):
    """
    Returns the random fourier features.
    """

    def __init__(self, args):
        super(Random_Fourier_Encoder, self).__init__()
        self.args = args
        if "sigma" not in self.args:
            self.args["sigma"] = 1

        gamma = 0.5 * 1 / self.args["sigma"] ** 2
        self.kernel_x_sampler = RBFSampler(
            n_components=self.args["DIM_FEATURES"],
            gamma=gamma,
            random_state=42,
        )  # gamma=5e-4

    def forward(self, x):
        return torch.tensor(
            self.kernel_x_sampler.fit_transform(x.cpu()), device=self.args["device"]
        )

