import logging
import os
import warnings

import hydra
import numpy
from omegaconf import DictConfig, OmegaConf

from srcs.model.eval_MV_RKM_nar import Eval_MV_RKM_nar
from srcs.utils.util import convert_to_AR, instantiate, standardize

warnings.filterwarnings("ignore")
os.environ["HYDRA_FULL_ERROR"] = "1"
logger = logging.getLogger("main")


@hydra.main(config_path="conf/", config_name="train_rrkm_nar")
def main(config: DictConfig) -> None:
    params = OmegaConf.to_container(config.hyperparameters)

    targets = config.arch._target_
    for target in targets:
        config.arch._target_ = target
        if "pre_trained_model_path" not in params:
            logger.info("\n\n Config: {}\n".format(config))

            # Load & standardize data
            train_data, _ = instantiate(config.data)
            train_data, _, _ = standardize(train_data.double())
            X_ar, Y_ar = convert_to_AR(
                data=train_data,
                lag=params["lag"],
                n_steps_ahead=params["n_steps_ahead"],
            )

            # Train
            rkm = instantiate(config.arch, **params)
            rkm.train(X=X_ar, Y=Y_ar)
            model_pth = f"{hydra.utils.os.getcwd()}/model_{rkm.__str__().split('_')[-1].split(' ')[0]}.pt"
        else:
            model_pth = (
                f'{hydra.utils.get_original_cwd()}/{params["pre_trained_model_path"]}'
            )

        # Evaluate Model
        eval_mdl = Eval_MV_RKM_nar(
            config=config,
            pre_trained_model_path=model_pth,
        )

        eval_mdl.eval_metrics()
        eval_mdl.plot_preds()
        eval_mdl.plot_eigenfunctions(title="Eigenfunctions (1)")

        if "stiefel" in eval_mdl.model.__str__():
            eval_mdl.plot_eigenfunctions(
                matrix=eval_mdl.model.new_H, title="Eigenfunctions (2)"
            )

            print(f"new_lambdas: {numpy.diag(eval_mdl.model.new_lambdas)}")
            print(f"Cov(H): {eval_mdl.model.H.T @ eval_mdl.model.H}")

    logger.info("--END--")


if __name__ == "__main__":
    main()
