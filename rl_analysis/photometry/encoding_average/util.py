from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from scipy.stats import zscore
import numpy as np
from typing import Optional

mcmc_defaults = {"num_warmup": 1000, "num_samples": 3000}


def run_inference(
    x: np.ndarray,
    y: np.ndarray,
    use_model,
    feature_indicator: Optional[np.ndarray] = None,
    rng_key=None,
    ncpus: int = 10,
    standardize: bool = True,
    mcmc_kwargs: dict = {},
) -> dict:
    import warnings

    use_mcmc_kwargs = {**mcmc_kwargs, **mcmc_defaults}
    kernel = NUTS(use_model)
    mcmc = MCMC(
        kernel,
        num_chains=ncpus,
        chain_method="parallel",
        progress_bar=False,
        **use_mcmc_kwargs,
    )

    if standardize:
        use_x = zscore(x)
        use_y = zscore(y)
    else:
        use_x = x
        use_y = y

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        mcmc.run(rng_key, use_x, use_y, feature_indicator)

    samples = mcmc.get_samples()
    return samples


def run_inference_and_ppc(
    train_x,
    train_y,
    test_x,
    train_ind,
    test_ind,
    use_model,
    ncpus=10,
    preproc=None,
    mcmc_kwargs={},
    rng_key=None,
    standardize=False,
):

    import warnings

    use_mcmc_kwargs = {**mcmc_kwargs, **mcmc_defaults}
    kernel = NUTS(use_model)
    mcmc = MCMC(
        kernel,
        num_chains=ncpus,
        chain_method="parallel",
        progress_bar=False,
        **use_mcmc_kwargs,
    )

    if standardize:
        use_train_x = zscore(train_x)
        use_train_y = zscore(train_y)
        use_test_x = zscore(test_x)
    else:
        use_train_x = train_x
        use_train_y = train_y
        use_test_x = test_x

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        mcmc.run(rng_key, use_train_x, use_train_y, train_ind)
    samples = mcmc.get_samples()
    predictive = Predictive(use_model, samples)

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        mcmc.run(rng_key, use_train_x, use_train_y, train_ind)
        ppc = predictive(rng_key, use_test_x, None, test_ind)
    ppc["indicator"] = test_ind

    if preproc is not None:
        ppc["y"] = preproc.inverse_transform(ppc["y"])

    return samples, ppc
