import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np
from jax import random


# regression model with continuous-valued outputs/responses
# http://num.pyro.ai/en/stable/examples/horseshoe_regression.html
def horseshoe_regression(X, Y, indicator=None):
    D_X = X.shape[1]

    # sample from horseshoe prior
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(D_X)))
    tau = numpyro.sample("tau", dist.HalfCauchy(jnp.ones(1)))

    # note that in practice for a normal likelihood we would probably want to
    # integrate out the coefficients (as is done for example in sparse_regression.py).
    # however, this trick wouldn't be applicable to other likelihoods
    # (e.g. bernoulli, see below) so we don't make use of it here.
    unscaled_betas = numpyro.sample("unscaled_w", dist.Normal(0.0, jnp.ones(D_X)))
    scaled_betas = numpyro.deterministic("w", tau * lambdas * unscaled_betas)

    # compute mean function using linear coefficients
    mean_function = jnp.dot(X, scaled_betas)

    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    numpyro.sample("y", dist.Normal(mean_function, sigma_obs), obs=Y)

def simple_logistic_regression(X, Y, indicator=None):
    if indicator is None:
        indicator = np.zeros((X.shape[0],), dtype="int")
    nindicator = len(set(indicator))
    w = numpyro.sample("w", dist.Normal(0, 10.).expand([nindicator, X.shape[1]]))
    intercept = numpyro.sample("intercept", dist.Normal(0., 10.))
    logits = (w[indicator] * X).sum(axis=1) + intercept
    y = numpyro.sample("y", dist.Bernoulli(logits=logits), obs=Y)


def ordinal_regression(X, Y, indicator=None, nclasses=1):
    
    if indicator is None:
        indicator = np.zeros((X.shape[0],), dtype="int")
    nindicator = len(set(indicator))

    if Y is not None:
        nclasses = len(set(Y))

    w = numpyro.sample("w", dist.Normal(0, 5).expand([nindicator, X.shape[1]]))
    c_y = numpyro.sample(
        "c_y",
        dist.ImproperUniform(
            support=dist.constraints.ordered_vector,
            batch_shape=(),
            event_shape=(nclasses - 1,),
        ),
    )
    eta = (X * w).sum(axis=1)
    y = numpyro.sample("y", dist.OrderedLogistic(eta, c_y), obs=Y)




def simple_regression(X, Y, indicator=None):
    if indicator is None:
        indicator = np.zeros((X.shape[0],), dtype="int")
    nindicator = len(set(indicator))
    w = numpyro.sample("w", dist.Normal(0, 0.5).expand([nindicator, X.shape[1]]))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0).expand([nindicator]))
    y = numpyro.sample("y", dist.Normal((X * w[indicator]).sum(axis=1), sigma[indicator]), obs=Y)


def robust_regression(X, Y, indicator=None):
    if indicator is None:
        indicator = np.zeros((X.shape[0],), dtype="int")
    nindicator = len(set(indicator))
    w = numpyro.sample("w", dist.Normal(0, 0.5).expand([nindicator, X.shape[1]]))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0).expand([nindicator]))
    nu = numpyro.sample("nu", dist.HalfNormal(2.0).expand([nindicator]))
    y = numpyro.sample(
        "y",
        dist.StudentT(df=nu[indicator], loc=(X * w[indicator]).sum(axis=1), scale=sigma[indicator]),
        obs=Y,
    )

# and hierarchical formulations (draw each feature from its own distribution????
def hierarchical_regression(X, Y, indicator):
    # we need to make distributions
    nindicator = len(set(indicator))

    mu = numpyro.sample("µ_a", dist.Normal(0.0, 1.0))
    sig = numpyro.sample("σ_a", dist.HalfNormal(1))

    w = numpyro.sample("w", dist.Normal(mu, sig).expand([nindicator, X.shape[1]]))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0).expand([nindicator]))
    y = numpyro.sample("y", dist.Normal((X * w[indicator]).sum(axis=1), sigma[indicator]), obs=Y)


# and hierarchical formulations (draw each feature from its own distribution????
def hierarchical_robust_regression(X, Y, indicator):
    # we need to make distributions
    nindicator = len(set(indicator))

    mu = numpyro.sample("µ_a", dist.Normal(0.0, 1.0))
    sig = numpyro.sample("σ_a", dist.HalfNormal(1))

    w = numpyro.sample("w", dist.Normal(mu, sig).expand([nindicator, X.shape[1]]))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0).expand([nindicator]))
    nu = numpyro.sample("nu", dist.HalfNormal(2.0).expand([nindicator]))
    y = numpyro.sample(
        "y",
        dist.StudentT(df=nu[indicator], loc=(X * w[indicator]).sum(axis=1), scale=sigma[indicator]),
        obs=Y,
    )

@jax.jit
def jitted_split(key):
    key, subkey = random.split(key)
    return key, subkey


def parallel_pred(samples, rng_key, X, func=None, ncpus=len(jax.devices("cpu"))):
	from jax.experimental.maps import soft_pmap
	import warnings
    
	keys = list(samples.keys())
	rem = len(samples[keys[0]]) % ncpus
	test_samples = {k: v[:len(v) - rem] for k, v in samples.items()}

	# we need to make the keys ahead of time
	rng_keys = []
	for i in range(len(test_samples[keys[0]])):
		rng_key, rng_key_ = jitted_split(rng_key)
		rng_keys.append(rng_key_) 

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		pred = soft_pmap(func, in_axes=[0, 0, None])(test_samples, np.array(rng_keys), X)
	return pred