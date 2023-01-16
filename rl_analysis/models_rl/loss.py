import jax
from rl_analysis.models_rl.models import simulate


# switch to tm evaluation?
@jax.jit
def rl_loss(params, q_table, sequence, shifted_sequence, key):
    tup = simulate(params, q_table, sequence, key)
    shifted_tup = simulate(params, q_table, shifted_sequence, key)
    # return ll
    return tup, shifted_tup