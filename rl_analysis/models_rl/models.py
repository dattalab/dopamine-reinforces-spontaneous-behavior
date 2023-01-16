from functools import partial
import jax.numpy as jnp
import numpy as np
import jax


# RL model with dynamic temperature inner loop:
# 
# Obeys the standard Q-learning formulation w/ a 
# softmax policy, except temperature is allowed to
# vary over time
#
# Note that this uses the jax.lax.scan mechanism
# carry variables:
#   ll: total log-likelihood
#   prev_action: previous action taken by model [CURRENTLY DEPRECATED]
#   temperature: current temperature
#   q_table: current q_table
#   key: jax PRNG key
# 
# step variables:
#   choice: syllable selected by mouse on this timestep
#   next_choice: next syllable selected by mouse
#   reward: DA at current timestep
#
# alpha: learning rate
# gamma: discount rate
# temperature_baseline: baseline temperature
# temperature_alpha: temperature exponential smoothing timescale
# temperature_threshold1: if rew > x, increase temp
# temperature_threshold2: if rew < x, decrease temp
# temperature_peak: value to add/subtract if threshold conditions 1 or 2 are met
# eps: added to prevent log(0)
def sim_body(
    carry,
    step,
    alpha=0.2,
    gamma=0.2,
    temperature_baseline=1.0,
    temperature_alpha=0.2,
    temperature_threshold1=3,
    temperature_threshold2=0,
    temperature_peak=0.25,
    eps=1e-7,
):
    ll, prev_action, temperature, q_table, key = carry
    choice, next_choice, reward = step

    # make sure nothing funky happened here...
    # CHOICE -> what mouse did at T
    # NEXT_CHOICE -> what mouse did at T + 1
    # TEMPERATURE -> current temperature value
    # Q_TABLE -> q_table
    # KEY -> prng key
    choice = jnp.int16(choice)
    next_choice = jnp.int16(next_choice)
    prev_action = jnp.int16(prev_action) # DEPRECATED

    key, subkey = jax.random.split(key)

    # temperature uses simple exponential smoothing
    temperature = (
        temperature_alpha * temperature_baseline + (1 - temperature_alpha) * temperature
    )

    # CONDITION 1: if rew > x, increment temperature
    temperature = jax.lax.cond(
        reward > temperature_threshold1,
        lambda x: x + temperature_peak,
        lambda x: x,
        temperature,
    )

    # CONDITION 2: if rew < x, decrement temperature
    temperature = jax.lax.cond(
        reward < temperature_threshold2,
        lambda x: x - temperature_peak,
        lambda x: x,
        temperature,
    )

    # bit of a hack, but need to prevent numerical oddities for now...
    temperature = jnp.clip(temperature, 1e-2, 100)

    # stable softmax
    p_vec = (q_table[choice] - jnp.nanmax(q_table[choice])) / temperature
    p_vec = jnp.exp(p_vec)
    ps = p_vec / jnp.nansum(p_vec)

    # swap out nans
    ps = jnp.nan_to_num(ps, nan=0)
    action = jax.random.choice(subkey, jnp.arange(len(q_table)), p=ps)

    # note that out of bounds will return the end value...
    # do we update using action or next choice?
    cur_value = q_table[choice, action]
    new_value = reward + gamma * jnp.nanmax(q_table[action, :])
    td = alpha * (new_value - cur_value)

    # if we landed on a nan just don't update the q-table
    td = jnp.nan_to_num(td, nan=0)
    q_table = q_table.at[choice, action].add(td)
    cur_ll = jnp.log(ps[next_choice] + eps)

    return (ll + cur_ll, jnp.int16(action), temperature, q_table, key), (
        cur_ll,
        ps,
        action,
    )


# simulation wrapper
@jax.jit
def simulate(params, q_table, sequence, key):
    _sim_body = partial(sim_body, **params)
    # key = jax.random.PRNGKey(0)
    (ll, action, final_temperature, q_table, key), cur_ll = jax.lax.scan(
        _sim_body,
        (0, jnp.int16(sequence[0, 0]), params["temperature_baseline"], q_table, key),
        sequence,
    )
    return ll, cur_ll, q_table





# def simulate_active(q_table, rewards, rng=None, alpha=.1, gamma=.6, temperature=1., nsteps=10000, da_mapping="reward", **kwargs): 
#     state = 0
#     actions = []
#     reward_history = []
    
#     if rng is None:
#         rng = np.random.default_rng()
    
#     for i in range(nsteps):
    
#         ps = stable_softmax(q_table[state], temperature)
#         ps[np.isnan(ps)] = 0
#         action = rng.choice(np.arange(len(ps)), p=ps)
        
#         try:
#             reward = rng.choice(rewards[state, action])
#         except ValueError:
#             continue

#         if np.isnan(q_table[state,action]):
#             continue
            
#         if da_mapping == "reward":
#             q_value_predict = q_table[state, action]
#             q_value_real = reward + gamma * np.nanmax(q_table[action])
#             q_table[state, action] += alpha * (q_value_real - q_value_predict)
#         elif da_mapping == "rpe":
#             q_table[state, action] += alpha * reward
#         state = action
#         actions.append(action)
#         reward_history.append(reward)
        
#     return q_table, np.array(actions), reward_history


# def simulate_active_dynamic(
#     q_table,
#     rewards,
#     alpha=0.1,
#     gamma=0.6,
#     temperature_baseline=1.0,
#     temp_tau=10.,
#     temp_threshold1=1,
#     temp_threshold2=-np.inf,
#     temp_peak=1,
#     rng=None,
#     nsteps=10000,
#     da_mapping="reward",
#     **kwargs
# ):
    
#     rng = np.random.default_rng(rng)
#     state = 0
#     actions = []
#     temp_alpha = 1. / temp_tau
#     temperature = deepcopy(temperature_baseline)
#     temperatures = []
#     reward_history = []
    
#     def smooth(x):
#         return temp_alpha * temperature_baseline + (1 - temp_alpha) * x

#     for i in range(nsteps):

#         temperature = smooth(temperature)
#         # print(temperature)
#         ps = stable_softmax(q_table[state], temperature)
#         ps[np.isnan(ps)] = 0
#         # print(ps)
#         # try:
#         action = rng.choice(np.arange(len(ps)), p=ps)
#         # except ValueError:
#             # action = rng.choice(np.arange(len(ps)))

#         try:
#             reward = rng.choice(rewards[state, action])
#         except ValueError:
#             continue

#         if reward > temp_threshold1:
#             temperature = np.clip(temperature + temp_peak, 1e-2, 100)
#         elif reward < temp_threshold2:
#             temperature = np.clip(temperature - temp_peak, 1e-2, 100)

#         if np.isnan(q_table[state,action]):
#             continue
            
#         if da_mapping == "reward":
#             q_value_predict = q_table[state, action]
#             q_value_real = reward + gamma * np.nanmax(q_table[action])
#             q_table[state, action] += alpha * (q_value_real - q_value_predict)
#         elif da_mapping == "rpe":
#             q_table[state, action] += alpha * reward

#         state = action
#         actions.append(action)
#         temperatures.append(temperature)
#         reward_history.append(reward)

#     return q_table, np.array(actions), reward_history, temperatures

