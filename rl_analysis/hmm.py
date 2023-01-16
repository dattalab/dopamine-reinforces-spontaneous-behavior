import numpy as np
import joblib
from scipy.special import logsumexp, gammaln
from tqdm.auto import tqdm


def get_obs_parameters(arhmm):
	from pybasicbayes.util.general import inv_psd
	sigs = [inv_psd(o.sigma, return_chol=True) for o in arhmm.obs_distns]
	obs_parameters = {}
	obs_parameters["As"] = np.stack([o.A[:, :-1] for o in arhmm.obs_distns], axis=0)
	obs_parameters["bs"] = np.stack([o.A[:, -1] for o in arhmm.obs_distns], axis=0)
	obs_parameters["D"] = arhmm.obs_distns[0].D_out
	obs_parameters["nus"] = np.array([o.nu for o in arhmm.obs_distns])
	obs_parameters["sigma_invs"] = np.stack([_[0] for _ in sigs], axis=0)
	obs_parameters["Ls"] = [_[1] for _ in sigs]
	return obs_parameters


def load_pcs_and_online_hmm(data_file, arhmm_path, obs_parameters, lookahead=0):
	arhmm = joblib.load(arhmm_path)["model"]
	obs_parameters = get_obs_parameters(arhmm)
	try:
		with h5py.File(data_file, "r") as f:
			pcs = f["/metadata/misc/realtime_pc_scores/data"][()]
		online_estimate = online_hmm(
			pcs, arhmm=arhmm, obs_parameters=obs_parameters, lookahead=lookahead
		)
		true_estimate = arhmm.heldout_viterbi(pcs)
	except KeyError:
		online_estimate = None
		true_estimate = None

	return online_estimate, true_estimate


def log_likelihood_vec(xy, As, bs, sigma_invs, Ls, nus, D=10):
	assert isinstance(xy, (tuple, np.ndarray))
	x, y = (xy[:, :-D], xy[:, -D:]) if isinstance(xy, np.ndarray) else xy

	x = np.atleast_2d(x)
	pdt = np.matmul(As, x.T) + bs[..., None]
	r = np.tile(y.T, (len(pdt), 1, 1)) - pdt
	z = np.matmul(sigma_invs, r)

	out = []
	for nu, _r, _z, L in zip(nus, r, z, Ls):
		out.append(-0.5 * (nu + D) * np.log(1.0 + (_r * _z).sum(0) / nu))
		out[-1] += (
			gammaln((nu + D) / 2.0)
			- gammaln(nu / 2.0)
			- D / 2.0 * np.log(nu)
			- D / 2.0 * np.log(np.pi)
			- np.log(np.diag(L)).sum()
		)

	return np.array(out).squeeze()


def forward_pass(curr_lp, ll, log_trans_mat):
	tmp = ll + logsumexp(curr_lp[..., None] + log_trans_mat, axis=0)
	return tmp - logsumexp(tmp)


def backward_pass(ll, beta, log_trans_mat):
	return logsumexp(log_trans_mat + beta + ll[None, ...], axis=1)


def online_hmm(data, arhmm, obs_parameters, lookahead=0):

	# initial state estimate
	alphas = [np.log(arhmm.init_state_distn.pi_0)]
	smoothed_states = []
	nlags = arhmm.nlags
	strider = arhmm.obs_distns[0]._ensure_strided
	log_trans_mat = np.log(arhmm.trans_distn.trans_matrix)

	for i in tqdm(range(nlags, len(data) - lookahead)):
		# collect data, get data ll
		strided_data = strider(data[i - (nlags) : i + 1])
		ll = log_likelihood_vec(strided_data, **obs_parameters)

		# compute forward pass
		alphas.append(forward_pass(alphas[-1], ll, log_trans_mat))

		# compute backward pass and smooth
		beta = 0
		for j in range(i + lookahead, i, -1):
			strided_data = strider(data[j - (nlags + 1) : j])
			ll = log_likelihood_vec(strided_data, **obs_parameters)
			beta = backward_pass(ll, beta, log_trans_mat)
			beta = beta - logsumexp(beta)

		#     beta = backward_pass(ll, log_trans_mat)
		tmp = alphas[-1] + beta
		smoothed_states.append(tmp - logsumexp(tmp))

	return np.array(smoothed_states)


def threshold_and_debounce_likes(probs, threshold=.8, debounce=3, censor=5):
	count = 0
	count_off = 0
	flip_flop = True
	time_since = np.inf

	feedback = []

	for p in np.exp(probs):
		# assuming that all experiments we did were a gt comparison
		if p > threshold:
			count += 1
			count_off = 0
		else:
			count = 0
			count_off += 1
		if (count_off > debounce) and not flip_flop:
			flip_flop = True

		if (count > debounce) and (time_since > censor) and flip_flop:
			feedback.append(1)
			time_since = 0
			count = 0
			count_off = 0
			flip_flop = False
		else:
			time_since += 1
			feedback.append(-1)
	return feedback