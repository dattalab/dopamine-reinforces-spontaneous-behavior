import pandas as pd
import numpy as np
from sklearn import linear_model
from typing import Optional, Protocol
from rl_analysis.util import ScikitModel, ScikitSplit


def model_fit(
    model_features_z: pd.DataFrame,
    splits: ScikitSplit,
    return_models: bool = False,
    x: list[str] = ["signal_reref_dff_z_max"],
    partial_x: Optional[list[str]]  = None,
    y: str = "total_duration",
    shuffle_x: bool = False,
    shuffle_y: bool = False,
    return_input: bool = False,
	random_state: Optional[int] = None,
    clf: ScikitModel = linear_model.LinearRegression(),
):

	from sklearn.utils import shuffle
	from scipy.stats import pearsonr
	from copy import deepcopy

	use_x = model_features_z[x]
	use_y = model_features_z[y]
	nans = use_x.isnull() | use_y.isnull()
	
	use_x = use_x.loc[~nans]
	use_y = use_y.loc[~nans]
	
	if partial_x is not None:
		use_z = model_features_z[partial_x].loc[~nans]
		# any additional nans

		nans = use_z.isnull().any(axis=1)
		use_z = use_z.loc[~nans]
		use_x = use_x.loc[~nans]
		use_y = use_y.loc[~nans]

		partial_clf = linear_model.LinearRegression(fit_intercept=True)
		partial_clf.fit(use_z, use_x)
		use_x -= partial_clf.predict(use_z)

	preds = []
	vals = []
	models = []
	if shuffle_x:
		tmp_x = shuffle(use_x, random_state=random_state)
	else:
		tmp_x = use_x

	if shuffle_y:
		tmp_y = shuffle(use_y, random_state=random_state)
	else:
		tmp_y = use_y
	for train_idx, test_idx in splits.split(use_x, use_y):
		x_train, x_test = tmp_x.iloc[train_idx], tmp_x.iloc[test_idx]
		y_train, y_test = tmp_y.iloc[train_idx], tmp_y.iloc[test_idx]
		clf.fit(x_train.values.reshape(-1, 1), y_train.values.ravel())
		preds.append(clf.predict(x_test.values.reshape(-1, 1)).ravel())
		vals.append(y_test.values.ravel())
		if return_models:
			models.append(deepcopy(clf))

	preds = np.concatenate(preds).ravel()
	vals = np.concatenate(vals).ravel()

	if return_input and return_models:
		return pearsonr(preds, vals)[0], models, (tmp_x, tmp_y)
	elif return_models:
		return pearsonr(preds, vals)[0], models
	else:
		return pearsonr(preds, vals)[0]