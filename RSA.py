import numpy as np
import pandas as pd
from IPython.display import display
from tabulate import tabulate


def rownorm(mat):
	"""
	Normalization in rows.
	"""
	return (mat.T / mat.sum(axis=1)).T

def safelog(vals):
	"""
	Slience distracting warning about log(0).
	"""
	with np.errstate(divide='ignore'):
		# return np.log(vals)
		return np.log(vals.astype('float64'))

def safelog2(vals):
	"""
	Slience distracting warning about log2(0).
	"""
	with np.errstate(divide='ignore'):
		# return np.log(vals)
		return np.log2(vals.astype('float64'))


class RSA:
	"""
	Implementation of the core Rational Speech Acts model.
	Original version of code from Ling 130a/230a: Introduction to semantics and pragmatics, Winter 2023
	http://web.stanford.edu/class/linguist130a/

    Parameters
    ----------
    lexicon : `np.array` or `pd.DataFrame`
        Messages along the rows, states along the columns.
    prior : array-like
        Prior probs of referents. Same length as the number of colums in `lexicon`.
    costs : array-like
        Costs of messages. Same length as the number of rows in `lexicon`.
        If one message should be lessened, the cost should be negative.
    alpha : float
        The temperature parameter. Default: 1.0
    """

	def __init__(self, lexicon, prior=None, costs=None, alpha=1.0):

		self.lexicon = lexicon

		# Predefine prior, costs.
		if prior == None:
			prior = [1.0 / lexicon.shape[1]] * lexicon.shape[1]
		if costs == None:
			costs = [0.0] * lexicon.shape[0]

		self.prior = np.array(prior)
		self.costs = np.array(costs)
		# print(self.prior, self.costs)
		self.alpha = alpha

	def literal_listener(self):
		"""
		Literal listener predictions,
		which corresponds intuitively to truth conditions with priors.

		Returns
        -------
        np.array or pd.DataFrame, depending on `self.lexicon`.
        The rows correspond to messages, the columns to states.
		"""
		return rownorm(self.lexicon * self.prior)

	def speaker(self):
		"""
		Returns a matrix of pragmatic speaker predictions.

        Returns
        -------
        np.array or pd.DataFrame, depending on `self.lexicon`.
        The rows correspond to states, the columns to states.
        """
		lit = self.literal_listener().T
		utilities = self.alpha * (safelog(lit) + self.costs)
		return rownorm(np.exp(utilities))

	def listener(self):
		"""
        Returns a matrix of pragmatic listener predictions.

        Returns
        -------
        np.array or pd.DataFrame, depending on `self.lexicon`.
        The rows correspond to messages, the columns to states.
        """
		spk = self.speaker().T
		return rownorm(spk * self.prior)

	def display_reference_game(self):
		"""
		Display the RSA framework as a reference game format.
		"""
		d = self.lexicon.copy()
		d['costs'] = self.costs
		d.loc['prior'] = list(self.prior) + ['']
		d.loc['alpha'] = [self.alpha] + [' '] * self.lexicon.shape[1]
		display(d)

	def _entropy_listener_r(self):
		"""
		Calculate the corresponding entropy of each referent.

		Returns
        -------
        entropy_list: list(float), the entropy of each referent.
        entropy_sum: float, the sum of entropy of all referents.
		"""
		entropy_sum = 0.
		entropy_list = []
		for i in range(1, self.lexicon.shape[1] + 1):
			entropy_i = self.prior[i - 1] * (- np.sum(self.listener()[self.listener().columns[i - 1]] * safelog2(self.listener()[self.listener().columns[i - 1]])))
			entropy_sum += entropy_i
			entropy_list.append(np.abs(entropy_i))  # to discard the minus sign of -0.0 
		return entropy_list, entropy_sum

	def _entropy_listener_m(self):
		"""
		Calculate the corresponding entropy of each message.

		Returns
        -------
        entropy_list: list(float), the entropy of each message.
		"""
		entropy_list = []
		for msg in self.lexicon.index:
			entropy_i = - np.sum(self.listener().loc[msg] * safelog2(self.listener().loc[msg]))
			entropy_list.append(np.abs(entropy_i))  # to discard the minus sign of -0.0 
		return entropy_list

	def display_pragmatic_listener(self):
		"""
		Display the pragmatic listener with including additional entropy, prior, and alpha information.

		Returns
        -------
        d: np.array or pd.DataFrame, depending on `self.lexicon`.
		"""
		d = self.listener()

		# round each float:
		for row in d.index:
			d.loc[row] = [round(x, 3) for x in d.loc[row]]

		d['costs'] = self.costs

		d['entropy'] = self._entropy_listener_m()
		d['entropy'] = [round(x, 3) for x in d['entropy']]

		d.loc['prior'] = [round(x, 3) for x in self.prior] + [''] + ['']
		d.loc['alpha'] = [self.alpha] + [' '] * (self.lexicon.shape[1] + 1)
		e, e_s = self._entropy_listener_r()

		d.loc['entropy'] = [round(x, 3) for x in e] + [''] + ['']
		d.loc['entropy_sum'] = [round(e_s, 3)] + [' '] * (self.lexicon.shape[1] + 1)

		return d

	def display(self):
		"""
		Display the whole RSA framework.
		including its reference game format, literal listener, pragmatic speaker, and pragmatic listener.
		"""
		print("\nThe reference game is as follows:")
		self.display_reference_game()
		print("\nLiteral Listener:")
		print(tabulate(self.literal_listener(), headers='keys', tablefmt='psql', floatfmt='.3f'))
		print("\nPragmatic Speaker:")
		print(tabulate(self.speaker(), headers='keys', tablefmt='psql', floatfmt='.3f'))
		print("\nPragmatic Listener:")
		print(tabulate(self.display_pragmatic_listener(), headers='keys', tablefmt='psql', floatfmt='.3f'))
		