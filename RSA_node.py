import pandas as pd
import numpy as np

from RSA import RSA


def no_classification(dicts):
	"""
	Ignore the 'classification' attribute of instances.
	"""
	return [{a: d[a] for a in d if a != 'classification'} for d in dicts]


class RSA_CobwebNode_refer:
	"""
	Construct an RSA framework of the instances included in some CobwebNode.
	Note that, this module is used for mushroom examples with referring game purposes ONLY.

	Parameters
	----------
	node: CobwebNode
	include_classification: Boolean.
	If False, the classification of the data will NOT be included as messages of the framework.
	level: float.
	The depth of the node in its corresponding tree.
	"""

	def __init__(self, node, state_name='name', include_name=False, level=1.0):
		self.instances = node.instances
		self.count = len(node.instances)
		self.include_name = include_name

		# Generate set of states (names) within the node:
		self.states = []
		for instance in self.instances:
			self.states.append(instance[state_name])

		self.msgs = self._get_msgs()
		self.lexicon = self._build_lexicon()
		# print(self.lexicon)

		self.alpha = level
		self.RSA = self._build_RSA()

	def _get_msgs(self):
		"""
		Returns the messages of the framework.
		----------
		Returns list(str).
		"""
		msgs = set()
		for instance in self.instances:
			msgs.update(instance.values())
		msgs = list(msgs)
		if self.include_name == False:
			for _state in self.states:
				if _state in msgs:
					msgs.remove(_state)
		return msgs

	def _build_lexicon(self):
		"""
		Build the lexicon of the framework.
		Rows (Indices): Messages
		Columns: Referents (Classifications)
		----------
		Returns DataFrame.
		"""
		msgs = self._get_msgs()
		lex = pd.DataFrame(0, index=msgs, columns=self.states)
		for mushroom in self.instances:
			for msg in msgs:
				if msg in mushroom.values():
					lex.at[msg, mushroom['name']] += 1
		return lex

	def _build_RSA(self):
		"""
		Build the RSA framework of the node.
		The prior is a uniform distribution among all the mushrooms.
		The alpha is determined by the depth of the node.
		We assume as the node becomes deeper, pragmatic level of the node is higher.
		Future potential work: adjust cost of messages.
		----------
		Returns DataFrame.
		"""
		lexicon = self.lexicon
		prior = [1 / self.count] * self.count  # Uniform distribution
		rsa_framework = RSA(lexicon=lexicon, prior=prior, alpha=self.alpha)
		return rsa_framework

	def display_RSA(self):
		"""
		Display the RSA framework.
		"""
		self.RSA.display()



class RSA_CobwebNode_classification:
	"""
	Construct an RSA framework of the instances included in some CobwebNode.
	Note that, this module is used for mushroom examples with classification (edible, poisonous) purposes ONLY.

	Parameters
	----------
	node: CobwebNode
	include_classification: Boolean.
	If False, the classification of the data will NOT be included as messages of the framework.
	level: float.
	The depth of the node in its corresponding tree.
	"""

	def __init__(self, node, include_classification=False, level=1.0):
		self.instances = node.instances
		self.count = len(node.instances)
		self.include_classification = include_classification
		self.msgs = self._get_msgs()
		self.states = ['edible', 'poisonous']

		self.lexicon = self._build_lexicon()
		self.alpha = level
		self.RSA = self._build_RSA()

	def _get_msgs(self):
		"""
		Returns the messages of the framework.
		----------
		Returns list(str).
		"""
		msgs = set()
		for instance in self.instances:
			msgs.update(instance.values())
		msgs = list(msgs)
		if self.include_classification == False:
			for _class in ['edible', 'poisonous']:
				if _class in msgs:
					msgs.remove(_class)
		return msgs

	def _build_lexicon(self):
		"""
		Build the lexicon of the framework.
		Rows (Indices): Messages
		Columns: Referents (Classifications)
		----------
		Returns DataFrame.
		"""
		mushrooms_e = [mushroom for mushroom in self.instances if mushroom['classification'] == 'edible']
		mushrooms_e_no_c = no_classification(mushrooms_e)
		mushrooms_p = [mushroom for mushroom in self.instances if mushroom['classification'] == 'poisonous']
		mushrooms_p_no_c = no_classification(mushrooms_p)
		self.n_edible = len(mushrooms_e)
		self.n_poisonous = len(mushrooms_p)

		msgs = self._get_msgs()
		lex = pd.DataFrame(0, index=msgs, columns=['edible', 'poisonous'])
		for mushroom in mushrooms_e_no_c:
			for msg in msgs:
				if msg in mushroom.values():
					lex.at[msg, 'edible'] += 1
		for mushroom in mushrooms_p_no_c:
			for msg in msgs:
				if msg in mushroom.values():
					lex.at[msg, 'poisonous'] += 1
		return lex

	def _build_RSA(self):
		"""
		Build the RSA framework of the node.
		The prior is determined by the portion of edible (or poisonous) mushrooms within the node.
		The alpha is determined by the depth of the node.
		We assume as the node becomes deeper, pragmatic level of the node is higher.
		Future potential work: adjust cost of messages.
		----------
		Returns DataFrame.
		"""
		lexicon = self.lexicon
		prior = [self.n_edible / self.count, self.n_poisonous / self.count]
		rsa_framework = RSA(lexicon=lexicon, prior=prior, alpha=self.alpha)
		return rsa_framework

	def display_RSA(self):
		"""
		Display the RSA framework.
		"""
		self.RSA.display()


# class RSA_node(RSA_CobwebNode_refer, RSA_CobwebNode_classification):

# 	def __init__(self, node, task='refer'):
# 		if task in ('refer', 'refering', 'referring', 'refer game', 'referring game', 'refering game'):
# 			RSA_CobwebNode_refer.__init__(self, node)
# 		else:
# 			RSA_CobwebNode_classification.__init__(self, node)
