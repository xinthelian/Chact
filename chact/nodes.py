import numpy as np
import pandas as pd
from IPython.display import display
from tabulate import tabulate
from random import shuffle
from random import random
from math import log
import os

from chact.utils import rownorm, safelog, safelog2, weighted_choice, most_likely_choice, no_objects


class chaiNode(object):
	"""
	Implementation.
	Assume all objects/referents/classifications have their name under the attribute "object".
	"""

	def __init__(self, costs=None):

		"""
		========
		These properties are used in this class,
		so migrate these properties from the original CobwebNode.
		"""
		# self.concept_id = self.gensym()
		self.count = 0.0
		# self.av_counts = {}
		self.children = []
		self.parent = None
		# self.tree = None
		# Update:
		self.instances = []

		# if otherNode:
		#     self.tree = otherNode.tree
		#     self.parent = otherNode.parent
		#     self.update_counts_from_node(otherNode)

		#     for child in otherNode.children:
		#         self.children.append(self.__class__(child))

		# self.cobweb_node = node
		# self.count = node.count
		# self.instances = node.instances
		# self.children = node.children

		self.costs = costs
		

	# def activate(self):
	# 	# This function should not be invoked until the whole tree finishes training.
	# 	# Activate the nodes that at the bottom first.

	# 	self.objects = self._get_objects()
	# 	self.utterances = self._get_utterances()

	# 	# Elements and parameters of the framework:
	# 	self.lexicon = self._build_lexicon()
	# 	self.alpha = node.depth() + 1
	# 	self.prior = self._find_prior()
	# 	if costs == None:
	# 		self.costs = [0.0] * len(self.utterances)
	# 	else:
	# 		if len(costs) != len(self.utterances):
	# 			raise ValueError("The number of costs given is not aligned with the number of utterances within the node.")
	# 		self.costs = costs

	# 	# Check if the node is a leaf for the tree:
	# 	if len(self.children) < 1:
	# 		self.leaf = True
	# 	else:
	# 		self.leaf = False

	# 	# P(phi), the relative probability wrt parent:
	# 	self.prob = self.count / self.parent.count


	# 	# Calculate the listener and speaker models:
	# 	if self.leaf:
	# 		self.speaker = self._speaker_leaf()
	# 		self.listener = self._listener_leaf()
	# 	else:
	# 		self.utility = self._utility()
	# 		self.speaker = self._speaker()
	# 		self.listener = self._listener()


	def _get_objects(self):
		"""
		Return the objects (referants, classifications) of the framework within the node.
		We assume that the object is reflected on the attribute 'object' of each instance.
		--------------
		Return a dictionary of possible objects and its frequency.
		"""
		# if not return_dict:
		# 	objects = set()
		# 	for inst in self.instances:
		# 		objects.update(inst['object'])
		# 	objects = list(objects)
		# else:

		objects = {}
		for inst in self.instances:
			if 'object' in inst:
				obj_name = inst['object']
				objects[obj_name] = objects.get(obj_name, 0) + 1
		return objects


	def _get_utterances(self, include_obj=False):
		"""
		Return the utterances (messages) of the framework within the node.
		---------------
		Return a dictionary of possible utterances and its frequency.
		"""
		# utterances = set()
		# for inst in self.instaces:
		# 	utterances.update(inst.values())
		# utterances = list(utterances)
		# if self.include_obj == False:
		# 	for _object in self.objects:
		# 		if _object in utterances:
		# 			utterances.remove(_object)
		# return utterances

		utterances = {}

		if not include_obj:
			instances = no_objects(self.instances)
		else:
			instances = self.instances

		for inst in instances:
			for utter in inst.values():
				utterances[utter] = utterances.get(utter, 0) + 1
		return utterances


	def _build_lexicon(self):
		"""
		Build the lexicon of the framework.
		Rows (indices): Utterances (messages)
		Columns: Objects (Referents, classifications)
		----------
		Returns DataFrame.
		"""
		lex = pd.DataFrame(0, index=self.utterances.keys(), columns=self.objects.keys())
		for inst in self.instances:
			for utter in self.utterances:
				if utter in inst.values():
					lex.at[utter, inst['object']] += 1
		return lex


	def _find_prior(self):
		"""
		Return the priors of all objects.
		----------
		Returns dict.
		"""
		priors = {}
		for obj_name in self.objects:
			priors[obj_name] = self.objects[obj_name] / self.count
		return priors


	# def _prob_children(self):
	# 	"""
	# 	Returns the array of probability of each children.
	# 	Calculated by # of instances of child / # of instances of the current node
	# 	"""
	# 	n_inst_children = []
	# 	for child in self.children:
	# 		n_inst_children.append(child.count)
	# 	prob_children = np.array(n_inst_children)
	# 	prob_children /= self.node.count
	# 	return prob_children


	def _speaker_leaf(self):
		"""
		If the node is a leaf of the tree, 
		we return the speaker values under the formula of the original RSA framework.
		"""
		literal_listener = rownorm(self.lexicon * self.prior.values())
		utilities = self.alpha * (safelog(literal_listener.T) + self.costs)
		return rownorm(np.exp(utilities))


	def _listener_leaf(self):
		"""
		If the node is a leaf of the tree, 
		we return the (pragmatic) listener values under the formula of the original RSA framework.
		"""
		return rownorm(self.speaker.T * self.prior)


	def _utility(self):
		utilities = pd.DataFrame(0, index=self.objects.keys(), columns=self.utterances.keys())
		# print(utilities)
		for child in self.children:
			utilities_child = safelog(child.listener.T) + child.costs
			for obj in child.objects:
				utilities.loc[obj] += child.prob * utilities_child.loc[obj]
		return self.alpha * utilities


	def _speaker(self):
		return rownorm(np.exp(self.utility))


	def _listener(self):
		listener = pd.DataFrame(0, index=self.utterances.keys(), columns=self.objects.keys())
		# print(listener)
		for child in self.children:
			listener_child = safelog(child.speaker.T)
			for utter in child.utterances:
				listener.loc[utter] += child.prob * listener_child.loc[utter]
		return rownorm(np.exp(self.alpha * listener))


	def _entropy_listener_object(self, df):
		"""
		Calculate the corresponding entropy of each object.

		Returns
        -------
        entropy_list: list(float), the entropy of each referent.
        entropy_sum: float, the sum of entropy of all referents.
		"""
		entropy_sum = 0.
		entropy_list = []
		for obj in df.columns[:-2]:
			entropy_obj = self.prior[obj] * (- np.sum(self.listener[obj] * safelog2(self.listener[obj])))
			entropy_sum += entropy_obj
			entropy_list.append(np.abs(entropy_obj))  # to discard the minus sign of -0.0

		# for i in range(1, self.lexicon.shape[1] + 1):
		# 	entropy_i = self.prior[i - 1] * (- np.sum(self.listener[self.listener.columns[i - 1]] * safelog2(self.listener[self.listener.columns[i - 1]])))
		# 	entropy_sum += entropy_i
		# 	entropy_list.append(np.abs(entropy_i))  # to discard the minus sign of -0.0 
		return entropy_list, entropy_sum


	def _entropy_listener_utterance(self, df):
		"""
		Calculate the corresponding entropy of each utterance.

		Returns
        -------
        entropy_list: list(float), the entropy of each message.
		"""
		entropy_list = []
		for utter in df.index:
			entropy_i = - np.sum(self.listener.loc[utter] * safelog2(self.listener.loc[utter]))
			entropy_list.append(np.abs(entropy_i))  # to discard the minus sign of -0.0 
		return entropy_list


	def _rational_speaker(self):
		return self.speaker.dropna(axis=0, how='all').dropna(axis=1, how='all').fillna(0)


	def _rational_listener(self):
		"""
		Display the pragmatic listener with including additional entropy, prior, and alpha information.

		Returns
        -------
        d: np.array or pd.DataFrame, depending on `self.lexicon`.
		"""
		d = self.listener.copy()

		# round each float:
		for row in d.index:
			d.loc[row] = [round(x, 3) for x in d.loc[row]]

		d = d.dropna(axis=1, how='all').dropna(axis=0, how='all').fillna(0)

		costs = []
		for utter in d.index:
			costs.append(self.costs[utter])
		d['costs'] = costs

		d['entropy'] = self._entropy_listener_utterance(d)
		d['entropy'] = [round(x, 3) for x in d['entropy']]

		prior = []
		# columns = list(d.columns)
		# columns.remove('costs')
		for obj in d.columns[:-2]:
			prior.append(self.prior[obj])
		d.loc['prior'] = [round(x, 3) for x in prior] + [''] + ['']

		d.loc['alpha'] = [self.alpha] + [' '] * (d.shape[1] - 1)
		e, e_s = self._entropy_listener_object(d)

		d.loc['entropy'] = [round(x, 3) for x in e] + [''] + ['']
		d.loc['entropy_sum'] = [round(e_s, 3)] + [' '] * (d.shape[1] - 1)

		return d


	def display_reference_game(self, return_d=False):
		"""
		Display the framework as a reference game format.
		"""
		d = self.lexicon.copy()
		d['costs'] = self.costs
		d.loc['prior'] = list(self.prior.values()) + ['']
		d.loc['alpha'] = [self.alpha] + [' '] * self.lexicon.shape[1]
		if return_d:
			return d
		else:
			display(d)


	def display(self):
		print("\nThe reference game is as follows:")
		self.display_reference_game()
		print("\nSpeaker:")
		print(tabulate(self.rational_speaker, headers='keys', tablefmt='psql', floatfmt='.3f'))
		print("\nListener:")
		print(tabulate(self.rational_listener, headers='keys', tablefmt='psql', floatfmt='.3f'))
		# d_listener = self.display_listener()
		# d_listener.to_csv('listener{}.csv'.format(self.concept_id), index=True)



class chactNode(chaiNode):

	"""
	A CobwebNode that integrates chaiNode.
	"""

	_counter = 0

	def __init__(self, otherNode=None):

		super().__init__()

		self.concept_id = self.gensym()
		# self.count = 0.0
		self.av_counts = {}
		# self.children = []
		# self.parent = None
		self.tree = None
		# Update:
		# self.instances = []

		if otherNode:
			self.tree = otherNode.tree
			self.parent = otherNode.parent
			self.update_counts_from_node(otherNode)

			for child in otherNode.children:
			    self.children.append(self.__class__(child))


	def shallow_copy(self):
		temp = self.__class__()
		temp.tree = self.tree
		temp.parent = self.parent
		temp.update_counts_from_node(self)
		return temp


	def attrs(self, attr_filter=None):
		if attr_filter is None:
		    return filter(lambda x: x[0] != "_", self.av_counts)
		elif attr_filter == 'all':
		    return self.av_counts
		else:
		    return filter(attr_filter, self.av_counts)


	def increment_counts(self, instance, instance_with_c=None):
		self.count += 1
		for attr in instance:
			if attr not in self.av_counts:
			    self.av_counts[attr] = {}
			if instance[attr] not in self.av_counts[attr]:
			    self.av_counts[attr][instance[attr]] = 0
			self.av_counts[attr][instance[attr]] += 1
		
		# Update:
		if instance_with_c:
		    self.instances.append(instance_with_c)
		else:
		    self.instances.append(instance)


	def update_counts_from_node(self, node):
		self.count += node.count
		for attr in node.attrs('all'):
		    if attr not in self.av_counts:
		        self.av_counts[attr] = {}
		    for val in node.av_counts[attr]:
		        if val not in self.av_counts[attr]:
		            self.av_counts[attr][val] = 0
		        self.av_counts[attr][val] += node.av_counts[attr][val]

		# Update:
		for instance in node.instances:
		    self.instances.append(instance)


	def expected_correct_guesses(self):
		correct_guesses = 0.0
		attr_count = 0
		for attr in self.attrs():
		    attr_count += 1
		    if attr in self.av_counts:
		        for val in self.av_counts[attr]:
		            prob = (self.av_counts[attr][val]) / self.count
		            correct_guesses += (prob * prob)
		return correct_guesses / attr_count

	def category_utility(self):
		if len(self.children) == 0:
		    return 0.0

		child_correct_guesses = 0.0

		for child in self.children:
		    p_of_child = child.count / self.count
		    child_correct_guesses += (p_of_child *
		                              child.expected_correct_guesses())

		return ((child_correct_guesses - self.expected_correct_guesses()) /
		        len(self.children))


	def get_best_operation(self, instance, best1, best2, best1_cu, possible_ops=['best', 'new', 'merge', 'split']):
		if not best1:
		    raise ValueError("Need at least one best child.")

		operations = []

		if "best" in possible_ops:
		    operations.append((best1_cu, random(), "best"))
		if "new" in possible_ops:
		    operations.append((self.cu_for_new_child(instance), random(),
		                       'new'))
		if "merge" in possible_ops and len(self.children) > 2 and best2:
		    operations.append((self.cu_for_merge(best1, best2, instance),
		                       random(), 'merge'))
		if "split" in possible_ops and len(best1.children) > 0:
		    operations.append((self.cu_for_split(best1), random(), 'split'))

		operations.sort(reverse=True)
		best_op = (operations[0][0], operations[0][2])
		return best_op


	def two_best_children(self, instance):
		if len(self.children) == 0:
		    raise Exception("No children!")

		children_relative_cu = [(self.relative_cu_for_insert(child, instance),
		                         child.count, random(), child) for child in
		                        self.children]
		children_relative_cu.sort(reverse=True)

		# Convert the relative CU's of the two best children into CU scores
		# that can be compared with the other operations.
		const = self.compute_relative_CU_const(instance)

		best1 = children_relative_cu[0][3]
		best1_relative_cu = children_relative_cu[0][0]
		best1_cu = (best1_relative_cu / (self.count+1) / len(self.children)
		            + const)

		best2 = None
		if len(children_relative_cu) > 1:
		    best2 = children_relative_cu[1][3]

		return best1_cu, best1, best2


	def compute_relative_CU_const(self, instance):
		temp = self.shallow_copy()
		temp.increment_counts(instance)
		ec_root_u = temp.expected_correct_guesses()

		const = 0
		for c in self.children:
		    const += ((c.count / (self.count + 1)) *
		              c.expected_correct_guesses())

		const -= ec_root_u
		const /= len(self.children)
		return const


	def relative_cu_for_insert(self, child, instance):
		temp = child.shallow_copy()
		temp.increment_counts(instance)
		return ((child.count + 1) * temp.expected_correct_guesses() -
		        child.count * child.expected_correct_guesses())


	def cu_for_insert(self, child, instance):
		temp = self.shallow_copy()
		temp.increment_counts(instance)

		for c in self.children:
		    temp_child = c.shallow_copy()
		    temp.children.append(temp_child)
		    temp_child.parent = temp
		    if c == child:
		        temp_child.increment_counts(instance)
		return temp.category_utility()


	def create_new_child(self, instance, instance_with_c=None):
		new_child = self.__class__()
		new_child.parent = self
		new_child.tree = self.tree
		# new_child.increment_counts(instance)

		# Update:
		new_child.increment_counts(instance, instance_with_c)

		self.children.append(new_child)
		return new_child


	def create_child_with_current_counts(self):
		if self.count > 0:
			new = self.__class__(self)
			new.parent = self
			new.tree = self.tree
			self.children.append(new)
			return new


	def cu_for_new_child(self, instance):
		temp = self.shallow_copy()
		for c in self.children:
		    temp.children.append(c.shallow_copy())

		# temp = self.shallow_copy()

		temp.increment_counts(instance)
		temp.create_new_child(instance)
		return temp.category_utility()


	def merge(self, best1, best2):
		new_child = self.__class__()
		new_child.parent = self
		new_child.tree = self.tree

		new_child.update_counts_from_node(best1)
		new_child.update_counts_from_node(best2)
		best1.parent = new_child
		# best1.tree = new_child.tree
		best2.parent = new_child
		# best2.tree = new_child.tree
		new_child.children.append(best1)
		new_child.children.append(best2)
		self.children.remove(best1)
		self.children.remove(best2)
		self.children.append(new_child)

		return new_child


	def cu_for_merge(self, best1, best2, instance):
		temp = self.shallow_copy()
		temp.increment_counts(instance)

		new_child = self.__class__()
		new_child.tree = self.tree
		new_child.parent = temp
		new_child.update_counts_from_node(best1)
		new_child.update_counts_from_node(best2)
		new_child.increment_counts(instance)
		temp.children.append(new_child)

		for c in self.children:
		    if c == best1 or c == best2:
		        continue
		    temp_child = c.shallow_copy()
		    temp.children.append(temp_child)

		return temp.category_utility()


	def split(self, best):
		self.children.remove(best)
		for child in best.children:
		    child.parent = self
		    child.tree = self.tree
		    self.children.append(child)


	def cu_for_fringe_split(self, instance):
		temp = self.shallow_copy()

		temp.create_child_with_current_counts()
		temp.increment_counts(instance)
		temp.create_new_child(instance)

		return temp.category_utility()


	def cu_for_split(self, best):
		temp = self.shallow_copy()

		for c in self.children + best.children:
		    if c == best:
		        continue
		    temp_child = c.shallow_copy()
		    temp.children.append(temp_child)

		return temp.category_utility()


	def is_exact_match(self, instance):
		for attr in set(instance).union(set(self.attrs())):
		    if attr[0] == '_':
		        continue
		    if attr in instance and attr not in self.av_counts:
		        return False
		    if attr in self.av_counts and attr not in instance:
		        return False
		    if attr in self.av_counts and attr in instance:
		        if instance[attr] not in self.av_counts[attr]:
		            return False
		        if not self.av_counts[attr][instance[attr]] == self.count:
		            return False
		return True


	def __hash__(self):
		# Update:
		return hash("chactNode" + str(self.concept_id))


	def gensym(self):
		self.__class__._counter += 1
		return self.__class__._counter


	def __str__(self):
		return self.pretty_print()


	def pretty_print(self, depth=0):
		ret = str(('\t' * depth) + "|-" + str(self.av_counts) + ":" +
		          str(self.count) + '\n')

		for c in self.children:
		    ret += c.pretty_print(depth+1)

		return ret


	def depth(self):
		if self.parent:
		    return 1 + self.parent.depth()
		return 0


	def level(self):
		return int(self.depth() + 1)


	def is_parent(self, other_concept):
		temp = other_concept
		while temp is not None:
		    if temp == self:
		        return True
		    try:
		        temp = temp.parent
		    except Exception:
		        print(temp)
		        assert False
		return False


	def num_concepts(self):
		children_count = 0
		for c in self.children:
		    children_count += c.num_concepts()
		return 1 + children_count


	def output_json(self):
		output = {}
		output['name'] = "Concept" + str(self.concept_id)
		output['size'] = self.count
		output['children'] = []

		temp = {}
		for attr in self.attrs('all'):
		    for value in self.av_counts[attr]:
		        temp[str(attr)] = {str(value): self.av_counts[attr][value] for
		                           value in self.av_counts[attr]}
		        # temp[attr + " = " + str(value)] = self.av_counts[attr][value]

		for child in self.children:
		    output["children"].append(child.output_json())

		output['counts'] = temp

		return output


	def get_weighted_values(self, attr, allow_none=True):
		choices = []
		if attr not in self.av_counts:
		    choices.append((None, 1.0))
		    return choices

		val_count = 0
		for val in self.av_counts[attr]:
		    count = self.av_counts[attr][val]
		    choices.append((val, count / self.count))
		    val_count += count

		if allow_none:
		    choices.append((None, ((self.count - val_count) / self.count)))

		return choices


	def predict(self, attr, choice_fn="most likely", allow_none=True):
		if choice_fn == "most likely" or choice_fn == "m":
		    choose = most_likely_choice
		elif choice_fn == "sampled" or choice_fn == "s":
		    choose = weighted_choice
		else:
		    raise Exception("Unknown choice_fn")

		if attr not in self.av_counts:
		    return None

		choices = self.get_weighted_values(attr, allow_none)
		val = choose(choices)
		return val


	def probability(self, attr, val):
		if val is None:
		    c = 0.0
		    if attr in self.av_counts:
		        c = sum([self.av_counts[attr][v] for v in
		                 self.av_counts[attr]])
		    return (self.count - c) / self.count

		if attr in self.av_counts and val in self.av_counts[attr]:
		    return self.av_counts[attr][val] / self.count

		return 0.0


	def log_likelihood(self, child_leaf):
		ll = 0

		for attr in set(self.attrs()).union(set(child_leaf.attrs())):
		    vals = set([None])
		    if attr in self.av_counts:
		        vals.update(self.av_counts[attr])
		    if attr in child_leaf.av_counts:
		        vals.update(child_leaf.av_counts[attr])

		    for val in vals:
		        op = child_leaf.probability(attr, val)
		        if op > 0:
		            p = self.probability(attr, val) * op
		            if p >= 0:
		                ll += log(p)
		            else:
		                raise Exception("Should always be greater than 0")

		return ll


	# ======= UPDATE FOR CHAI =======

	def chai_activate(self, include_obj, rsa=False):
		# This function should not be invoked until the whole tree finishes training.
		# Activate the nodes that at the bottom first.
		# The core functionality of CHAI within the node.

		self.objects = self._get_objects()
		self.utterances = self._get_utterances(include_obj=include_obj)
		# print(self.objects, self.utterances)

		# Elements and parameters of the framework:
		self.lexicon = self._build_lexicon()
		self.alpha = self.level()
		self.prior = self._find_prior()
		if self.costs == None:
			self.costs = {}
			for utter in self.utterances:
				self.costs[utter] = 0.0
			# self.costs = [0.0] * len(self.utterances)
		else:
			if len(costs) != len(self.utterances):
				raise ValueError("The number of costs given is not aligned with the number of utterances within the node.")
			self.costs = costs

		if not rsa:
			# Activate CHAI framework.
			# Check if the node is a leaf for the tree:
			if len(self.children) < 1:
				self.leaf = True
			else:
				self.leaf = False

			# P(phi), the relative probability wrt parent:
			if self.parent is None:
				self.prob = 1.  # the root
			else:
				self.prob = self.count / self.parent.count


			# Calculate the listener and speaker models:
			if self.leaf:
				self.speaker = self._speaker_leaf()
				self.listener = self._listener_leaf()
			else:
				self.utility = self._utility()
				self.speaker = self._speaker()
				self.listener = self._listener()
		else:
			# Activate RSA Framework instead
			self.speaker = self._speaker_leaf()
			self.listener = self._listener_leaf()

		self.rational_speaker = self._rational_speaker()
		self.rational_listener = self._rational_listener()


	def chai_display(self):
		print('\n\n' + ' chactNode {} '.format(self.concept_id).center(70, '*'))
		print("Level:", self.level())
		if len(self.children) > 0:
			id_children = []
			for child in self.children:
				id_children.append(child.concept_id)
			print("Parent of nodes", id_children)
		if self.parent:
			print("Child of node", self.parent.concept_id)

		self.display()


	def save_outputs(self, verbose=True, display=True):
		if verbose:
			print('chactNode {}'.format(self.concept_id))

		folder = "chact-outputs"
		if not os.path.exists(folder):
			os.makedirs(folder)

		lexicon_file = os.path.join(folder, "lexicon-{}-{}.csv".format(self.level(), self.concept_id))
		speaker_file = os.path.join(folder, "speaker-{}-{}.csv".format(self.level(), self.concept_id))
		listener_file = os.path.join(folder, "listener-{}-{}.csv".format(self.level(), self.concept_id))
		
		if display:
			self.display_reference_game(return_d=True).to_csv(lexicon_file, index=True)
			self.rational_speaker.to_csv(speaker_file, index=True)
			self.rational_listener.to_csv(listener_file, index=True)
		else:
			self.lexicon.to_csv(lexicon_file, index=True)
			self.speaker.to_csv(speaker_file, index=True)
			self.listener.to_csv(listener_file, index=True)


	def _trail_iter(self, obj):
		# Used for trail function in chactTree
		if obj in self.rational_listener.columns:
			listener = self.rational_listener.iloc[:-4, :-2]
			# print(listener)
			utter = pd.to_numeric(listener[obj]).idxmax()
			prob = listener.loc[utter, obj]
			print("--> Level {}, chactNode {}, utterance {}, probability {}.".format(self.level(), self.concept_id, utter, prob))
			if prob == 1.:
				if self.count == 1:
					print("    Note that, this is the only object for this node.")
		else:
			print("--> Level {}, chactNode {}, no appropriate utterance for the object.".format(self.level(), self.concept_id))
		
		if len(self.children) < 1:
			print("Reach the leaf. End of trail.")
		else:
			for child in self.children:
				if obj in child.objects:
					child._trail_iter(obj)


















	