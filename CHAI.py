import numpy as np
import pandas as pd
from IPython.display import display
from tabulate import tabulate

from RSA import RSA
from cobweb.cobweb import CobwebTree, CobwebNode


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

def traverse_bottom_to_top(node, sequence):
	if node is None:
		return
	for child in node.children:
		traverse_bottom_to_top(child, sequence)
	sequence.append(node)



class ChaiTree(object):
	"""
	Implementation of the simplified CHAI theory of one node in the model.
	ASSUME the CobwebTree inheritted has been done training. (CAN BE AN INVALID ASSUMPTION)

	I think this class is about how to build the tree from the bottom to the top.
	Maybe the process should be as follows:
	- After fitting all the instances, return a list of leaves of the tree. 
		(By how? Note that when we displaying frameworks for each node before, we are doing a DFS.)
	- Start with leaves. Construct CHAI frameworks. Then turn to their parents.
	  One way to traverse a tree from bottom to top (left leaves -> right leaves, then start from the deepest parents).
	  See traverse_from_bottom_to_top.ipynb in Google Colab

	Also a mechanism for defining the parameters for each node is required.
	- alpha = depth + 1
	- cost? Is there a mechanism? Should it defined in the universe level or wrt nodes?
	Parameters
	----------
	node: `CobwebNode`
	lexicon: `np.array` or `pd.DataFrame`
        Messages along the rows, states along the columns.
    prior: array-like
    	Prior probs of referents. Same length as the number of rows iin `lexicon`.
    costs: array-like
    	Costs of messages. Same length as the number of columns in `lexicon`.
    	If one message should be lessened, the cost should be negative.
    alpha: float
    	The temperature parameter. Default: 1.0
	"""

	def __init__(self, task='refer', prior=None, costs=None):

		# self.cobweb_tree = tree
		# self.lexicon = lexicon
		# self.root = tree.root

		# self.children = node.children
		# self.task = task  # 'refer' or 'class'

		# if len(self.children) < 1:

		# if self.node.count < 1:
		# 	raise ValueError("There is no instance for this node.")
		# self.probs_children = self._prob_children()

		# if len(self.children) < 1:
		# 	if task in ('refer', 'refering', 'referring', 'refer game', 'referring game', 'refering game'):

		# 首先，fit好一个tree（可能不能inherit，只能把所有CobwebTree的东西复制过来）
		# 当然fit的node都是ChaiNode.
		# 这个时候没有建立ChaiTree。
		# 然后再这个module里面设置一个function用于建立ChaiTree, from bottom to top.每个遍历过的node进行activate

		def __init__(self, reserve_classification=True):
        """
        The tree constructor.
        """
        self.root = ChaiNode()
        self.root.tree = self
        self.reserve_classification = reserve_classification

    def clear(self):
        """
        Clears the concepts of the tree.
        """
        self.root = ChaiNode()
        self.root.tree = self

    def __str__(self):
        return str(self.root)

    def _sanity_check_instance(self, instance):
        for attr in instance:
            try:
                hash(attr)
                attr[0]
            except Exception:
                raise ValueError('Invalid attribute: '+str(attr) +
                                 ' of type: '+str(type(attr)) +
                                 ' in instance: '+str(instance) +
                                 ',\n'+type(self).__name__ +
                                 ' only works with hashable ' +
                                 'and subscriptable attributes' +
                                 ' (e.g., strings).')
            try:
                hash(instance[attr])
            except Exception:
                raise ValueError('Invalid value: '+str(instance[attr]) +
                                 ' of type: '+str(type(instance[attr])) +
                                 ' in instance: '+str(instance) +
                                 ',\n'+type(self).__name__ +
                                 ' only works with hashable values.')
            if instance[attr] is None:
                raise ValueError("Attributes with value None should"
                                 " be manually removed.")

    def ifit(self, instance):
        self._sanity_check_instance(instance)
        return self.cobweb(instance)


    def fit(self, instances, iterations=1, randomize_first=True):
        instances = [i for i in instances]

        for x in range(iterations):
            if x == 0 and randomize_first:
                shuffle(instances)
            for i in instances:
                self.ifit(i)
            shuffle(instances)

    def cobweb(self, instance):
        current = self.root

        # Update:
        if not self.reserve_classification:
            instance_with_c = instance.copy()
            instance = {key: value for key, value in instance.items() if key != 'classification'}

        while current:
            # the current.count == 0 here is for the initially empty tree.
            if not current.children and (current.is_exact_match(instance) or
                                         current.count == 0):
                # print("leaf match")

                # current.increment_counts(instance)
                # Update:
                current.increment_counts(instance, instance_with_c)

                break

            elif not current.children:
                # print("fringe split")
                new = current.__class__(current)
                current.parent = new
                new.children.append(current)

                if new.parent:
                    new.parent.children.remove(current)
                    new.parent.children.append(new)
                else:
                    self.root = new

                # new.increment_counts(instance)
                # Update:
                new.increment_counts(instance, instance_with_c)

                # current = new.create_new_child(instance)
                # Update:
                current = new.create_new_child(instance, instance_with_c)

                break

            else:
                best1_cu, best1, best2 = current.two_best_children(instance)
                _, best_action = current.get_best_operation(instance, best1,
                                                            best2, best1_cu)

                # print(best_action)
                if best_action == 'best':
                    # current.increment_counts(instance)
                    # Update:
                    current.increment_counts(instance, instance_with_c)

                    current = best1
                elif best_action == 'new':
                    # current.increment_counts(instance)
                    # Update:
                    current.increment_counts(instance, instance_with_c)

                    # current = current.create_new_child(instance)
                    # Update:
                    current = current.create_new_child(instance, instance_with_c)
                    break
                elif best_action == 'merge':
                    # current.increment_counts(instance)
                    # Update:
                    current.increment_counts(instance, instance_with_c)

                    new_child = current.merge(best1, best2)
                    current = new_child
                elif best_action == 'split':
                    current.split(best1)
                else:
                    raise Exception('Best action choice "' + best_action +
                                    '" not a recognized option. This should be'
                                    ' impossible...')

        return current

    def _cobweb_categorize(self, instance):
        current = self.root
        while current:
            if not current.children:
                return current

            _, best1, best2 = current.two_best_children(instance)
            current = best1

    def infer_missing(self, instance, choice_fn="most likely",
                      allow_none=True):
        self._sanity_check_instance(instance)
        temp_instance = {a: instance[a] for a in instance}
        concept = self._cobweb_categorize(temp_instance)

        for attr in concept.attrs('all'):
            if attr in temp_instance:
                continue
            val = concept.predict(attr, choice_fn, allow_none)
            if val is not None:
                temp_instance[attr] = val

        return temp_instance

    def categorize(self, instance):
        self._sanity_check_instance(instance)
        return self._cobweb_categorize(instance)
    	

    def chai(self):
    	"""
    	The core functionality of CHAI model.
    	"""
    	sequence = []
    	traverse_bottom_to_top(self.root, sequence)
    	for node in sequence:
    		node.activate()

    	


class ChaiNode(object):
	"""
	Implementation.
	Assume all objects/referents/classifications have their name under the attribute "object".
	"""

	def __init__(self, costs=None):

		"""
		========
		initialization of the node as a CobwebNode:
		"""
		self.concept_id = self.gensym()
		self.count = 0.0
		self.av_counts = {}
		self.children = []
		self.parent = None
		self.tree = None
		# Update:
		self.instances = []

		if otherNode:
		    self.tree = otherNode.tree
		    self.parent = otherNode.parent
		    self.update_counts_from_node(otherNode)

		    for child in otherNode.children:
		        self.children.append(self.__class__(child))

		# self.cobweb_node = node
		# self.count = node.count
		# self.instances = node.instances
		# self.children = node.children
		
	"""
	=========
	The following is about the CHAI model, ChaiNode.
	"""

	def activate(self):
		# This function should not be invoked until the whole tree finishes training.
		# Activate the nodes that at the bottom first.

		self.objects = self._get_objects()
		self.utterances = self._get_utterances()

		# Elements and parameters of the framework:
		self.lexicon = self._build_lexicon()
		self.alpha = node.depth() + 1
		self.prior = self._find_prior()
		if costs == None:
			self.costs = [0.0] * len(self.utterances)
		else:
			if len(costs) != len(self.utterances):
				raise ValueError("The number of costs given is not aligned with the number of utterances within the node.")
			self.costs = costs

		# Check if the node is a leaf for the tree:
		if len(self.children) < 1:
			self.leaf = True
		else:
			self.leaf = False

		# P(phi), the relative probability wrt parent:
		self.prob = self.count / self.parent.count


		# Calculate the listener and speaker models:
		if self.leaf:
			self.speaker = self._speaker_leaf()
			self.listener = self._listener_leaf()
		else:
			self.utility = self._utility()
			self.speaker = self._speaker()
			self.listener = self._listener()


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


	def _get_utterances(self):
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
		for inst in self.instances:
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
		utilities = pd.DataFrame(0, index=self.utterances.keys(), columns=self.objects.keys())
		for child in self.children:
			utilities_child = safelog(child.listener.T) + child.costs
			for obj in child.objects:
				utilities.loc[obj] += child.prob * utilities_child.loc[obj]
		return self.alpha * utilities


	def _speaker(self):
		return rownorm(np.exp(self.utility))


	def _listener(self):
		listener = pd.DataFrame(0, index=self.utterances.keys(), columns=self.objects.keys())
		for child in self.children:
			listener_child = safelog(child.speaker.T)
			for utter in child.utterances:
				listener.loc[utter] += child.prob * listener_child.loc[utter]
		return rownorm(np.exp(self.alpha * listener))


	def _entropy_listener_object(self):
		"""
		Calculate the corresponding entropy of each object.

		Returns
        -------
        entropy_list: list(float), the entropy of each referent.
        entropy_sum: float, the sum of entropy of all referents.
		"""
		entropy_sum = 0.
		entropy_list = []
		for i in range(1, self.lexicon.shape[1] + 1):
			entropy_i = self.prior[i - 1] * (- np.sum(self.listener[self.listener.columns[i - 1]] * safelog2(self.listener[self.listener.columns[i - 1]])))
			entropy_sum += entropy_i
			entropy_list.append(np.abs(entropy_i))  # to discard the minus sign of -0.0 
		return entropy_list, entropy_sum


	def _entropy_listener_utterance(self):
		"""
		Calculate the corresponding entropy of each utterance.

		Returns
        -------
        entropy_list: list(float), the entropy of each message.
		"""
		entropy_list = []
		for utter in self.lexicon.index:
			entropy_i = - np.sum(self.listener.loc[utter] * safelog2(self.listener.loc[utter]))
			entropy_list.append(np.abs(entropy_i))  # to discard the minus sign of -0.0 
		return entropy_list


	def display_listener(self):
		"""
		Display the pragmatic listener with including additional entropy, prior, and alpha information.

		Returns
        -------
        d: np.array or pd.DataFrame, depending on `self.lexicon`.
		"""
		d = self.listener

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


	def display_reference_game(self):
		"""
		Display the RSA framework as a reference game format.
		"""
		d = self.lexicon.copy()
		d['costs'] = self.costs
		d.loc['prior'] = list(self.prior) + ['']
		d.loc['alpha'] = [self.alpha] + [' '] * self.lexicon.shape[1]
		display(d)


	def display(self):
		print("\nThe reference game is as follows:")
		self.display_reference_game()
		print("\nSpeaker:")
		print(tabulate(self.speaker, headers='keys', tablefmt='psql', floatfmt='.3f'))
		print("\nListener:")
		print(tabulate(self.display_listener(), headers='keys', tablefmt='psql', floatfmt='.3f'))


	"""
	The following is the rearrangement of some functions under the original `CobwebNode` module.
	"""
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

	def 






