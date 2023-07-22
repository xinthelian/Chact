import pandas as pd
import numpy as np
import random

from collections import Counter
from IPython.display import display
from tabulate import tabulate

from concept_formation.datasets import load_mushroom
from concept_formation.RSA_CobwebNode import RSA_CobwebNode_refer
from concept_formation.cobweb_for_RSA import CobwebTree
from concept_formation.visualize import visualize


# Set up the display figuration of dataframes
pd.set_option("display.max_columns", None)  # to display the whole dataframe
pd.set_option("display.max_rows", None)  # to display the whole dataframe
pd.options.display.float_format = "{:.3f}".format  # round up to 3 decimals when displaying floats


# Load data and preprocess them
mushrooms = load_mushroom()[:20]
# mushrooms_e = [mushroom for mushroom in mushrooms if mushroom['classification'] == 'edible'][:25]  # edible
# mushrooms_p = [mushroom for mushroom in mushrooms if mushroom['classification'] == 'poisonous'][:25]  # poisonous
# mushrooms = mushrooms_e + mushrooms_p  # Then there are 25 examples for edible and 25 for poisonous mushrooms
# random.seed(123)
# random.shuffle(mushrooms)

# print(mushrooms[0])

preserved_attrs = ['cap-color', 'cap-shape', 'cap-surface', 'classification', 'gill-attachment', 'gill-spacing', 'habitat']
mushrooms = [{a: mushroom[a] for a in mushroom if a in preserved_attrs}
			 for mushroom in mushrooms]

# Add name for each mushroom:
for i in range(len(mushrooms)):
	state = "%d" % (i+1)
	mushrooms[i]["name"] = state

random.seed(123)
random.shuffle(mushrooms)


def profile_data(data, display=True):
	"""
	Display the profile of the data.
	The information will be actually similar as the one in the root of the CobwebTree
	when visualizing it.
	Instead the information is displayed in probabilities instead of counts.
	========================
	Parameters:
	data: list of dicts.
	display: Boolean.
		If display=True, display the profile and returning it. Otherwise just return it.
	========================
	returns a dictionary with keys str (attribute), and values dictionaries:
	whose keys are strs (attr value, or message in the RSA case),
	and whose values are floats (probability).
	"""
	counts = {}
	for d in data:
		for key, value in d.items():
			if key in counts:
				counts[key].update([value])
			else:
				counts[key] = Counter([value])

	probs = {}
	for attr, count_item in counts.items():
		# print(attr + ":")
		total = sum(count_item.values())
		probs[attr] = {}
		for value, count in count_item.items():
			probs[attr][value] = count / total
	if display:
		print("\nThe profile of the data is as follows:")
		for attr, d in probs.items():
			print(attr + ":")
			for value, prob in d.items():
				print("-> " + value, ":", prob)
		print("\n")
	return probs


def _display_node(node, level):
	"""
	Create and display the RSA framework of some node.
	========================
	Parameters:
	node: CobwebNode
		The node used to create the RSA framework.
	level: int
		The depth of node.
	"""
	rsa_node = RSA_CobwebNode_refer(node=node, state_name="name", include_name=False, level=level)
	rsa_node.display_RSA()


def display_nodes(current):
	"""
	The recursive function of displaying all the nodes within a fitted tree.
	========================
	Parameter:
	current: CobwebNode. 
		If want to display all the nodes of the tree, current should be the root of the tree.
	"""
	print("\n\n/////////////////////////////////////")
	print("Now display node", current.concept_id)
	level = int(current.depth() + 1)
	print("Level:", level)
	_display_node(current, level)
	if len(current.children) > 0:
		for child in current.children:
			display_nodes(child)


# Display the profile of the data, mushrooms.
profile = profile_data(mushrooms)

# Fit the data into a CobwebTree.
tree = CobwebTree(reserve_classification=False)
for mushroom in mushrooms:
	tree.ifit(mushroom)
visualize(tree)

# Then build RSA framework for each CobwebNode:
print("The RSA framework for each node is as follows:")
display_nodes(tree.root)




