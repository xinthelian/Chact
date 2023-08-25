import pandas as pd
import numpy as np
import random

from collections import Counter
from IPython.display import display
from tabulate import tabulate
from tqdm import tqdm

from chact.chact import chactTree
from chact.datasets import load_mushroom
from chact.visualize import visualize


def add_shared_key(instances, new_key):
	tracking_dict = {}
	count = 1

	for instance in instances:
		values = tuple(instance.values())
		if values in tracking_dict:
			instance[new_key] = "M%d" % tracking_dict[values]
		else:
			tracking_dict[values] = count
			instance[new_key] = "M%d" % count
			count += 1

	return count

# Display figuration of DataFrames in Pandas:
pd.set_option("display.max_columns", None)  # to display the whole dataframe
pd.set_option("display.max_rows", None)  # to display the whole dataframe
pd.options.display.float_format = "{:.3f}".format  # round up to 3 decimals when displaying floats

# Load data:
mushrooms = load_mushroom()[:550]
mushrooms_e = [mushroom for mushroom in mushrooms if mushroom['classification'] == 'edible'][:50]  # edible
mushrooms_p = [mushroom for mushroom in mushrooms if mushroom['classification'] == 'poisonous'][:50]  # poisonous
mushrooms = mushrooms_e + mushrooms_p  # Then there are 25 examples for edible and 25 for poisonous mushrooms

preserved_attrs = ['cap-color', 'cap-shape', 'cap-surface', 'classification', 'gill-attachment', 'gill-spacing', 'habitat']
mushrooms = [{a: mushroom[a] for a in mushroom if a in preserved_attrs}
			 for mushroom in mushrooms]

# Add object name to each mushroom.
# Note that, mushrooms with the same key-value pairs will share the same object name.
count = add_shared_key(mushrooms, "object")
random.seed(123)
random.shuffle(mushrooms)
# for mushroom in mushrooms:
# 	if mushroom['object'] == 'M33':
# 		print(mushroom)


# Add object name for each mushroom:
# for i in range(len(mushrooms)):
# 	object_name = "M%d" % (i+1)
# 	mushrooms[i]["object"] = object_name

# random.seed(123)
# random.shuffle(mushrooms)


def profile_data(data, display=True, obj=False):
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
			if not obj:
				if attr != 'object':
					print(attr + ":")
					for value, prob in d.items():
						print("-> " + value, ":", prob)
			else:
				print(attr + ":")
				for value, prob in d.items():
					print("-> " + value, ":", prob)
		print("\n")
	return probs

# Display the profile of the data, mushrooms.
# profile = profile_data(mushrooms)

"""
Fit the tree
"""
tree = chactTree(reserve_obj=False, chai_obj=False)
print("START TREE FITTING...")
for mushroom in tqdm(mushrooms, desc="Processing", unit="item"):
	tree.ifit(mushroom)
visualize(tree)  # visualize the CobwebTree

# Then fit it as a ChaiTree with introduction of CHAI framework for every node:
tree.chai(verbose=True)

"""
Functionalities
"""

# Display the framework of level 7:
tree.display_chai(level=7)

# Save framework for all the nodes:
tree.save_chai()

# Trail objects:
tree.trail_object(["M2", "M11"])

# Trail utterances:
inst = {'cap-surface': 'scaly', 'cap-shape': 'convex'}
tree.trail_utter(inst)


