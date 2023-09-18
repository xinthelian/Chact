import pandas as pd
import numpy as np
import random

from collections import Counter
from IPython.display import display
from tabulate import tabulate
from tqdm import tqdm

from chact.chact import chactTree
from chact.datasets import load_tuna
from chact.visualize import visualize


trials = load_tuna(num_trials=10)

# trials = trials[:2]
for trial in trials:
	trial['instances']

# Display figuration of DataFrames in Pandas:
pd.set_option("display.max_columns", None)  # to display the whole dataframe
pd.set_option("display.max_rows", None)  # to display the whole dataframe
pd.options.display.float_format = "{:.3f}".format  # round up to 3 decimals when displaying floats

random.seed(123)
for trial in trials:
	random.shuffle(trial['instances'])

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


"""
Fit the tree
"""
tree = chactTree(reserve_obj=False, chai_obj=False)
print("START TREE FITTING...")

trials = [trials[8]]
targets = [trial['target'] for trial in trials]

print("Targets:")
for i in range(len(trials)):
	print(f"{trials[i]['id']}: {targets[i]}")

for i in range(len(trials)):
	trial = trials[i]
	print(f"\n\n=== Fitting Data from Trial {trial['id']} ===")
	print(f"Target: {trial['target']}")
	for instance in tqdm(trial['instances'], desc="Processing", unit="item"):
		tree.ifit(instance)
	# visualize(tree)  # visualize the CobwebTree

	# Then fit it as a ChaiTree with introduction of CHAI framework for every node:
	if i == 0:
		tree.chai(verbose=True)
	else:
		tree.chai(verbose=True, clear_first=True)
	# tree.display_chai()

tree.display_chai()
tree.trail_object(targets, second=True)
# tree.trail_utter({'x-dimension': '2', 'type': 'chair'})
# tree.trail_utter({'y-dimension': '2'})
# tree.trail_utter({'color': 'green', 'size': 'small'})
tree.trail_utter({'color': 'blue'})

	# """
	# Functionalities
	# """

	# # Display the framework of level 7:
	# tree.display_chai(level=7)

	# # Save framework for all the nodes:
	# tree.save_chai()

	# # Trail objects:
	# tree.trail_object(["M2", "M11"])

	# # Trail utterances:
	# inst = {'cap-surface': 'scaly', 'cap-shape': 'convex'}
	# tree.trail_utter(inst)

