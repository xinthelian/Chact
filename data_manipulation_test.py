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

# def add_shared_key(instances, new_key):
# 	tracking_dict = {}
# 	count = 1

# 	for instance in instances:
# 		kv_pairs = tuple((k, v) for k, v in instance.items() if k != new_key)
# 		if kv_pairs in tracking_dict:
# 			count = tracking_dict[kv_pairs]
# 		else:
# 			tracking_dict[kv_pairs] = count

# 		instance[new_key] = "M%d" % count
# 		count += 1


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
			# tracking_dict[values] = tracking_dict.values() + 1

	print(tracking_dict)


# def add_shared_key(dictionaries, new_key):
#     # Create dictionaries to keep track of the key-value pairs and their counts
#     tracking_dict = {}
#     value_count_dict = {}
#     count = 1

#     # Iterate through each dictionary in the list
#     for dictionary in dictionaries:
#         # Create a tuple of key-value pairs excluding the new key to be added
#         key_value_pairs = tuple((key, value) for key, value in dictionary.items() if key != new_key)

#         # Check if the key-value pairs are already in the tracking dictionary
#         if key_value_pairs in tracking_dict:
#             # Get the existing count for the key-value pairs
#             count = tracking_dict[key_value_pairs]
#         else:
#             # If the key-value pairs are not in the tracking dictionary, add them with the current count
#             tracking_dict[key_value_pairs] = count

#         # Check if the current key-value pairs and the value are already in the value_count_dict
#         value_tuple = (key_value_pairs, dictionary[new_key])
#         if value_tuple in value_count_dict:
#             # Get the existing count for the key-value pairs and the value
#             count = value_count_dict[value_tuple]
#         else:
#             # If the key-value pairs and the value are not in the value_count_dict, add them with the current count
#             value_count_dict[value_tuple] = count

#         # Set the new value with the current count for the new key
#         dictionary[new_key] = "M%d" % count

#         # Increment the count for the next unique set of key-value pairs
#         count += 1

# Convert each dictionary to a tuple and create a set of unique tuples


# def unique_dicts(dicts):
# 	unique_tuples = set(tuple(dictionary.items()) for dictionary in dicts)
# 	unique_dicts = [dict(items) for items in unique_tuples]
# 	return unique_dicts


# def add_shared_key(dicts, unique_dicts, new_key):
# 	new_dicts = dicts.copy()
# 	for unique_d in unique_dicts:
# 		count = 1
# 		for d in new_dicts:
# 			if d == unique_d:
# 				d[new_key] = "M%d" % count
# 		count += 1



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

# Clear the duplicates:
# unique_mushrooms = unique_dicts(mushrooms)
new_key = "object"
add_shared_key(mushrooms, new_key)

print(len(mushrooms))
print([mushroom['object'] for mushroom in mushrooms])
print([mushroom for mushroom in mushrooms if mushroom['object'] == 'M7'])
