# Chact

**Chact** /tʃækt/, stands for **Continual Hierarchical Adaption through Cobweb Tree**.

An integration of human-like incremental and hierarchical learning system, [COBWEB](https://link.springer.com/content/pdf/10.1007/BF00114265.pdf), and continual pragmatics adaption theory, [CHAI](https://psycnet.apa.org/manuscript/2022-53084-001.pdf) (Continual Hierarchical Adaption through Inference) with some rearrangement of the structure.

### Data Preparation

We assume each referant has a set of attribute-value pairs describing them (in `dict`s). So the values become the utterances applied to the referants.

### Fit a Chact Tree

First define the tree module:
	tree = chactTree(reserve_obj=False, chai_obj=False)
Here `reserve_obj=False` so the name of an object (entity) won't be included when fitting the tree (but still reserved as an information for each instance/referant). `chai_obj=False` so the attribute name `object` will never be an utterance in any CHAI/RSA framework.

##### Fit It as a Cobweb Tree

Feed the tree module with each referant incrementally:

	for instance in instaces:
		tree.ifit(instance)

You may visualize the fitted Cobweb tree with
	visualize(tree)

##### Then Fit It as a Chact Tree

After that, starting from it leaves, generate CHAI framework for every node in the BFS order (MUST start from the leaves as part of the mechanism):

	tree.chai(verbose=True, rsa=False)

`verbose=True` introduces a progress bar and prompts. If you just want to introduce the original [RSA](https://www.sciencedirect.com/science/article/pii/S136466131630122X) (Rational Speech Framework) for every node, set `rsa=True`, otherwise set it to `False`.

Now we have the whole Chact tree.

### Functionalities

##### CHAI/RSA Frameworks of Nodes

To see the detailed CHAI/RSA frameworks of nodes in a tree:

	tree.display_chai(depth_first=False, level=None)

`level` indicates the frameworks of nodes from which level you want to observe (start from level 1). Displays all levels by default. The nodes will always be presented in a BFS order unless you set `depth_first=True` when you try to display all nodes in the tree, and it turns to the DFS order.

You may save these frameworks with

	tree.save_chai(depth_first=False, level=None)

Then the framework of each node becomes a `csv` file named in the format `[lexicon/listener/speaker]-[level]-[nodeID].csv` under `./chact-outputs`.


##### Trails of Objects

You can even see the trail of an object, indicating which utterance is best used for the specified object in each level.

Note that, it is possible that no "good" utterance exists in some level for the object.

	tree.trail_object(object_name)

##### Trails of Utterances

Similarly, you can see the trail of an utterance, indicating which object is best referred for the specified utterance in each level.

Note that, it is possible that no "good" object exists in some level for the utterance.

	tree.trail_utter(instance)

`instance` should be a dictionary that may include all or part of an instance. For instance, if an instance fitted into the tree looks like this:

	{'cap-color': 'brown', 'cap-shape': 'convex', 'cap-surface': 'scaly', 'classification': 'poisonous', 'gill-attachment': 'free', 'gill-spacing': 'closed', 'habitat': 'urban', 'object': 'M35'}

When displaying the trail with some given utterance(s), like `scaly` and `poisonous`, the `instance` can be like

	{'cap-surface': 'scaly', 'classification': 'poisonous'}


### Implementation Example

We offer the example script `./mushroom_chact.py` with the preprocessing of mushroom data.


### Acknowledgements

Thanks to [Chris MacLellan](https://chrismaclellan.com/) (and all the potential contributors of the `concept-formation` repo - Erik Harpstead, Lane Lawley, etc.) for their commitment of the original Cobweb modules.
