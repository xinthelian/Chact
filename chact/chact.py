"""
ChactTree Module.
Thanks to Chris MacLellan (and any potential contributors, like Erik Harpstead. and me) 
for their commitment of the original Cobweb Modules.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import shuffle
from random import random
from math import log
from tqdm import tqdm

from chact.utils import weighted_choice, most_likely_choice, dfs, bfs
from chact.nodes import chactNode


class chactTree(object):

    """
    The structure of a ChactTree.
    Designed based on an original CobwebTree with additional functionalities and updated Nodes.
    To fit a chactTree, first fit the tree as a CobwebTree.
    Then from down to top, fit the CHAI framework for every node in the tree. (with self.chai())
    """

    def __init__(self, reserve_obj=True, chai_obj=False):
        """
        The tree constructor.
        """
        self.root = chactNode()
        self.root.tree = self
        self.reserve_obj = reserve_obj  # If it is False, the attr 'object' will NOT be included when training.
                                              # Note that the original instance with attr 'object' will still be in the tree.
        self.chai_obj = chai_obj  # If it is False, 'object' will never be an utterance in any CHAI framework.

    def clear(self):
        """
        Clears the concepts of the tree.
        """
        self.root = chactNode()
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
        if not self.reserve_obj:
            instance_with_c = instance.copy()
            instance = {key: value for key, value in instance.items() if key != 'object'}

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
    	

    def chai(self, verbose=True, rsa=False):
        """
        The core functionality of the CHAI framework.
        """
        sequence = []
        dfs(self.root, sequence)
        if verbose:
            print("START CHAI ACTIVATING...")
            sequence_id = [node.concept_id for node in sequence]
            # print("The traverse sequence is as follows (DFS):")
            # print(sequence_id)

            for node in tqdm(sequence, desc="Processing", unit="item"):
                node.chai_activate(include_obj=self.chai_obj, rsa=rsa)
        else:
            for node in sequence:
            	node.chai_activate(include_obj=self.chai_obj, rsa=rsa)


    def display_chai(self, depth_first=False, level=None):
        """
        Display all the CHAI frameworks of nodes chosen.
        """
        if level is None:
            print("\n\n" + "Now display the CHAI frameworks of all the nodes.")
        else:
            print("\n\n" + "Now display the CHAI frameworks the nodes at level {}.".format(level))
        current = self.root
        if depth_first:
            current.chai_display()
            if len(current.children) > 0:
                for child in current.children:
                    chlild.display_chai()
        else:
            # Display all the nodes at the same level first (bfs).
            sequence = bfs(self.root)
            for node in sequence:
                if level is None:
                    node.chai_display()
                else:
                    tree_depth = sequence[-1].depth() + 1
                    if level > tree_depth:
                        raise ValueError("The number of level exceeds the depth of the tree. FYI, the depth of the tree is {}.".format(tree_depth))
                    if node.depth() + 1 == level:
                        node.chai_display()


    def save_chai(self, depth_first=False, level=None):
        """
        Save all the CHAI frameworks of nodes chosen.
        """
        if level is None:
            print("\n\n" + "Now save the CHAI frameworks of all the nodes.")
        else:
            print("\n\n" + "Now save the CHAI frameworks the nodes at level {}.".format(level))
        current = self.root
        if depth_first:
            current.chai_display()
            if len(current.children) > 0:
                for child in current.children:
                    chlild.save_outputs(display=True)
        else:
            # Display all the nodes at the same level first (bfs).
            sequence = bfs(self.root)
            for node in sequence:
                if level is None:
                    node.save_outputs()
                else:
                    tree_depth = sequence[-1].depth() + 1
                    if level > tree_depth:
                        raise ValueError("The number of level exceeds the depth of the tree. FYI, the depth of the tree is {}.".format(tree_depth))
                    if node.depth() + 1 == level:
                        node.save_outputs(display=True)
        print("\n\nDone Saving. You can find the data files stored under the folder ./chact-outputs/")
        print("The files are named in the format 'lexicon/listener/speaker-level-concept_id'.")


    def trail_object(self, obj):

        def is_iterable(variable):
            try:
                if isinstance(obj, str):
                    return False
                iter(variable)
                return True
            except TypeError:
                return False

        if is_iterable(obj):
            for i in obj:
                if i not in self.root.objects:
                    raise ValueError("{} is not an object in the tree.".format(i))
                print("\n\nStart seeking for utterances that fit object {}:".format(i))
                # print("Original {}:", )
                self.root._trail_iter_object(i)
        else:
            # print(obj)
            if obj not in self.root.objects:
                raise ValueError("{} is not an object in the tree.".format(obj))
            print("\n\nStart seeking for utterances that fit object {}:".format(obj))
            self.root._trail_iter_object(obj)


    def trail_utter(self, instance):
        node = self.categorize(instance)
        print(node.rational_listener)

        utterances = list(instance.values())
        print("The utterances entered are:", utterances)
        for utter in utterances:
            if utter not in self.root.utterances:
                raise ValueError("{} is not an utternace in the tree.".format(utter))
        print("\n\nStart seeking for objects that fit the utterance(s) {}:".format(utterances))
        node._trail_iter_utter(instance)



