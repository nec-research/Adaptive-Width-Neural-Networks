from collections import defaultdict

import numpy as np
import torch

################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-04-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
#                                                                              #
# Modifications:                                                               #
#   - Copyright (c) 2025 Federico Errica <federico.errica@neclab.eu>           #
#   - Modified on 2025-10-28: added function to properly migrate state between #
#     conceptually identical parameters with different ids                     #
#     (e.g., for width adaptation)                                             #
################################################################################
"""
    Utilities to handle optimizer's update when using dynamic architectures.
    Dynamic Modules (e.g. multi-head) can change their parameters dynamically
    during training, which usually requires to update the optimizer to learn
    the new parameters or freeze the old ones.
"""

colors = {
    "END": "\033[0m",
    0: "\033[32m",
    1: "\033[33m",
    2: "\033[34m",
    3: "\033[35m",
    4: "\033[36m",
}
colors[None] = colors["END"]


def _map_optimized_params(optimizer, parameters, old_params=None):
    """
    Establishes a mapping between a list of named parameters and the parameters
    that are in the optimizer, additionally,
    returns the lists of:

    returns:
        new_parameters: Names of new parameters in the provided "parameters" argument,
                        that are not in the old parameters
        changed_parameters: Names and indexes of parameters that have changed (grown, shrink)
        not_found_in_parameters: List of indexes of optimizer parameters
                                 that are not found in the provided parameters
    """

    if old_params is None:
        old_params = {}

    group_mapping = defaultdict(dict)
    new_parameters = []

    found_indexes = []
    changed_parameters = []
    for group in optimizer.param_groups:
        params = group["params"]
        found_indexes.append(np.zeros(len(params)))

    for n, p in parameters.items():
        gidx = None
        pidx = None

        # Find param in optimizer
        found = False

        if n in old_params:
            search_id = id(old_params[n])
        else:
            search_id = id(p)

        for group_idx, group in enumerate(optimizer.param_groups):
            params = group["params"]
            for param_idx, po in enumerate(params):
                if id(po) == search_id:
                    gidx = group_idx
                    pidx = param_idx
                    found = True
                    # Update found indexes
                    assert found_indexes[group_idx][param_idx] == 0
                    found_indexes[group_idx][param_idx] = 1
                    break
            if found:
                break

        if not found:
            new_parameters.append(n)

        if search_id != id(p):
            if found:
                changed_parameters.append((n, gidx, pidx))

        if len(optimizer.param_groups) > 1:
            group_mapping[n] = gidx
        else:
            group_mapping[n] = 0

    not_found_in_parameters = [np.where(arr == 0)[0] for arr in found_indexes]

    return (
        group_mapping,
        changed_parameters,
        new_parameters,
        not_found_in_parameters,
    )

def _build_tree_from_name_groups(name_groups):
    root = _TreeNode("")  # Root node
    node_mapping = {}

    # Iterate through each string in the list
    for name, group in name_groups.items():
        components = name.split(".")
        current_node = root

        # Traverse the tree and construct nodes for each component
        for component in components:
            if component not in current_node.children:
                current_node.children[component] = _TreeNode(
                    component, parent=current_node
                )
            current_node = current_node.children[component]

        # Update the groups for the leaf node
        if group is not None:
            current_node.groups |= set([group])
            current_node.update_groups_upwards()  # Inform parent about the groups

        # Update leaf node mapping dict
        node_mapping[name] = current_node

    # This will resolve nodes without group
    root.update_groups_downwards()
    return root, node_mapping


def _print_group_information(node, prefix=""):
    # Print the groups for the current node

    if len(node.groups) == 1:
        pstring = (
            colors[list(node.groups)[0]]
            + f"{prefix}{node.global_name()}: {node.groups}"
            + colors["END"]
        )
        print(pstring)
    else:
        print(f"{prefix}{node.global_name()}: {node.groups}")

    # Recursively print group information for children nodes
    for child_name, child_node in node.children.items():
        _print_group_information(child_node, prefix + "  ")


def migrate_optimizer_state(optimizer, old_param, new_param):
    """
    Attempt to preserve optimizer state tensors when replacing a parameter
    (e.g., after expanding a layer). Works generically for any optimizer
    without assuming specific state structure (e.g., Adam, SGD, RMSProp).

    This function copies overlapping tensor slices from the old parameter’s
    optimizer state into the new parameter’s state, preserving the optimizer’s
    momentum or moving averages for the overlapping region. It then removes
    the old parameter’s state entry.

    Example:
        >>> migrate_optimizer_state(optimizer, old_layer.weight, new_layer.weight)

    Args:
        optimizer (Optimizer):
            A PyTorch optimizer whose state should be migrated.
        old_param (torch.nn.Parameter):
            The old parameter being replaced (must exist in optimizer.state).
        new_param (torch.nn.Parameter):
            The new parameter taking its place (e.g., after layer expansion).

    Notes:
        - If `old_param` has no entry in the optimizer state, this is a no-op.
        - For each tensor in the old state, the overlapping region
          (defined by the minimum shape along each dimension) is copied.
        - Non-tensor fields (e.g., 'step' counters) are shallow-copied.
        - The new state is created even if shapes differ.
        - This is optimizer-agnostic: it works for Adam, SGD, RMSProp, etc.
        - This function does *not* modify `optimizer.param_groups`.

    Behavior overview:
        - Preserves partial optimizer statistics for overlapping tensor regions.
        - Zeros out extra entries in new parameters beyond the old shape.
        - Removes old parameter’s state from the optimizer.

    """
    # Skip if the old parameter has no state
    if old_param not in optimizer.state:
        return

    old_state = optimizer.state[old_param]
    new_state = {}

    for key, value in old_state.items():
        # Handle tensor-valued state entries (e.g., Adam's exp_avg, exp_avg_sq)
        # the second check avoids that Adam's parameters like step for a layer like "layers.0.weight" get a wrong shape. They will be handled differently. 
        if isinstance(value, torch.Tensor) and (len(value.shape) == len(new_param.data.shape)):
            
            # Create a zero tensor matching the new parameter's shape
            new_value = torch.zeros_like(new_param.data)

            # Determine the overlapping slice between old and new tensors
            common_shape = tuple(min(a, b) for a, b in zip(value.shape, new_param.shape))
            slices = tuple(slice(0, s) for s in common_shape)

            # Copy the overlapping region from old state tensor
            new_value[slices] = value[slices]
            new_state[key] = new_value
        else:
            # Non-matching fields are copied as-is
            new_state[key] = value

    # Register new parameter state and remove old one
    if new_state:
        optimizer.state[new_param] = new_state
    else:
        # Leave it uninitialized: the optimizer will lazily create state on step()
        optimizer.state.pop(new_param, None)

    optimizer.state.pop(old_param, None)


def update_optimizer(
    optimizer,
    new_params,
    optimized_params=None,
    reset_state=False,
    remove_params=False,
    verbose=False,
):
    """Update the optimizer by adding new parameters,
    removing removed parameters, and adding new parameters
    to the optimizer, for instance after model has been adapted
    to a new task. The state of the optimizer can also be reset,
    it will be reset for the modified parameters.

    Newly added parameters are added by default to parameter group 0

    :param new_params: Dict (name, param) of new parameters
    :param optimized_params: Dict (name, param) of
                             currently optimized parameters
    :param reset_state: Whether to reset the optimizer's state (i.e momentum).
                        Defaults to False.
    :param remove_params: Whether to remove parameters that were in the optimizer
                          but are not found in new parameters. For safety reasons,
                          defaults to False.
    :param verbose: If True, prints information about inferred
                    parameter groups for new params

    :return: Dict (name, param) of optimized parameters
    """
    (
        group_mapping,
        changed_parameters,
        new_parameters,
        not_found_in_parameters,
    ) = _map_optimized_params(
        optimizer, new_params, old_params=optimized_params
    )

    # Change reference to already existing parameters
    # i.e growing IncrementalClassifier
    for name, group_idx, param_idx in changed_parameters:
        group = optimizer.param_groups[group_idx]
        old_p = optimized_params[name]
        new_p = new_params[name]
        # Look for old parameter id in current optimizer
        group["params"][param_idx] = new_p
        if old_p in optimizer.state:
            migrate_optimizer_state(optimizer, old_p, new_p)

    # Remove parameters that are not here anymore
    # This should not happend in most use case
    if remove_params:
        for group_idx, idx_list in enumerate(not_found_in_parameters):
            for j in sorted(idx_list, key=lambda x: x, reverse=True):
                p = optimizer.param_groups[group_idx]["params"][j]
                optimizer.param_groups[group_idx]["params"].pop(j)
                if p in optimizer.state:
                    optimizer.state.pop(p)
                del p

    # Add newly added parameters (i.e Multitask, PNN)

    param_structure = _ParameterGroupStructure(group_mapping, verbose=verbose)

    # New parameters
    for key in new_parameters:
        new_p = new_params[key]
        group = param_structure[key].single_group
        optimizer.param_groups[group]["params"].append(new_p)
        optimized_params[key] = new_p
        optimizer.state[new_p] = {}

    if reset_state:
        optimizer.state = defaultdict(dict)

    return new_params

class _ParameterGroupStructure:
    """
    Structure used for the resolution of unknown parameter groups,
    stores parameters as a tree and propagates parameter groups from leaves of
    the same hierarchical level
    """

    def __init__(self, name_groups, verbose=False):
        # Here we rebuild the tree
        self.root, self.node_mapping = _build_tree_from_name_groups(
            name_groups
        )
        if verbose:
            _print_group_information(self.root)

    def __getitem__(self, name):
        return self.node_mapping[name]


class _TreeNode:
    def __init__(self, name, parent=None):
        self.name = name
        self.children = {}
        self.groups = (
            set()
        )  # Set of groups (represented by index) this node belongs to
        self.parent = parent  # Reference to the parent node
        if parent:
            # Inform the parent about the new child node
            parent.add_child(self)

    def add_child(self, child):
        self.children[child.name] = child

    def update_groups_upwards(self):
        if self.parent:
            if self.groups != {None}:
                self.parent.groups |= (
                    self.groups
                )  # Update parent's groups with the child's groups
            self.parent.update_groups_upwards()  # Propagate the group update to the parent

    def update_groups_downwards(self, new_groups=None):
        # If you are a node with no groups, absorb
        if len(self.groups) == 0 and new_groups is not None:
            self.groups = self.groups.union(new_groups)

        # Then transmit
        if len(self.groups) > 0:
            for key, children in self.children.items():
                children.update_groups_downwards(self.groups)

    def global_name(self, initial_name=None):
        """
        Returns global node name
        """
        if initial_name is None:
            initial_name = self.name
        elif self.name != "":
            initial_name = ".".join([self.name, initial_name])

        if self.parent:
            return self.parent.global_name(initial_name)
        else:
            return initial_name

    @property
    def single_group(self):
        if len(self.groups) == 0:
            raise AttributeError(
                f"Could not identify group for this node {self.global_name()}"
            )
        elif len(self.groups) > 1:
            raise AttributeError(
                f"No unique group found for this node {self.global_name()}"
            )
        else:
            return list(self.groups)[0]
