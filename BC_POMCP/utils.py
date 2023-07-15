import copy
import pickle
import numpy as np
import random
from BC_POMCP.root_node import RootNode
from BC_POMCP.robot_action_node import RobotActionNode
from BC_POMCP.human_action_node import HumanActionNode

def _printRed(text):
    print("\033[91m {}\033[00m" .format(text))

def _printCyan(text):
    print("\033[96m {}\033[00m".format(text))

def _print_tree_helper(root_node, value, level):
    if root_node == "empty":
        return
    if value == "root":
        print(" " * 8 * level + '->' + str(value)+ ':' + '(B=%d, Pos=%d)' %(len(root_node.belief), root_node.belief[0][0]))
    elif isinstance(root_node, RobotActionNode):
        # Robot action nodes
        _printRed(" " * 8 * level + '->' + str(value) + ':' + '(N=%d,V=%.2f,Pos=%d)' %(root_node.visited, root_node.value,
                                                                                        root_node.position))
    elif isinstance(root_node, HumanActionNode):
        # Human action nodes
        _printCyan(" " * 8 * level + '->' + str(value) + ':' + '(N=%d,V=%.2f,B=%d,Pos=%d)' % (root_node.visited,
                                                                                              root_node.value,
                                                                                              len(root_node.belief),
                                                                                              root_node.belief[0][0]))

    if level % 2 == 0:
        # Robot action nodes
        for i, child in enumerate(root_node.robot_node_children):
            _print_tree_helper(child, i, level + 1)

    else:
        # Human action nodes
        for i, child in enumerate(root_node.human_node_children):
            _print_tree_helper(child, i, level + 1)


def visualize_tree(root_node):
    """
    Visualize the search tree given the root node.
    :param root_node:
    :return:
    """
    _print_tree_helper(root_node, "root", 0)


def estimate_task_difficulty(world_state):
    """
    Returns the estimated difficulty of the task or the chances of solving based on the remaining number
    of code possibilities and the current turn in the game.
    :param world_state:
    :return:
    """
    pass


def save_tree(root_node):
    """
    Save the search tree as a list
    :param root_node:
    :return:
    """
    print(root_node)
    with open("tree_data.pkl", "wb") as f:
        pickle.dump(root_node, f)
    # nodes_list = []
    # tree_root = copy.deepcopy(root_node)
    # nodes_list.append({"type": "root", "value": 0, "visited": 0, "belief": root_node.belief})
    # node_iter = 0
    #
    # while node_iter < len(nodes_list):
    #     if tree_root.type == "empty":
    #         # TODO: handle Leaf nodes
    #         continue
    #     elif tree_root.type == "root" or tree_root.type == "human":
    #         # Store values of human node's children (i.e., robot nodes)
    #         for action, child in enumerate(tree_root.robot_node_children):
    #             if child == "empty":
    #                 nodes_list.append({})
    #             else:
    #                 nodes_list.append({"type": "robot", "value": child.value, "visited": child.visited, "action": action,
    #                                    "belief": None})  # Robot nodes do not have belief
    #
    #     else:
    #         # Store values of robot node's children (i.e., human nodes)
    #         for action, child in enumerate(tree_root.human_node_children):
    #             if child == "empty":
    #                 nodes_list.append({})
    #             else:
    #                 nodes_list.append({"type": "human", "value": child.value, "visited": child.visited, "action": action,
    #                                    "belief": child.belief})  # Human nodes store belief updates
    #
    #     # Make the next node as the root
    return 0


def restore_tree(filepath):
    with open(filepath, "rb") as f:
        root_node = pickle.load(filepath)

    return root_node