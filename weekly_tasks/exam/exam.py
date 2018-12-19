# Beginning of file for exam

class BinaryTree(object):
    """
    Binary tree class.

    Parameters
    ----------

    label: string: what label the node should have

    left: the left child of the node (None if the node has no left child)
    right: the right child of the node (None if the node has no right child)

    if the node has only one child, it must be the left child (right child
    remains empty)

    """

    def __init__(self, label, left=None, right=None):
        if not left and right:
            raise ValueError("If the node has only one child, it must be the left child")
        self.label = label
        self.left = left
        self.right = right

    def __repr__(self):
        if self.left and self.right:
            return " ".join(["(", str(self.left), str(self.label), str(self.right),\
            ")"])
        elif self.left:
            return " ".join(["(", str(self.left), str(self.label),\
            ")"])
        else:
            return str(self.label)

    def __eq__(self, other):
        # Ex. 5
        pass

    def get_label(self):
        """
        Returns the label of the tree.
        """
        # Ex. 1
        pass

    def get_leaves(self):
        """
        Return the list of labels of leaves present in the tree.
        >>> BinaryTree("+", BinaryTree(3), BinaryTree(2)).get_leaves()
        [3, 2]
        """
        # Ex. 4
        pass

    def has_children(self):
        """
        Return True if the tree has children.
        """
        # Ex. 2
        pass

    def get_children(self):
        """
        Return an iterable of all children. If the node has no children, return empty iterable.
        """
        # Ex. 6
        pass

    def get_max_depth(self):
        """
        Return the number representing the depth of the tree. If the tree has no
        children = depth 0

        >>> BinaryTree("*", BinaryTree("+", BinaryTree(1), BinaryTree(2)),
    BinaryTree("-", BinaryTree(5), BinaryTree(4))).get_max_depth()
        2
        """
        # Ex. 3
        pass

    def get_ndeep_tree(self, n):
        """
        Return the list of trees that are n-deep in the tree. The root is of
        depth 0. If there is no tree n-deep in the tree, return the empty list, []

        >>> BinaryTree("*", BinaryTree("+", BinaryTree(1), BinaryTree(2)),
    BinaryTree("-", BinaryTree(5), BinaryTree(4))).get_ndeep_tree(1)
        [ BinaryTree("+", BinaryTree(1), BinaryTree(2)), BinaryTree("-",
            BinaryTree(5), BinaryTree(4)) ]
        """
        # Ex. 7
        pass

def translate_into_binary_tree(hierarchical_list):
    """
    Helper function. You don't have to use it. But in case you want to create more BinaryTrees, it might be convenient

    hierarchical_list: list: list that might have other lists embedded

    Return BinaryTree that takes the first element in the list as label, and the
    second and third element as left and right children
    Raise ValueError if some list has more than 3 elements or less than 1
    element (it is missing label)

    >>> translate_into_binary_tree(["+", [1], [2]])
    BinaryTree("+", BinaryTree(1), BinaryTree(2))
    """
    if len(hierarchical_list) > 3:
        raise ValueError("Too many elements")
    if len(hierarchical_list) < 1:
        raise ValueError("Too few elements")
    if len(hierarchical_list) == 3:
        return BinaryTree(hierarchical_list[0],
            translate_into_binary_tree(hierarchical_list[1]), translate_into_binary_tree(hierarchical_list[2]))
    elif len(hierarchical_list) == 2:
        return BinaryTree(hierarchical_list[0],
            translate_into_binary_tree(hierarchical_list[1]))
    else:
        return BinaryTree(hierarchical_list[0])

#print(translate_into_binary_tree(["+", [1], [2]]))

class Tree(object):
    """
    Tree class.

    Parameters
    ----------

    label: string: what label the node should have

    children: all children of the node

    """

    def __init__(self, label, *children):
        self.label = label
        self.children = children

    def __repr__(self):
        if self.children:
            return " ".join(["(", str(self.label)] + [str(x) for x in
                self.children] + [ ")" ])
        else:
            return str(self.label)

    def create_binary_tree(self):
        """
        Create BinaryTree out of Tree.

        If there are more children, the tree will be binarized:
        >>> Tree("+", Tree(1), Tree(5), Tree(3), Tree(10)).create_binary_tree()
        BinaryTree("+", BinaryTree(1) ("+", BinaryTree(5), ("+", BinaryTree(3),
            BinaryTree(10))))
        """
        # Ex. 8
        pass


#############################################################################
############################# E X E R C I S E S #############################
#############################################################################

### Exercise 1
# Implement the get_label method. The method returns the label of the tree:

print("Test 1.1", BinaryTree("+", BinaryTree(1), BinaryTree(0)).get_label() ==
        "+")
print("Test 1.2", BinaryTree(1).get_label() == 1)

### Exercise 2
# Implement the method has_children. Returns True if the tree has children,
# False otherwise.

print("Test 2.1", BinaryTree("+", BinaryTree(1), BinaryTree(0)).has_children()
        == True)
print("Test 2.2", BinaryTree(1).has_children()
        == False)
print("Test 2.3", BinaryTree("exp", BinaryTree(1)).has_children()
        == True)

### Exercise 3
# Implement the method get_max_depth. This method returns the maximal depth of a
# tree (how many steps it maximally takes to get from the root to the leaves).

print("Test 3.1", BinaryTree(2).get_max_depth() == 0)
print("Test 3.2", BinaryTree("+", BinaryTree(3), BinaryTree(2)).get_max_depth()
        == 1)
print("Test 3.2", BinaryTree("+", BinaryTree("+", BinaryTree("+",
    BinaryTree("+", BinaryTree(1), BinaryTree(2)), BinaryTree(2)), BinaryTree(2)), BinaryTree(2)).get_max_depth()
        == 4)

### Exercise 4
# Implement get_leaves method. The method returns the list of the labels of all leaves.
# The list is ordered in the same way as tree leaves.
# For example BinaryTree("*", BinaryTree("+", BinaryTree(1), BinaryTree(2)),
#    BinaryTree("-", BinaryTree(5), BinaryTree(4))).get_leaves()
# returns [1, 2, 5, 4]

print("Test 4.1", BinaryTree("+", BinaryTree(3), BinaryTree(2)).get_leaves() == [3, 2])
print("Test 4.2", BinaryTree("*", BinaryTree("+", BinaryTree(1), BinaryTree(2)),
    BinaryTree("-", BinaryTree(5), BinaryTree(4))).get_leaves() == [1, 2, 5, 4])

### Exercise 5
# Implement the __eq__ method, testing for equivalence. Two trees are equivalent
# if they have the same label and the same left and right children (if any).

print("Test 5.1", BinaryTree(2) == BinaryTree(2))
print("Test 5.2", BinaryTree(2) != BinaryTree(3))
print("Test 5.3", BinaryTree(5) != 5)

### Exercise 6
# Implement the method get_children. Return an iterable of children (empty iterable if no children are present). It is up to you to choose the iterable.

print("Test 6.1", list(BinaryTree("+", BinaryTree(1), BinaryTree(0)).get_children())
== [BinaryTree(1), BinaryTree(0)])
print("Test 6.2", list(BinaryTree(5).get_children()) == [])


### Exercise 7
# Implement the get_ndeep_tree method. The method returns the list of all trees
# that are n-deep in the tree (counting from the root, which has embedding
# 0). It returns the empty list if there is no n-deep tree.

print("Test 7.1", BinaryTree(2).get_ndeep_tree(0) ==  [ BinaryTree(2) ])
print("Test 7.2", BinaryTree(2).get_ndeep_tree(1) ==  [])
print("Test 7.3", BinaryTree("*", BinaryTree("+", BinaryTree(1), BinaryTree(2)),
    BinaryTree("-", BinaryTree(5), BinaryTree(4))).get_ndeep_tree(2) == [
        BinaryTree(1), BinaryTree(2), BinaryTree(5), BinaryTree(4) ])
print("Test 7.4", BinaryTree("+", BinaryTree("+", BinaryTree("+",
    BinaryTree("+", BinaryTree(1), BinaryTree(2)), BinaryTree(2)),
    BinaryTree(2)), BinaryTree(2)).get_ndeep_tree(3) == [ BinaryTree("+",
        BinaryTree(1), BinaryTree(2)), BinaryTree(2) ])


### Exercise 8
# Consider the class Tree. It is like BinaryTree but it allows any number of
# children.
# Implement the method create_binary_tree. The method should take an instance of
# class Tree and return an instance of class BinaryTree. This requires that if
# there are more children, they are binarized as follows:
# the leftmost child is kept; the following children become a new node, with
# the label the same as in the original tree
# See tests for examples

print("Test 8.1", Tree("+", Tree(1), Tree(5)).create_binary_tree() ==\
BinaryTree("+", BinaryTree(1), BinaryTree(5)))
print("Test 8.2", Tree("+", Tree(1), Tree(5), Tree(3)).create_binary_tree() ==\
BinaryTree("+", BinaryTree(1), BinaryTree("+", BinaryTree(5), BinaryTree(3))))
print("Test 8.3", Tree("+", Tree(1), Tree(5), Tree(3), Tree(10)).create_binary_tree() ==\
BinaryTree("+", BinaryTree(1), BinaryTree("+", BinaryTree(5), BinaryTree("+",\
    BinaryTree(3), BinaryTree(10)))))
