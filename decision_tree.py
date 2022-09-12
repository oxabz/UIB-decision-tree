
from itertools import count
import json
from random import shuffle
from typing import Any, Callable, Optional, Tuple, Union
from typing_extensions import Self
from sklearn.model_selection import train_test_split
import numpy as np
import utils as ut

class Node:
    def predict(self, x):
        raise NotImplementedError("Node is abstract")
    def prunning_prep(self, x, y) -> int:
        raise NotImplementedError("Node is abstract")
    def prune(self, x, y) -> Tuple[Self, int]:
        raise NotImplementedError("Node is abstract")
    def toDot(self, parent, elem_count):
        raise NotImplementedError("Node is abstract")
    def toDotWData(self, x, y, parent, elem_count):
        raise NotImplementedError("Node is abstract")


class Leaf(Node):
    label: Any
    
    def __init__(self, label) -> None:
        super().__init__()
        self.label = label

    def predict(self, x):
        # When we reach a leaf we return the labels 
        # along the index of the x associated with the label
        return np.concatenate([
                x[:,-1].reshape(-1,1), 
                np.full((x.shape[0],1), self.label)
            ], axis=1)

    def prunning_prep(self, x, y)->int:
        return np.sum(y == self.label)

    def prune(self, x, y) -> Tuple[Node, int]:
        return (self, np.sum(y == self.label))

    def toDot(self, parent, elem_count):
        elem_count["0"] += 1
        node = str(self.label) + str(elem_count["0"])
        print("""   {}[shape=circle label=\"{}\"]""".format(node, self.label))
        if parent is not None:
            print("""   {} -> {} """.format(parent, node))
    
    def toDotWData(self, x, y, parent, elem_count):
        counts = ut.count_vals(y)
        elem_count["0"] += 1
        node = str(self.label) + str(elem_count["0"])
        print("""   {}[label=\"{}\\n{}\"]""".format(node, self.label, counts))
        if parent is not None:
            print("""   {} -> {} """.format(parent, node))



class Branch(Node): 
    feature:int
    boundry:float
    sup_branch:Node
    inf_branch:Node
    
    def __init__(self, feature:int, boundry:float, sup_branch:Node, inf_branch:Node, majority_label:Any) -> None:
        super().__init__()
        self.feature = feature
        self.boundry = boundry
        self.sup_branch = sup_branch
        self.inf_branch = inf_branch
        self._prunning_hits = None
        self._majority_label = majority_label

    def predict(self,x):
        infmask =  x[:, self.feature] < self.boundry
        xinf = x[infmask]
        xsup = x[~infmask]
        yinf = self.inf_branch.predict(xinf)
        ysup = self.sup_branch.predict(xsup)
        return np.concatenate([yinf, ysup])
    
    def prunning_prep(self, x, y)->int:
        infmask =  x[:, self.feature] < self.boundry
        xinf = x[infmask]
        xsup = x[~infmask]
        yinf = y[infmask]
        ysup = y[~infmask]
        self._prunning_hits = self.inf_branch.prunning_prep(xinf,yinf) + self.sup_branch.prunning_prep(xsup,ysup)
        return self._prunning_hits

    # def prune(self, x, y) -> Node:
    #     if self._prunning_hits is None: 
    #         raise "prune should never be call before prunning_prep"
        
    #     acc = np.sum(y == self._majority_label)

    #     if self._prunning_hits > acc:
    #         infmask =  x[:, self.feature] < self.boundry
    #         xinf = x[infmask]
    #         xsup = x[~infmask]
    #         yinf = y[infmask]
    #         ysup = y[~infmask]
    #         self.inf_branch = self.inf_branch.prune(xinf,yinf)
    #         self.sup_branch = self.sup_branch.prune(xsup,ysup)
    #         return self
    #     else: 
    #         return Leaf(self._majority_label)

    def prune(self, x, y):
        infmask =  x[:, self.feature] < self.boundry
        xinf = x[infmask]
        xsup = x[~infmask]
        yinf = y[infmask]
        ysup = y[~infmask]
        inf_branch, inf_acc = self.inf_branch.prune(xinf,yinf)
        sup_branch, sup_acc = self.sup_branch.prune(xsup,ysup)

        acc = np.sum(y == self._majority_label)

        if acc < inf_acc + sup_acc :
            self.sup_branch = sup_branch
            self.inf_branch = inf_branch
            return (self, inf_acc + sup_acc)
        else :
            return (Leaf(self._majority_label), acc)

    def toDot(self, parent, elem_count):
        elem_count["0"] += 1
        node = str(self._majority_label) + str(elem_count["0"])
        print("""   {}[shape=box label=\"{}\\n x[{}] < {:.2f}\"]""".format(node, self._majority_label, self.feature, self.boundry))
        if parent is not None:
            print("""   {} -> {}""".format(parent, node))
        self.inf_branch.toDot(node, elem_count)
        self.sup_branch.toDot(node, elem_count)

    def toDotWData(self, x, y, parent, elem_count):
        counts = ut.count_vals(y)
        elem_count["0"] += 1
        node = str(self._majority_label) + str(elem_count["0"])
        print("""   {}[shape=box label=\"{}\\n x[{}] < {:.2f}\\n{}\"]""".format(node, self._majority_label, self.feature, self.boundry, counts))
        if parent is not None:
            print("""   {} -> {}""".format(parent, node))
        
        infmask =  x[:, self.feature] < self.boundry
        xinf = x[infmask]
        xsup = x[~infmask]
        yinf = y[infmask]
        ysup = y[~infmask]

        self.inf_branch.toDotWData(xinf, yinf, node, elem_count)
        self.sup_branch.toDotWData(xsup, ysup, node, elem_count)

class DecisionTree:
    impurity_mesurement: Callable[[np.array], float]
    root_node: Optional[Node]

    def __init__(self, impurity_mesurement: Union[str,Callable[[np.array], float]] = "entropy"):
        if type(impurity_mesurement) == Callable[[np.array], float]:
            self.impurity_mesurement = impurity_mesurement
        elif impurity_mesurement == "entropy":
            self.impurity_mesurement = ut.entropy
        elif impurity_mesurement == "gini":
            self.impurity_mesurement = ut.gini_impurity

    def _build_tree(self, x:np.ndarray, y:np.ndarray) -> Node:
        if np.all(y == y[0]):
            # If every y are equal return a leaf
            return Leaf(y[0])
        elif np.all(x == x[0]):
            # If every x are equal then we return a leaf with the most common label
            count = ut.count_vals(y)
            return Leaf(ut.dict_max(count)[0])
        else:
            # Infos about the best split found
            best_IG = 0
            best_feature = None
            best_split = None

            y_cached_split = [] # We can cache some of the work we already did
            cached_mask = None  # trying to find the best split
            
            # Useful stuff to compute IG but doesnt change between features
            E_base = self.impurity_mesurement(y)
            n = len(y)

            for feat in range(x.shape[1]):
                # We split our classes along the mean of the feature 
                split_plane = x[:, feat].mean()
                split_mask = x[:, feat] < split_plane
                ysubset1 = y[split_mask]
                ysubset2 = y[np.logical_not(split_mask)]

                # We compute the information gain for that feature
                E_s1 = self.impurity_mesurement(ysubset1)
                E_s2 = self.impurity_mesurement(ysubset2)
                E_s = (E_s1 * len(ysubset1) + E_s2 * len(ysubset2))/ n
                IG_s = E_base - E_s

                # If it is better than the previous IG save the split
                if IG_s >= best_IG:
                    best_IG = IG_s
                    best_feature = feat
                    best_split = split_plane
                    y_cached_split = [ysubset1, ysubset2]
                    cached_mask = split_mask
            
            counts = ut.count_vals(y)
            majority_label = ut.dict_max(counts)[0]

            return Branch(
                best_feature, 
                best_split, 
                inf_branch=self._build_tree(x[cached_mask], y_cached_split[0]), 
                sup_branch=self._build_tree(x[~cached_mask], y_cached_split[1]),
                majority_label=majority_label
            )

    def fit(self, x, y, skip_pruning=False, pruning_size=0.3):
        x_train, x_prune, y_train, y_prune = train_test_split(x, y, test_size=pruning_size, shuffle=False)
        self.root_node = self._build_tree(x_train, y_train)
        if skip_pruning:
            return
        #self.root_node.prunning_prep(x_prune, y_prune)
        self.root_node = self.root_node.prune(x_prune, y_prune)[0]
        #self.toDotWData(x_prune,y_prune)


    def predict(self, x):
        # We add indices to the xs so that we can order the ys afterward
        indices = np.arange(x.shape[0]).reshape((-1, 1))
        preds = self.root_node.predict(np.concatenate([x,indices], axis=1))
        preds = preds[preds[:, 0].argsort()]
        return preds[:,-1]

    def toDot(self):
        print("digraph D {")
        self.root_node.toDot(None, {"0":0})
        print("}")

    def toDotWData(self, x, y):
        print("digraph D {")
        self.root_node.toDotWData(x, y, None, {"0":0})
        print("}")

    def __call__(self, x):
        return self.predict(x)
