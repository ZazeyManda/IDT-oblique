import numpy as np
import pandas as pd

UPPERBOUND = 1e+15

class Isotonic:
    def __init__(self, ordermat: np.array, y: np.array, w: np.array=None) -> None:
        if w is None:
            w = np.repeat(1, len(y))
        n = len(y)
        y = np.pad(y, (1,0), 'constant', constant_values=-1)
        w = np.pad(w, (1,0), 'constant', constant_values=-1)
        ordermat = np.pad(ordermat, ((1,0),(1,0)), 'constant', constant_values=-1)
        out = self.monoreg(ordermat, y, w, np.arange(1,n+1))
        indices = np.argsort(out['g_star_index'])
        self.result = out['g_star_value'][indices]
    
    def maxflow(self, C):
        """
        Maxflow on the network C, returns labels of nodes that belong to maximum upperset
        """
        n = len(C) - 1
        F = np.zeros((n,n)) 
        F = np.pad(F, ((1,0), (1,0)), 'constant', constant_values=-1)
        ready = False
        labels = np.zeros((n,2))
        labels = np.pad(labels, ((1,0), (1,0)), 'constant', constant_values=-1)
        while not ready:
            labels.fill(0)
            labels[1, 1:] = np.array([1, 1000000000000000]) #TODO: vraag Ad wat deze grote constant is? kan die worden vervangen door upperbound?
            queue = np.array([0,1])
            index = 1
            while labels[n, 1] == 0:
                if len(queue) - 1 == 0:
                    return labels
                v = queue[index]
                queue = np.delete(queue, index, axis=0)
                index1 = np.arange(1,n+1)[ (C[v, 1:] > 0) & (labels[1:, 1] == 0) ]
                index2 = np.arange(1,n+1)[ (C[1:, v] > 0) & (labels[1:, 1] == 0) ]
                for i in index1:
                    if F[v, i] < C[v, i]:
                        labels[i, 1] = v
                        labels[i, 2] = min(labels[v, 2], C[v, i] - F[v, i])
                        queue = np.concatenate([queue,[i]])

                for i in index2:
                    if F[i, v] > 0:
                        labels[i, 1] = v
                        labels[i, 2] = min(labels[v, 2], F[i, v])
                        queue = np.concatenate([queue,[i]])
            v = n
            lambda_ = labels[n, 2]
            while v != 1:
                u = int(labels[v,1])
                if C[u, v] > 0:
                    F[u, v] = F[u, v] + lambda_
                else:
                    F[v, u] = F[v, u] - lambda_
                v = u
        return labels
    
    def monoreg(self, ordermat: np.array, g: np.array, w: np.array, subset: np.array) -> pd.DataFrame:
        """
        Implements the algorithm of 1. Maxwell and Muckstadt, 2. Spouge, Wan and Wilbur, 3. Picard
        ordermat is an incidence matrix representing the partial order, that is: ordermat[i,j] == 1 iff i <= j. 
        g contains the unconstrained estimates to be made monotone
        w[i] is the weight of g[i], typically the number of observations on which the unconstrained
        estimate g[i] has been computed
        """
        m = len(subset)
        # Trivial case: the subset only contains one element
        if m == 1:
            g_star_index = subset
            g_star_value = g[subset]
        else:
            # Compute the weighted average of g, with weights w, on the subset
            av = np.sum(g[subset] * w[subset]) / np.sum(w[subset])
            b = w[subset] * (g[subset] - av)
            b = np.pad(b, (1,0), 'constant', constant_values=-1)
            index_pos = np.arange(1, m + 1)[b[1:]> 0]
            index_neg = np.arange(1, m + 1)[b[1:] < 0]

            if (len(index_pos) == 0 and len(index_neg) == 0):
                g_star_index = subset
                g_star_value = np.repeat(av, m)
            else:
                # Include source and sink
                network = np.zeros((m+2, m+2)) 
                network = np.pad(network, ((1,0),(1,0)), 'constant', constant_values=None)

                # Add directed edge with "infinite" capacity from x to y iff x <= y.
                network[2:(m + 2), 2:(m + 2)] = ordermat[subset, :][:, subset] * UPPERBOUND

                # Add edge from source to elements with b > 0 with capacity b
                network[1, index_pos + 1] = b[index_pos]
                # Add edge from elements with b < 0 to sink, with capacity -b
                network[index_neg + 1, m + 2] = -b[index_neg]
                # Solve the maximum flow problem on the constructed network to find the maximum upper set
                labels = self.maxflow(network)
                temp = np.reshape(labels[2:(m+2), 2], -1, 'F')
                index = np.arange(1,m+1)[temp > 0]
                subset_padded = np.pad(subset, (1,0), constant_values=-1)
                upset = subset_padded[index]
                lowset = np.delete(subset_padded, np.append(index, 0))

                if (len(upset) == 0):
                    g_star_index = subset
                    g_star_value = np.repeat(av, m)
                
                elif (len(lowset) == 0):
                    g_star_index = subset
                    g_star_value = np.repeat(av, m)
                
                else:
                    # Recurse on the maximum upperset U and minimum lower set L
                    up = self.monoreg(ordermat, g, w, upset)
                    low = self.monoreg(ordermat, g, w, lowset)
                    # Thus we proceed with: g*|U = (g|U)* and g*|L = (g|L)*
                    g_star_index = np.concatenate((np.array(low['g_star_index']), np.array(up['g_star_index'])))
                    g_star_value = np.concatenate((np.array(low['g_star_value']), np.array(up['g_star_value'])))
        result = pd.DataFrame({'g_star_index': g_star_index, 'g_star_value': g_star_value})
        return result

# Isotonic(np.array([[0,1,1,1], [0,0,0,1], [0,0,0,1], [0,0,0,0]]), np.array([0.4, 0.2, 0.6, 0.4]))
    