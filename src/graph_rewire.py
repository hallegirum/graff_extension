## gradient-based rewiring
from itertools import product
import torch
from torch_geometric.utils import get_laplacian, remove_self_loops, add_remaining_self_loops, contains_self_loops, is_undirected
import torch_sparse


def has_edge(edge_index, i, j):
  return bool(((edge_index[0]==i) & (edge_index[1]==j)).any())

def add_edge(edge_index, u, v):
  new_edge = torch.tensor([u,v], dtype=edge_index.dtype, device = edge_index.device)
  edge_index = torch.cat([edge_index, new_edge.unsqueeze(1)], dim=1)
  return edge_index

def get_neighbours(edge_index,i):
  neighbours = edge_index[1][edge_index[0]==i]
  return neighbours.tolist()


def dirichlet_energy_grad(edge_index, n, X, edge_weight=None, norm_type=None):
  edge_index, L = get_laplacian(edge_index, edge_weight, norm_type)
  LX = torch_sparse.spmm(edge_index, L, n, n, X)
  return LX

def bottleneck_score(LX_i, LX_j,eps=1e-08, alpha=1, beta=1):
  mag_i = torch.norm(LX_i, dim=-1)
  mag_j = torch.norm(LX_j, dim=-1)
  mag = mag_i + mag_j 
  grad_diff = torch.norm(LX_i-LX_j,dim=-1) / (mag + eps)

  score = alpha * grad_diff + beta * mag
  return score 

def evaluate_neighbourhood(X, edge_index, n, local_neigh, eta=0.1):
    LX = dirichlet_energy_grad(edge_index, n, X)
    one_step_diffusion = X - eta * LX
    LX_new = dirichlet_energy_grad(edge_index, n, one_step_diffusion)

    local_score = 0.0
    count = 0
    seen = set()
    local_set = set(local_neigh)
    for u in local_neigh:
        for v in get_neighbours(edge_index, u):
            if v in local_set:
                a, b = sorted((u, v))
                if (a, b) in seen:
                    continue
                seen.add((a, b))

                local_score += bottleneck_score(LX_new[u], LX_new[v])
                count += 1

    if count == 0:
        return 0.0

    return local_score / count

def energy_gradient_rewire(edge_index, n, X, k,eta=0.1):

  """
  edge_index : (2, m) indicating that there is an edge between node i and node j
  n : number of nodes
  X : (n,d) features
  k : number of edges add
  adj : (n,n) adjacency matrix
  """
  print(edge_index.shape)
  for idx in range(k):
    edge_index, _ = remove_self_loops(edge_index)
    LX = dirichlet_energy_grad(edge_index,n,X)
    X_prop = X - eta * LX
    LX_prop = dirichlet_energy_grad(edge_index, n, X_prop)
    score = bottleneck_score(LX_prop[edge_index[0]],LX_prop[edge_index[1]])
    # print("min:", score.min().item())
    # print("max:", score.max().item())
    # print("std:", score.std().item())
    bottleneck= torch.argmin(score,dim=-1)
    node_i = edge_index[0,bottleneck].item()
    node_j = edge_index[1,bottleneck].item()
    print("node_i",node_i)
    print("node_j",node_j)
    neigh_i = get_neighbours(edge_index,node_i)
    neigh_j = get_neighbours(edge_index,node_j)
    # print("neighboursi", len(neigh_i))
    # print("neighboursj",len(neigh_j))

    max_score = (-float('inf'),(-1,-1))
    print("old_bn", score.min())
    local_neigh = list(set(neigh_i + neigh_j + [node_i, node_j]))
    old_score = evaluate_neighbourhood(X,edge_index,n,local_neigh)
    for (u,v) in product(neigh_i,neigh_j):

        if has_edge(edge_index,u,v) or u==v:
          continue
        rewired_edge_index = add_edge(edge_index,u,v)
        new_score = evaluate_neighbourhood(X,rewired_edge_index,n,local_neigh)
        relief = new_score - old_score
        # print("old_score",old_score)
        # print("new_score",new_score)
        if max_score[0] < relief:
          max_score = (relief, (u,v))


    if max_score[1] != (-1,-1):
      print("edge_added")
      u,v = max_score[1]
      edge_index = add_edge(edge_index,u,v)
      edge_index = add_edge(edge_index,v,u)
      # LX_new = dirichlet_energy_grad(edge_index,n,X)
      # X = X - eta *LX_new

  edge_index, _ = add_remaining_self_loops(edge_index)
  print(edge_index.shape)
  return edge_index




