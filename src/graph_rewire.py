## gradient-based rewiring
from itertools import product
import torch
from torch_geometric.utils import get_laplacian
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


def energy_gradient_rewire(edge_index, n, X, k):

  """
  edge_index : (2, m) indicating that there is an edge between node i and node j
  n : number of nodes
  X : (n,d) features
  k : number of edges add
  adj : (n,n) adjacency matrix
  """

  for idx in range(k):
    LX = dirichlet_energy_grad(edge_index,n,X)
    diff_energy = (LX[edge_index[0]] - LX[edge_index[1]]).pow(2).sum(dim=-1)
    print("min:", diff_energy.min().item())
    print("max:", diff_energy.max().item())
    print("mean:", diff_energy.mean().item())
    bottleneck= torch.argmin(diff_energy,dim=-1)
    node_i = edge_index[0,bottleneck]
    node_j = edge_index[1,bottleneck]
    print("node_i",node_i)
    print("node_j",node_j)
    neigh_i = get_neighbours(edge_index,node_i)
    neigh_j = get_neighbours(edge_index,node_j)
    print("neighboursi", len(neigh_i))
    print("neighboursj",len(neigh_j))

    max_score = (0,(-1,-1))
    for (u,v) in product(neigh_i,neigh_j):

        if has_edge(edge_index,u,v):
          continue
        rewired_edge_index = add_edge(edge_index,u,v)
        dir_energy = dirichlet_energy(rewired_edge_index,n,X)
        dir_energy_ij = (dir_energy[node_i] - dir_energy[node_j]).pow(2).sum(dim=-1)
        if max_score[0] < dir_energy_ij:
          max_score = (dir_energy_ij, (u,v))


    if max_score[1] != (-1,-1):
      u,v = max_score[1]
      edge_index = add_edge(edge_index,u,v)


  return edge_index





