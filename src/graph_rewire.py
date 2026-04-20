## gradient-based rewiring
from itertools import product
import torch
from torch_geometric.utils import get_laplacian, remove_self_loops, add_remaining_self_loops, contains_self_loops, is_undirected
import torch_sparse
from collections import defaultdict
import numpy as np

def has_edge(edge_index, i, j):
  return bool(((edge_index[0]==i) & (edge_index[1]==j)).any())

def add_edge(edge_index, u, v):
  new_edge = torch.tensor([u,v], dtype=edge_index.dtype, device = edge_index.device)
  edge_index = torch.cat([edge_index, new_edge.unsqueeze(1)], dim=1)
  return edge_index

def get_neighbours(edge_index,i):
  neighbours = edge_index[1][edge_index[0]==i]
  return neighbours.tolist()

def build_neighbour_dict(edge_index):
    neigh = defaultdict(list)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        neigh[u].append(v)
    return neigh

def add_edge_to_neighbour_dict(neighbour_dict, u, v):
    new_dict = neighbour_dict.copy()
    new_dict[u] = neighbour_dict[u] + [v]
    new_dict[v] = neighbour_dict[v] + [u]
    return new_dict


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
def tango_bottleneck_score(edge_index, n, X, eta=0.1, eps=1e-8):
    LX = dirichlet_energy_grad(edge_index, n, X)
    X_smooth = X - eta * LX

    LX_smooth = dirichlet_energy_grad(edge_index, n, X_smooth)
    diff = torch.norm(
        LX_smooth[edge_index[0]] - LX_smooth[edge_index[1]], dim=-1
    )**2
    mag = torch.norm(LX_smooth[edge_index[0]], dim=-1)**2 + \
          torch.norm(LX_smooth[edge_index[1]], dim=-1)**2
    return diff + mag

# def bottleneck_score_v2(edge_index, n, X, eta=0.1, eps=1e-8):
#     """
#     Normalised gradient difference — matches tango_like_bottleneck 
#     structure but uses Dirichlet energy gradient instead of learned energy.
#     """
#     LX = dirichlet_energy_grad(edge_index, n, X)
#     X_smooth = X - eta * LX  # X_smooth[i] = weighted average of neighbourhood
    
#     i = edge_index[0]
#     j = edge_index[1]
    
#     # Score by normalised FEATURE difference after smoothing
#     # NOT gradient difference
#     diff = torch.norm(X_smooth[i] - X_smooth[j], dim=-1)
#     mag  = torch.norm(X_smooth[i], dim=-1) + torch.norm(X_smooth[j], dim=-1)
    
#     score = diff / (mag + eps)
#     return score

def bottleneck_score_v2(edge_index, n, X, eta=None, eps=1e-8):
    
    # Compute max degree for stable eta
    deg = torch.zeros(n, device=edge_index.device)
    deg.scatter_add_(
        0, edge_index[0], 
        torch.ones(edge_index.shape[1], device=edge_index.device)
    )
    d_max = deg.max().item()
    
    # Stable eta: must satisfy eta < 1/d_max for unnormalised Laplacian
    if eta is None:
        eta = 0.9 / (d_max + 1e-8)
    
    LX = dirichlet_energy_grad(edge_index, n, X)
    X_smooth = X - eta * LX
    
    i = edge_index[0]
    j = edge_index[1]
    
    diff = torch.norm(X_smooth[i] - X_smooth[j], dim=-1)
    mag  = torch.norm(X_smooth[i], dim=-1) + torch.norm(X_smooth[j], dim=-1)
    
    score = diff / (mag + eps)
    return score

def evaluate_neighbourhood(X, edge_index, n, local_neigh, neigh_dict, eta=0.1):
    
    # Compute scores for ALL edges once — returns (m,) tensor
    all_scores = bottleneck_score_v2(edge_index, n, X)
    
    # Build edge-to-index lookup for fast access
    # Maps (u,v) -> index in edge_index
    edge_to_idx = {}
    for idx in range(edge_index.shape[1]):
        u = edge_index[0, idx].item()
        v = edge_index[1, idx].item()
        edge_to_idx[(u, v)] = idx

    local_score = 0.0
    count = 0
    seen = set()
    local_set = set(local_neigh)

    for u in local_neigh:
        for v in neigh_dict.get(u, []):
            if v not in local_set:
                continue
            a, b = min(u, v), max(u, v)
            if (a, b) in seen:
                continue
            seen.add((a, b))

            # Look up score for this specific edge
            if (u, v) in edge_to_idx:
                idx = edge_to_idx[(u, v)]
                local_score += all_scores[idx].item()
                count += 1

    return local_score / count if count > 0 else 0.0
# def evaluate_neighbourhood(X, edge_index, n, local_neigh,neigh_dict, eta=0.1):
#     LX = dirichlet_energy_grad(edge_index, n, X)
#     one_step_diffusion = X - eta * LX
#     LX_new = dirichlet_energy_grad(edge_index, n, one_step_diffusion)

#     local_score = 0.0
#     count = 0
#     seen = set()
#     local_set = set(local_neigh)
#     for u in local_neigh:
#         for v in neigh_dict[u]:
#             if v in local_set:
#                 a, b = sorted((u, v))
#                 if (a, b) in seen:
#                     continue
#                 seen.add((a, b))

#                 local_score += bottleneck_score_v2(edge_index,n,X)
#                 count += 1

#     if count == 0:
#         return 0.0

#     return local_score / count

def energy_gradient_rewire(edge_index, n, X, k,eta=0.1):

  """
  edge_index : (2, m) indicating that there is an edge between node i and node j
  n : number of nodes
  X : (n,d) features
  k : number of edges add
  adj : (n,n) adjacency matrix
  """
  for idx in range(k):
    edge_index, _ = remove_self_loops(edge_index)
    LX = dirichlet_energy_grad(edge_index,n,X)
    X_prop = X - eta * LX
    LX_prop = dirichlet_energy_grad(edge_index, n, X_prop)
    score = bottleneck_score(LX_prop[edge_index[0]],LX_prop[edge_index[1]])
    # print("min:", score.min().item())
    # print("max:", score.max().item())
    # print("std:", score.std().item())
    bottleneck= torch.argmax(score,dim=-1)
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
        relief =  old_score - new_score
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
  
# def evaluate_neighbourhood_fast(LX_smooth, edge_index, local_neigh, 
#                                   candidate_u=None, candidate_v=None):
#     """
#     Fast approximation of neighbourhood score using pre-computed LX_smooth.
    
#     Instead of recomputing LX after inserting candidate edge (u,v),
#     we approximate the effect of adding (u,v) on the neighbourhood score.
    
#     The approximation: adding edge (u,v) pulls LX[u] toward LX[v] and vice versa.
#     We estimate the updated gradients at u and v after the edge addition,
#     then rescore the neighbourhood using these updated values.
    
#     Args:
#         LX_smooth: pre-computed (n,d) gradient matrix
#         edge_index: current edge index
#         local_neigh: list of nodes in local neighbourhood
#         candidate_u, candidate_v: candidate edge to evaluate (or None for baseline)
#     """
#     local_set = set(local_neigh)
    
#     # If evaluating a candidate edge, approximate updated LX at u and v
#     if candidate_u is not None and candidate_v is not None:
#         # Adding edge (u,v) means u now aggregates from v and vice versa
#         # Updated gradient at u: LX[u] gets pulled toward (LX[u] - LX[v])
#         # because u now has v as a neighbour, adding (x_u - x_v) to LX[u]
#         # This is a first-order approximation of what LX would be after insertion
        
#         deg_u = (edge_index[0] == candidate_u).sum().item()
#         deg_v = (edge_index[0] == candidate_v).sum().item()
        
#         # Approximate new LX at u and v after edge addition
#         # New LX[u] = old LX[u] + (1/(deg_u+1)) * (x_u - x_v) contribution
#         # But we are working with LX_smooth which already has diffusion applied
#         # So we approximate: the new gradient difference at (u,v) after addition
#         # tends toward zero (the edge is added to reduce the gradient difference)
#         # We weight the update by 1/(deg+1) — smaller effect for high-degree nodes
        
#         LX_approx = LX_smooth.clone()
        
#         weight_u = 1.0 / (deg_u + 1)
#         weight_v = 1.0 / (deg_v + 1)
        
#         diff = LX_smooth[candidate_u] - LX_smooth[candidate_v]
        
#         # Adding the edge reduces the gradient difference at this edge
#         LX_approx[candidate_u] = LX_smooth[candidate_u] - weight_u * diff
#         LX_approx[candidate_v] = LX_smooth[candidate_v] + weight_v * diff
#     else:
#         LX_approx = LX_smooth

#     # Score the neighbourhood using approximated gradients
#     local_score = 0.0
#     count = 0
#     seen = set()
    
#     for u in local_neigh:
#         neighbours = edge_index[1][edge_index[0] == u].tolist()
#         for v in neighbours:
#             if v in local_set:
#                 a, b = min(u,v), max(u,v)
#                 if (a, b) in seen:
#                     continue
#                 seen.add((a, b))
#                 local_score += bottleneck_score(
#                     LX_approx[u], LX_approx[v]
#                 ).item()
#                 count += 1
    
#     # Also include candidate edge in scoring if provided
#     if candidate_u is not None and candidate_v is not None:
#         if candidate_u in local_set and candidate_v in local_set:
#             local_score += bottleneck_score(
#                 LX_approx[candidate_u], 
#                 LX_approx[candidate_v]
#             ).item()
#             count += 1

#     return local_score / count if count > 0 else 0.0

def score_neighbourhood(LX, neighbour_dict, local_neigh):
    local_set = set(local_neigh)
    seen = set()
    total = 0.0
    count = 0

    for u in local_neigh:
        for v in neighbour_dict[u]:
            if v in local_set:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                total += bottleneck_score(LX[u], LX[v]).item()
                count += 1

    return total / count if count > 0 else 0.0

def evaluate_neighbourhood_fast(LX_smooth, neighbour_dict, local_neigh, candidate_u, candidate_v):

    deg_u = len(neighbour_dict[candidate_u])
    deg_v = len(neighbour_dict[candidate_v])

    LX_approx = LX_smooth.clone()
    diff = LX_smooth[candidate_u] - LX_smooth[candidate_v]

    weight_u = 1.0 / (deg_u + 1)
    weight_v = 1.0 / (deg_v + 1)

    LX_approx[candidate_u] = LX_smooth[candidate_u] - weight_u * diff
    LX_approx[candidate_v] = LX_smooth[candidate_v] + weight_v * diff

    augmented_dict = add_edge_to_neighbour_dict(neighbour_dict, candidate_u, candidate_v)

    return score_neighbourhood(LX_approx, augmented_dict, local_neigh)

def energy_gradient_rewire_hybrid(edge_index, n, X, k, eta=0.1, 
                                   recompute_every=1):
    """
    Hybrid: exact recomputation every recompute_every steps,
    approximate updates in between.
    
    recompute_every=1 recovers your exact dynamic method (slow)
    recompute_every=k recovers pure fast batch (fastest, least accurate)
    recompute_every=5 is a good middle ground
    """
    edge_index, _ = remove_self_loops(edge_index)
    edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    neigh_dict = build_neighbour_dict(edge_index)
    
    # Initial exact computation
    LX = dirichlet_energy_grad(edge_index, n, X)
    X_smooth = X - eta * LX
    LX_smooth = dirichlet_energy_grad(edge_index, n, X_smooth)
    
    for idx in range(k):
        
        # Recompute exactly every recompute_every steps
        # This resets accumulated approximation error
        if idx > 0 and idx % recompute_every == 0:
            LX = dirichlet_energy_grad(edge_index, n, X)
            X_smooth = X - eta * LX
            LX_smooth = dirichlet_energy_grad(edge_index, n, X_smooth)
        
        # Score all edges using current LX_smooth (exact or approximate)
        # score = bottleneck_score(
        #     LX_smooth[edge_index[0]],
        #     LX_smooth[edge_index[1]]
        # )
        score= bottleneck_score_v2(edge_index,n,X)
        # score= tango_bottleneck_score(edge_index,n,X)

        # cv = score.std() / score.mean()
        # cv2 = score_2.std() / score_2.mean()
        # print(f"Coefficient of variation1: {cv:.3f}")
        # print(f"Coefficient of variation2: {cv2:.3f}")

        
        
        # Find worst bottleneck — skip if neighbourhood saturated
        sorted_bottlenecks = torch.argsort(score, descending=True)
        
        found = False
        for bottleneck_idx in sorted_bottlenecks:
            node_i = edge_index[0, bottleneck_idx].item()
            node_j = edge_index[1, bottleneck_idx].item()
            
            neigh_i = neigh_dict[node_i]
            neigh_j = neigh_dict[node_j]
            
            valid_candidates = [
                (u, v) for (u, v) in product(neigh_i, neigh_j)
                if (u, v) not in edge_set
                and (v, u) not in edge_set
                and u != v
            ]
            
            if not valid_candidates:
                continue
            
            local_neigh = list(set(neigh_i + neigh_j + [node_i, node_j]))
            old_score = evaluate_neighbourhood(X,edge_index,n,local_neigh,neigh_dict)
            
            best = (-float('inf'), (-1, -1))
            for (u, v) in valid_candidates:
                rewired_edge_index = add_edge(edge_index, u, v)
                rewired_edge_index = add_edge(rewired_edge_index,v,u)
                rewired_neigh_dict = add_edge_to_neighbour_dict(neigh_dict,u,v)
                new_score = evaluate_neighbourhood(
                    X, rewired_edge_index,n, local_neigh,neigh_dict
                )
                relief = old_score - new_score
                if best[0] < relief:
                    best = (relief, (u, v))
            
            if best[1] != (-1, -1):
                print("edge_added")
                u, v = best[1]
                edge_index = add_edge(edge_index, u, v)
                edge_index = add_edge(edge_index, v, u)
                edge_set.add((u, v))
                edge_set.add((v, u))
                neigh_dict[u].append(v)
                neigh_dict[v].append(u)
                
                # Approximate update to LX_smooth for next iteration
                # Avoids full recomputation between exact checkpoints
                deg_u =len(neigh_dict[u])
                deg_v = len(neigh_dict[v])
                diff = LX_smooth[u] - LX_smooth[v]
                LX_smooth[u] = LX_smooth[u] - (1.0/(deg_u+1)) * diff
                LX_smooth[v] = LX_smooth[v] + (1.0/(deg_v+1)) * diff
                
                found = True
                break
        
        if not found:
            print(f"Early stop at iteration {idx}: no valid candidates")
            break
    
    edge_index, _ = add_remaining_self_loops(edge_index)
    return edge_index




