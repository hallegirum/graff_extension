import time
import numpy as np
import torch

from GNN import GNN
from data import get_dataset, set_train_val_test_split
from heterophilic import get_fixed_splits
from data_synth_hetero import get_pyg_syn_cora
from utils import calc_stats, set_seed, add_labels, get_label_masks, print_model_params
from graff_params import get_args, load_best_params, tf_ablation_args
from graph_rewire import energy_gradient_rewire, energy_gradient_rewire_hybrid
from fosra_baseline import edge_rewire

def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def train(model, optimizer, data, pos_encoding=None):
    lf = torch.nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    feat = data.x
    if model.opt['use_labels']:
        train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

        feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
    else:
        train_pred_idx = data.train_mask

    out = model(feat, pos_encoding)

    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])

    model.fm.update(model.getNFE())
    model.resetNFE()
    loss.backward()
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()

    return loss.item()


@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
    model.eval()
    feat = data.x
    if model.opt['use_labels']:
        feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
    logits, accs = model(feat, pos_encoding), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def main(cmd_opt):

    opt = load_best_params(cmd_opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt['device'] = device
    rand_seed = np.random.randint(3, 10000)
    set_seed(rand_seed)
    opt['rand_seed'] = rand_seed
    if not opt['undirected'] and opt['dataset'] in ['texas', 'wisconsin', 'cornell', 'cornell_old', 'squirrel', 'chameleon']:
        opt['not_lcc'] = False # set to false when using opt['undirected'] = False

    dataset = get_dataset(opt, '../data', opt['not_lcc'])
    base_data = dataset[0].clone()
    pos_encoding = None
    this_test = test
    results = []
    k_values = [20,50,100,200,500]
    all_results = { 
      'none':  {k: [] for k in k_values},
      'fosr':  {k: [] for k in k_values},
      'diffusion':  {k: [] for k in k_values},
    }

    for i,k in enumerate(k_values):
      print(f"Running k={k}")
      for method in ['none','fosr','diffusion']:
        print("method")
        for rep in range(opt['num_splits']):
            if method == 'none' and i >0:
              continue
            data = base_data.clone()
            print(f"rep {rep}")
            if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
                dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data,
                                                        num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)
            if opt['dataset']== "Roman-empire":
              data.train_mask = data.train_mask[:, rep]
              data.val_mask   = data.val_mask[:, rep]
              data.test_mask  = data.test_mask[:, rep]

            
            data = data.to(device)

            if method == 'fosr':
              numpy_edge_index = data.edge_index.detach().cpu().numpy()
              fosr_edge_index, _, _ = edge_rewire(numpy_edge_index,num_iterations=k)
              data.edge_index = torch.tensor(fosr_edge_index).to(device)

            if method == "diffusion":
              rewired_edge_index = energy_gradient_rewire(data.edge_index,data.num_nodes, data.x,k)
              data.edge_index = rewired_edge_index

            dataset._data = data
            model = GNN(opt, dataset, device).to(device)

            parameters = [p for p in model.parameters() if p.requires_grad]
            print(opt)
            print_model_params(model)
            optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
            best_time = best_epoch = train_acc = val_acc = test_acc = 0
            if opt['patience'] is not None:
                patience_count = 0
            for epoch in range(1, opt['epoch']):
                start_time = time.time()
                loss = train(model, optimizer, data, pos_encoding)
                tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)

                best_time = opt['time']
                if tmp_val_acc > val_acc:
                    best_epoch = epoch
                    train_acc = tmp_train_acc
                    val_acc = tmp_val_acc
                    test_acc = tmp_test_acc
                    best_time = opt['time']
                    patience_count = 0
                else:
                    patience_count += 1
                print(f"Epoch: {epoch}, Runtime: {time.time() - start_time:.3f}, Loss: {loss:.3f}, "
                      f"forward nfe {model.fm.sum}, backward nfe {model.bm.sum}, "
                      f"tmp_train: {tmp_train_acc:.4f}, tmp_val: {tmp_val_acc:.4f}, tmp_test: {tmp_test_acc:.4f}, "
                      f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best time: {best_time:.4f}")

                if np.isnan(loss):
                    break
                if opt['patience'] is not None:
                    if patience_count >= opt['patience']:
                        break
            print(
                f"best val accuracy {val_acc:.3f} with test accuracy {test_acc:.3f} at epoch {best_epoch} and best time {best_time:2f}")

            # stats = calc_stats(model, data)
            # RQX0, RQXN, ev_max, ev_min, ev_av, ev_std = stats['RQX0'], stats['RQXN'], stats['ev_max'], stats['ev_min'], stats['ev_av'], stats['ev_std']
            # print(f"RQX0, RQXN, ev_max, ev_min, ev_av, l_std: {RQX0, RQXN, ev_max, ev_min, ev_av, ev_std}")

            # if opt['num_splits'] > 1:
            #     results.append([test_acc, val_acc, train_acc, RQX0, RQXN, ev_max, ev_min, ev_av, ev_std])

        # if opt['num_splits'] > 1:
        #     test_acc_mean, val_acc_mean, train_acc_mean, RQX0, RQXN, ev_max, ev_min, ev_av, ev_std = np.mean(results,
        #                                                                                                     axis=0)
        #     test_acc_mean = test_acc_mean * 100
        #     val_acc_mean = val_acc_mean * 100
        #     train_acc_mean = train_acc_mean * 100
        #     test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

        #     results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
        #               'test_acc_std': test_acc_std,
        #               'RQX0': RQX0, 'RQXN': RQXN, 'ev_max': ev_max, 'ev_min': ev_min, 'ev_av': ev_av, 'ev_std': ev_std}
        # else:
        #     results = {'test_acc': test_acc, 'val_acc': val_acc, 'train_acc': train_acc,
        #               'RQX0': RQX0, 'RQXN': RQXN, 'ev_max': ev_max, 'ev_min': ev_min, 'ev_av': ev_av, 'ev_std': ev_std}
        
        all_results[method][k].append(test_acc)
        print(f"  rep={rep}: test_acc={test_acc:.4f}")

    # --- Summarise results ---
    print(f"\n{'='*60}")
    print("SUMMARY: Test Accuracy Mean ± Std")
    print(f"{'='*60}")
    print(f"{'k':<8} {'No rewiring':<20} {'FoSR':<20} {'Ours':<20}")
    print("-" * 68)

    summary = {}
    for k in k_values:
        row = {}
        for method in ['none', 'fosr', 'ours']:
            accs = all_results[method][k]
            mean = np.mean(accs) * 100
            std  = np.std(accs) * 100
            row[method] = (mean, std)

        summary[k] = row
        print(
            f"{k:<8} "
            f"{row['none'][0]:.2f}±{row['none'][1]:.2f}{'':8}"
            f"{row['fosr'][0]:.2f}±{row['fosr'][1]:.2f}{'':8}"
            f"{row['ours'][0]:.2f}±{row['ours'][1]:.2f}"
        )

    plot_k_ablation(summary,k_values)
    return summary, all_results

def plot_k_ablation(summary, k_values):
    """
    Plot test accuracy vs k for all three methods.
    Called after main_k_ablation.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, label, color, marker in [
        ('none', 'No rewiring', 'gray',   '--'),
        ('fosr', 'FoSR',        'blue',   'o-'),
        ('ours', 'Ours',        'green',  's-'),
    ]:
        means = [summary[k][method][0] for k in k_values]
        stds  = [summary[k][method][1] for k in k_values]

        if method == 'none':
            ax.axhline(y=means[0], color=color, linestyle='--',
                      label=label, alpha=0.7)
        else:
            ax.errorbar(
                k_values, means, yerr=stds,
                fmt=marker, color=color, label=label,
                capsize=4, linewidth=2, markersize=6
            )

    ax.set_xlabel('k (edges added)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Roman-Empire: Accuracy vs Rewiring Budget', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roman_empire_k_ablation.png', dpi=150)
    plt.show()
    return fig



if __name__ == '__main__':
    opt = get_args()
    opt = tf_ablation_args(opt)
    # if not opt['wandb_sweep']:
    #     opt = graff_run_params(opt)
    main(opt)