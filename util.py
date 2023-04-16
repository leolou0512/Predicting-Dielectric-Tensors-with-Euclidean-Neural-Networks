import math
import time
from typing import Dict, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from tqdm import tqdm
import matplotlib as mpl
import pandas as pd
from e3nn.io import CartesianTensor
from sklearn.model_selection import train_test_split
import e3nn
import plotly.graph_objects as go
from ase.visualize.plot import plot_atoms
from e3nn import o3
from e3nn.io import SphericalTensor
from IPython.display import HTML
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import networkx as nx
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import (Convolution, smooth_cutoff,
                                             tp_path_exists)
from numpy import array
from scipy.stats import gaussian_kde
from torch_cluster import radius_graph
from torch_geometric.data import Data

from sklearn.metrics import r2_score


bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
palette = ['#2876B2', '#F39957', '#67C7C2', '#C86646']
datasets = ['train', 'valid', 'test']
colors = dict(zip(datasets, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])

##############################################################################################################
def load_data(filename, ct):
    # load data from a csv file and derive formula and species columns from structure
    df = pd.read_csv(filename)
    df['structure'] = df['structure'].map(lambda x: Atoms.fromdict(eval(x)))
    df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
    df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    
    cart_tensors = np.array([eval(x) for x in df["dielectric_tensor"]])
    represention = ct.from_cartesian(torch.from_numpy(cart_tensors))
    df["dielectric_irreps"] = represention.tolist()
    
    species = sorted(list(set(df['species'].sum())))
    return df, species

def build_data(entry, type_encoding, type_onehot, am_onehot, r_max=5.):
    symbols = list(entry.structure.symbols).copy()
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.structure, cutoff=r_max, self_interaction=True)
    
    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(np.int64(edge_src))]
    edge_vec = (positions[torch.from_numpy(np.int64(edge_dst))]
                - positions[torch.from_numpy(np.int64(edge_src))]
                + torch.einsum('ni,nij->nj', torch.tensor(np.int64(edge_shift), dtype=default_dtype), lattice[edge_batch]))

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    
    data = tg.data.Data(
        pos=positions, lattice=lattice, symbol=symbols,
        x=am_onehot[[type_encoding[specie] for specie in symbols]],   # atomic mass (node feature)
        z=type_onehot[[type_encoding[specie] for specie in symbols]], # atom type (node attribute)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, edge_len=edge_len,
        dielectric_irreps=torch.tensor(entry.dielectric_irreps).unsqueeze(0) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    )
    
    return data

def train_valid_test_split(df, species, valid_size, test_size, seed=12, plot=False):
    # perform an element-balanced train/valid/test split
    print('split train/dev ...')
    dev_size = valid_size + test_size
    stats = get_element_statistics(df, species)
    idx_train, idx_dev = split_data(stats, dev_size, seed)
    
    print('split valid/test ...')
    stats_dev = get_element_statistics(df.iloc[idx_dev], species)
    idx_valid, idx_test = split_data(stats_dev, test_size/dev_size, seed)
    idx_train += df[~df.index.isin(idx_train + idx_valid + idx_test)].index.tolist()

    print('number of training examples:', len(idx_train))
    print('number of validation examples:', len(idx_valid))
    print('number of testing examples:', len(idx_test))
    print('total number of examples:', len(idx_train + idx_valid + idx_test))
    assert len(set.intersection(*map(set, [idx_train, idx_valid, idx_test]))) == 0

    if plot:
        # plot element representation in each dataset
        stats['train'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_train)))
        stats['valid'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_valid)))
        stats['test'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_test)))
        stats = stats.sort_values('symbol')

        fig, ax = plt.subplots(2,1, figsize=(14,7))
        b0, b1 = 0., 0.
        for i, dataset in enumerate(datasets):
            split_subplot(ax[0], stats[:len(stats)//2], species[:len(stats)//2], dataset, bottom=b0, legend=True)
            split_subplot(ax[1], stats[len(stats)//2:], species[len(stats)//2:], dataset, bottom=b1)

            b0 += stats.iloc[:len(stats)//2][dataset].values
            b1 += stats.iloc[len(stats)//2:][dataset].values

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1)

    return idx_train, idx_valid, idx_test

def get_neighbors(df, idx):
    n = []
    for entry in df.iloc[idx].itertuples():
        N = entry.data.pos.shape[0]
        for i in range(N):
            n.append(len((entry.data.edge_index[0] == i).nonzero()))
    return np.array(n)

def train(model, optimizer, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, run_name,
          max_iter=101, scheduler=None, device="cpu"):
    model.to(device)

    checkpoint_generator = loglinspace(0, 1)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    
    # Early Stopping parameters
    best_loss = 9999999.
    best_weights = None
    delta = 0.001
    patience = 120
    
    try: model.load_state_dict(torch.load(run_name + '.torch')['state'])
    except:
        results = {}
        history = []
        s0 = 0
    else:
        results = torch.load(run_name + '.torch')
        history = results['history']
        s0 = history[-1]['step'] + 1

    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.
        loss_cumulative_mae = 0.
        
        for j, d in tqdm(enumerate(dataloader_train), total=len(dataloader_train), bar_format=bar_format):
            d.to(device)
            output = model(d)
            loss = loss_fn(output, d.dielectric_irreps).cpu()
            loss_mae = loss_fn_mae(output, d.dielectric_irreps).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        wall = end_time - start_time

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            valid_avg_loss = evaluate(model, dataloader_valid, loss_fn, loss_fn_mae, device)
            train_avg_loss = evaluate(model, dataloader_train, loss_fn, loss_fn_mae, device)

            history.append({
                'step': s0 + step,
                'wall': wall,
                'batch': {
                    'loss': loss.item(),
                    'mean_abs': loss_mae.item(),
                },
                'valid': {
                    'loss': valid_avg_loss[0],
                    'mean_abs': valid_avg_loss[1],
                },
                'train': {
                    'loss': train_avg_loss[0],
                    'mean_abs': train_avg_loss[1],
                },
            })

            results = {
                'history': history,
                'state': model.state_dict()
            }
            
            
            # Early Finish and restore best weights
            if valid_avg_loss[0] < best_loss - delta:
                best_loss = valid_avg_loss[0]
                best_weights = model.state_dict()
                patience_counter = 0
                torch.save(best_weights, "best_" + run_name + '.torch')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("EARLY STOPPING AT EPOCH" + str(step))
                    break
                    

            print(f"Iteration {step+1:4d}   " +
                  f"train loss = {train_avg_loss[0]:8.4f}   " +
                  f"valid loss = {valid_avg_loss[0]:8.4f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

            with open(run_name + '.torch', 'wb') as f:
                torch.save(results, f)

        if scheduler is not None:
            scheduler.step()
            
        if step == range(max_iter)[-1]:
            model.load_state_dict(best_weights)
            with open(run_name + '.torch', 'wb') as f:
                torch.save(results, f)

def evaluate(model, dataloader, loss_fn, loss_fn_mae, device):
    model.eval()
    loss_cumulative = 0.
    loss_cumulative_mae = 0.
    start_time = time.time()
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d)
            loss = loss_fn(output, d.dielectric_irreps).cpu()
            loss_mae = loss_fn_mae(output, d.dielectric_irreps).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()
    return loss_cumulative/len(dataloader), loss_cumulative_mae/len(dataloader)

def get_element_statistics(df, species):    
    # create dictionary indexed by element names storing index of samples containing given element
    species_dict = {k: [] for k in species}
    for entry in df.itertuples():
        for specie in entry.species:
            species_dict[specie].append(entry.Index)

    # create dataframe of element statistics
    stats = pd.DataFrame({'symbol': species})
    stats['data'] = stats['symbol'].astype('object')
    for specie in species:
        stats.at[stats.index[stats['symbol'] == specie].values[0], 'data'] = species_dict[specie]
    stats['count'] = stats['data'].apply(len)

    return stats

def split_data(df, test_size, seed):
    # initialize output arrays
    idx_train, idx_test = [], []
    
    # remove empty examples
    df = df[df['data'].str.len()>0]
    
    # sort df in order of fewest to most examples
    df = df.sort_values('count')
    
    for _, entry in tqdm(df.iterrows(), total=len(df), bar_format=bar_format):
        df_specie = entry.to_frame().T.explode('data')

        try:
            idx_train_s, idx_test_s = train_test_split(df_specie['data'].values, test_size=test_size,
                                                       random_state=seed)
        except:
            # too few examples to perform split - these examples will be assigned based on other constituent elements
            # (assuming not elemental examples)
            pass

        else:
            # add new examples that do not exist in previous lists
            idx_train += [k for k in idx_train_s if k not in idx_train + idx_test]
            idx_test += [k for k in idx_test_s if k not in idx_train + idx_test]
    
    return idx_train, idx_test

def element_representation(x, idx):
    # get fraction of samples containing given element in dataset
    return len([k for k in x if k in idx])/len(x)

def split_subplot(ax, df, species, dataset, bottom=0., legend=False):    
    # plot element representation
    width = 0.4
    color = [int(colors[dataset].lstrip('#')[i:i+2], 16)/255. for i in (0,2,4)]
    bx = np.arange(len(species))
        
    ax.bar(bx, df[dataset], width, fc=color+[0.7], ec=color, lw=1.5, bottom=bottom, label=dataset)
        
    ax.set_xticks(bx)
    ax.set_xticklabels(species)
    ax.tick_params(direction='in', length=0, width=1)
    ax.set_ylim(top=1.18)
    if legend: ax.legend(frameon=False, ncol=3, loc='upper left')

def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))

class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x
    
class Network(torch.nn.Module):
    r"""equivariant neural network
    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features
    irreps_hidden : `e3nn.o3.Irreps`
        representation of the hidden features
    irreps_out : `e3nn.o3.Irreps`
        representation of the output features
    irreps_node_attr : `e3nn.o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes
    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials
    layers : int
        number of gates (non linearities)
    max_radius : float
        maximum radius for the convolution
    number_of_basis : int
        number of basis on which the edge length are projected
    radial_layers : int
        number of hidden layers in the radial fully connected network
    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network
    num_neighbors : float
        typical number of nodes at a distance ``max_radius``
    num_nodes : float
        typical number of nodes in a graph
    """
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_node_attr,
        layers,
        mul,
        lmax,
        max_radius,
        number_of_basis=10,
        radial_layers=1,
        radial_neurons=100,
        num_neighbors=1.,
        num_nodes=1.,
        reduce_output=True,
    ) -> None:
        super().__init__()
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
            irreps = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
        )

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        if 'edge_index' in data:
            edge_src = data['edge_index'][0]  # edge source
            edge_dst = data['edge_index'][1]  # edge destination
            edge_vec = data['edge_vec']
        
        else:
            edge_index = radius_graph(data['pos'], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]

        return batch, edge_src, edge_dst, edge_vec

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """evaluate the network
        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        batch, edge_src, edge_dst, edge_vec = self.preprocess(data)
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='gaussian',
            cutoff=False
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and 'x' in data:
            assert self.irreps_in is not None
            x = data['x']
        else:
            assert self.irreps_in is None
            x = data['pos'].new_ones((data['pos'].shape[0], 1))

        if self.input_has_node_attr and 'z' in data:
            z = data['z']
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data['pos'].new_ones((data['pos'].shape[0], 1))

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return x
        
def plot_example(df, i=12, label_edges=False):
    # plot an example crystal structure and graph
    entry = df.iloc[i]['data']

    # get graph with node and edge attributes
    g = tg.utils.to_networkx(entry, node_attrs=['symbol'], edge_attrs=['edge_len'], to_undirected=True)

    # remove self-loop edges for plotting
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    node_labels = dict(zip([k[0] for k in g.nodes.data()], [k[1]['symbol'] for k in g.nodes.data()]))
    edge_labels = dict(zip([(k[0], k[1]) for k in g.edges.data()], [k[2]['edge_len'] for k in g.edges.data()]))

    # project positions of nodes to 2D for plotting
    pos = dict(zip(list(g.nodes), [np.roll(k,2)[:-1][::-1] for k in entry.pos.numpy()]))

    # plot unit cell
    fig, ax = plt.subplots(1,2, figsize=(14,10), gridspec_kw={'width_ratios': [2,3]})
    atoms = Atoms(symbols=entry.symbol, positions=entry.pos.numpy(), cell=entry.lattice.squeeze().numpy(), pbc=True)
    symbols = np.unique(entry.symbol)
    z = dict(zip(symbols, range(len(symbols))))
    norm = plt.Normalize(vmin=0, vmax=len(symbols)-1)
    color = [mpl.colors.to_hex(k) for k in cmap(norm([z[j] for j in entry.symbol]))]
    plot_atoms(atoms, ax[0], radii=0.25, colors=color, rotation=('0x,90y,0z'))

    # plot graph
    nx.draw_networkx(g, ax=ax[1], labels=node_labels, pos=pos, node_size=500, node_color=color,
                     edge_color='gray')
    
    if label_edges:
        nx.draw_networkx_edge_labels(g, ax=ax[1], edge_labels=edge_labels, pos=pos, label_pos=0.5)
    
    # format axes
    ax[0].set_xlabel(r'$x_1\ (\AA)$')
    ax[0].set_ylabel(r'$x_2\ (\AA)$')
    ax[0].set_title('Crystal structure', fontsize=fontsize)
    ax[1].set_aspect('equal')
    ax[1].axis('off')
    ax[1].set_title('Crystal graph', fontsize=fontsize)
    pad = np.array([-0.5, 0.5])
    ax[1].set_xlim(np.array(ax[1].get_xlim()) + pad)
    ax[1].set_ylim(np.array(ax[1].get_ylim()) + pad)
    fig.subplots_adjust(wspace=0.4)

        
def plot_adjusted(dfv, ax, x_axis, y_axis, xmax=15, xmin=0):
    x = np.array([x for x in dfv[x_axis]])
    y = np.array([x for x in dfv[y_axis]])
    
    mae = np.abs(x - y).mean()
    mse = np.power(x - y, 2).mean()
    r2 = r2_score(x, y)
    
    ax.scatter(x, y, s=1)
    ax.set(xlim=(xmin, xmax), ylim=(xmin, xmax), xlabel="Actual")
    ax.text(0.1, 0.9, f"MAE: {mae:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.text(0.1, 0.8, f"MSE: {mse:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.text(0.1, 0.7, f"R$^2$: {r2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.plot([0, xmax], [0, xmax], ls="--", c="grey")
    
    
def plot(dfv, ax, prop="dielectric_scalar", xmax=15, xmin=0):
    # x = np.array([x[0] for x in dfv["dielectric_tensor"]])
    # y = np.array([x[0][0] for x in dfv["dielectric_tensor_pred"]])
    x = np.array([x for x in dfv[f"{prop}"]])
    y = np.array([x for x in dfv[f"{prop}_pred"]])
    
    mae = np.abs(x - y).mean()
    mse = np.power(x - y, 2).mean()
    r2 = r2_score(x, y)
    
    ax.scatter(x, y, s=1)
    ax.set(xlim=(xmin, xmax), ylim=(xmin, xmax), xlabel="Actual")
    ax.text(0.1, 0.9, f"MAE: {mae:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.text(0.1, 0.8, f"MSE: {mse:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.text(0.1, 0.7, f"R$^2$: {r2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.plot([0, xmax], [0, xmax], ls="--", c="grey")
    
    
def print_info(y, df):
    if y in idx_train:
        print("Training Set")
    if y in idx_valid:
        print("Validation Set")
    if y in idx_test:
        print("Test Set")
        
    print("Entry: ", y)
    print(df["structure"].loc[y])
    print("-------------------\n")
    print("Dielectric")
    print("  Actual\n")
    print(np.array(eval(df["dielectric_tensor"].loc[y])).round(4))
    print("Actual irreps: ")
    print(np.array(df['dielectric_tensor_irreps'].loc[y]).round(4))
    print("\n")

    print("  Predicted\n")
    print(np.array(df["dielectric_tensor_pred_cart"].loc[y], dtype=float).round(4))
    print("Pred irreps: ")
    print(np.array(df['dielectric_tensor_pred'].loc[y]).round(4))

    print("\nAnistropy")
    print("  Actual:", df["dielectric_ar"].loc[y])
    print("  Pred:", df["dielectric_ar_pred"].loc[y])
    print("  Adjusted:", df["dielectric_ar_pred_adjusted"].loc[y])
    print("\nCell Parameter")
    print(df.loc[y]["structure"].cell.array)
    print("-------------------\n")
    
    return
##############################################################################################################