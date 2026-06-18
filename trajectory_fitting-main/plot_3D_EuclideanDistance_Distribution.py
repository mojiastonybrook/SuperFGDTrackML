import torch
import os
import math
import numpy as np
from modules import *
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from torch import Tensor

import matplotlib.pyplot as plt

# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

# RNN related functions
from fitting_algorithms import FittingRNN
# function to collate data samples into batched tensors for RNN
def collate_fn_rnn(batch, padding_value=0):
    (xx, yy) = zip(*batch)

    # remove None
    xx = [x for x in xx if x is not None]
    yy = [y for y in yy if y is not None]

    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=padding_value)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=padding_value)

    return xx_pad, yy_pad, x_lens, y_lens

# Transformer related functions
from fitting_algorithms import FittingTransformer
def create_mask_src(src):
    src_seq_len = src.shape[0]

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_padding_mask = (src[:, :, 0] == PAD_IDX).transpose(0, 1)

    return src_mask, src_padding_mask
# function to collate data samples into batched tensors
def collate_fn(batch):
    (xx, yy) = zip(*batch)

    xx = [x for x in xx]
    yy = [y for y in yy]

    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=False, padding_value=PAD_IDX)
    yy_pad = pad_sequence(yy, batch_first=False, padding_value=PAD_IDX)

    return xx_pad, yy_pad, x_lens, y_lens

# 3D Euclidean distance
def calculate_3d_distance(truth: Tensor, prediction: Tensor):
    # M.Jia: translate normalized results into real node positions
    prediction *= (DETECTOR_RANGES[0][1] - DETECTOR_RANGES[0][0])
    truth *= (DETECTOR_RANGES[0][1] - DETECTOR_RANGES[0][0])
    for i in range(3):
        prediction[:,:,i] += DETECTOR_RANGES[i][0]
        truth[:,:,i] += DETECTOR_RANGES[i][0]

    #
    residual = (truth-prediction).numpy()
    #
    distance = np.linalg.norm(residual, axis=2).reshape(-1)

    return distance

# make a distribution of distances
def calculate_density(
    distances: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate normalized histogram density.

    Important:
    The normalization uses the total number of valid node distances,
    including distances outside the plotted x-range.
    """

    counts, _ = np.histogram(
        distances,
        bins=bin_edges,
    )

    bin_widths = np.diff(bin_edges)

    density = counts / (distances.size * bin_widths)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    density = density.astype(float)
    density[density == 0.0] = np.nan

    return bin_centers, density

def collect_distances_over_samples(dataloader:DataLoader, model:dict):
    all_distances = []
    
    for i,data in enumerate(dataloader):
        src, tgt, src_len, tgt_len = data
        src = src.to(DEVICE)
    
        # run model
        if model['name'] == 'RNN':
           pred = model['networks'](src, src_len)
        elif model['name'] == "Transformer":
           pred = model['networks'](src, src_mask, src_padding_mask)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        # M.Jia: each sample has its own length of track, so there is a potential issue if two and more samples are rendered in one batch;
        #        add padding token to samples with shorter track length; the padding ones should be masked out 
        mask = (tgt != PAD_IDX).float()
        tgt = tgt * mask
        pred = pred * mask

        # calculate 3d distance
        distances = calculate_3d_distance(tgt, pred)
        all_distances.append(distances)

    pooled_distances = np.concatenate(all_distances)
    return pooled_distances

# plotting
def make_distance_distribution_plot(
    model: dict,
    output_file: str
) -> None:
    """

    """
    fig, ax = plt.subplots(figsize=(7.5, 5.3))

    total_nodes_by_model = {}

    # add rnn results
    #loader_rnn = DataLoader(dataset, collate_fn=collate_fn_rnn, batch_size=1,
    #                        num_workers=0, shuffle=False)
    
    pooled_distances = collect_distances_over_samples(
                       dataloader=model['loader'],
                       model = model 
                       )
    
    total_nodes_by_model[model['name']] = pooled_distances.size
    
    #X_MIN = np.min(pooled_distances_rnn)
    X_MIN = 0.0
    BIN_WIDTH = 0.1
    #X_MAX = np.max(pooled_distances_rnn)
    X_MAX =20.0
    
    bin_edges = np.arange(
        X_MIN,
        X_MAX + BIN_WIDTH,
        BIN_WIDTH,
    )
    bin_centers, density = calculate_density(distances=pooled_distances,bin_edges=bin_edges)
    ax.plot(
            bin_centers,
            density,
            linestyle=LINE_STYLES.get(model['name'], "-"),
            linewidth=1.5,
            label=model['name'],
            )
    
    # make plots
    unique_total_nodes = set(total_nodes_by_model.values())

    if len(unique_total_nodes) == 1:
        legend_title = f"N={next(iter(unique_total_nodes)):,} true nodes"
    else:
        legend_title = "Different N per model"

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_yscale("log")

    ax.set_xlabel("euclidean distance: true-pred [mm]")
    ax.set_ylabel("density (log-scale)")

    ax.grid(
        visible=True,
        which="major",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    ax.legend(
        title=legend_title,
        loc="upper right",
        frameon=True,
    )

    ax.text(
        -0.20,
        1.02,
        "a",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
    )

    fig.tight_layout()
    
    #plt.show()
    
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    OUTPUT_FILE = "distance_distribution_many_samples.png"
    #data_dir = "/storage/shared/chiaki/superFGD/testing"  
    data_dir = "/storage/shared/chiaki/superFGD/training"  
    LINE_STYLES = {
        "Transformer": "-",
        "RNN": ":",
        "SIR-PF (track hits)": "--",
        "SIR-PF (all hits)": "-.",
    }

    dataset = FittingDataset(data_dir)
    # For RNN
    rnn = FittingRNN(nb_layers=5, sum_outputs=True, rnn="gru", nb_lstm_units=50, input_size=4,
                     output_size=3, batch_size=BATCH_SIZE, dropout=0.1, bidirectional=True,
                     learn_init_states=False, init_states="rand", device=DEVICE)
    rnn = rnn.to(DEVICE)

    print("Loading saved model...")
    checkpoint = torch.load("models/rnn_best",
                            map_location=torch.device(DEVICE))
    rnn.load_state_dict(checkpoint['model_state_dict'])

    rnn.eval()
    torch.set_grad_enabled(False)

    loader_rnn = DataLoader(dataset, collate_fn=collate_fn_rnn, batch_size=1,
                            num_workers=0, shuffle=False)

    trained_model = {
                     "name": "RNN",
                     "networks": rnn,
                     "loader": loader_rnn 
                    }
    print("Making 3D Euclidean distance distribution...")
    make_distance_distribution_plot(model=trained_model, output_file=OUTPUT_FILE)

    print("DONE.")
