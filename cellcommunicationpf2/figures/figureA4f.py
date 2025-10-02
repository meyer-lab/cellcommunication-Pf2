"""
Figure A4f: RISE decomposition with different ranks and PaCMAP projections visualization.
"""

import anndata
import numpy as np
import matplotlib.pyplot as plt
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotPaCMAP import (
    plot_labels_pacmap,
    plot_wc_pacmap,
)
from ..tensor import rise_store_r2x
from ..import_data import import_alad
from parafac2.parafac2 import parafac2_nd, anndata_to_list
from pacmap import PaCMAP


def makeFigure():
    ax, f = getSetup((10, 10), (4, 4))
    subplotLabel(ax)

    X = import_alad(gene_threshold=0, normalize=True)
    print(X)
    

    rank = 15
    print(f"Running RISE with rank {rank}")
    gene_names = list(X.var_names)
    X_list = anndata_to_list(X)

    pf2_output, _ = parafac2_nd(
        X, rank=rank, n_iter_max=10000,random_state=0
    )
    _, _, projections = pf2_output

    projected_matrices = []
    for i, tensor in enumerate(X_list):
        proj = projections[i]
        # Convert tensor to NumPy
        tensor_np = tensor.get()
        projected_matrices.append(proj.T @ tensor_np)
        
    sg_index = X.obs["condition_unique_idxs"]
    X.obsm["projections"] = np.zeros((X.shape[0], rank))
    for i, p in enumerate(projections):
            X.obsm["projections"][sg_index == i, :] = p

    # for i, resolution in enumerate([2, 3, 5, 6, 8, 10, 15, 25, 50, 100, 200]):
    #     pcm = PaCMAP(random_state=0, n_neighbors=resolution)
    #     X.obsm["PaCMAP"] = pcm.fit_transform(X.obsm["projections"])
    #     plot_labels_pacmap(X, labelType="ALADstatus", ax=ax[i]) 

    pcm = PaCMAP(random_state=0, n_neighbors=10)
    
    XX = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")

    XX.obsm["PaCMAP"] = pcm.fit_transform(XX.obsm["projections"])
    
    XX.write_h5ad("cellcommunicationpf2/alad.h5ad")


    
    
    
    
    
    return f
