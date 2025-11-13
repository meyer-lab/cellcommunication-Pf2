"""
Figure A2c_d: CCC-RISE on BALF COVID-19 data. Showing PaCMAP of cells colored by cell type and disease condition.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotPaCMAP import (
    plot_labels_pacmap,
)

import seaborn as sns


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import Anndata file
    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")

    # Count the number of cells in each cell type
    celltype_counts = X.obs["celltype"].value_counts()
    print("Cell type counts:\n", celltype_counts)

    pal = sns.color_palette(palette="Set3")
    pal = pal.as_hex()
    plot_labels_pacmap(X, labelType="celltype", ax=ax[0], color_key=pal)
    pal = sns.color_palette("Set2")
    pal = [pal[0], pal[1], pal[2]]
    pal = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b in pal
    ]
    plot_labels_pacmap(X, labelType="condition", ax=ax[1], color_key=pal)
    plot_labels_pacmap(X, labelType="sample", ax=ax[2])

    return f
