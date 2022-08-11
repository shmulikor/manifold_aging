import os
import scanpy as sc
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
from BiWhitening import run_on_adata, find_alpha_beta, sinkhorn_knopp


TISSUE = 'Tissue'
CELLTYPE = 'Celltype'
K_HAT = 'k_hat'
KS = 'KS'
KS_PVAL = 'KS_pval'

K_HAT_CT = 'k_hat_celltype'
KS_CT = 'KS_celltype'
KS_PVAL_CT = 'KS_pval_celltype'
STD_CT = 'std_celltype'

K_HAT_YOUNG = 'k_hat_young'
STD_YOUNG = 'std_young'
KS_YOUNG = 'KS_young'
KS_PVAL_YOUNG = 'KS_pval_young'

K_HAT_OLD = 'k_hat_old'
STD_OLD = 'std_old'
KS_OLD = 'KS_old'
KS_PVAL_OLD = 'KS_pval_old'

GREATER = 'greater'

V_K_HAT_YOUNG = 'V_k_hat_young'
V_K_HAT_OLD = 'V_k_hat_old'
GREATER_V_K = 'greater V_k'

V_K_ALL_YOUNG = 'V_k_all_young'
V_K_ALL_OLD = 'V_k_all_old'

V_K_HAT_UMAP_YOUNG = 'V_k_hat_umap_young'
V_K_HAT_UMAP_OLD = 'V_k_hat_umap_old'

V_D_HAT_YOUNG = 'V_d_hat_young'
V_D_HAT_OLD = 'V_d_hat_old'
GREATER_V_D = 'greater V_d'


V_D_HAT = 'V_d_hat'
V_K_UMAP = 'V_k_UMAP'

N_GENES = 'n_genes'
N_CELLS = 'n_cells'
N_READS = 'n_reads'
READS_TO_CELL = 'reads_to_cell'
N_GENES_CT = 'n_genes_celltype'
N_CELLS_CT = 'n_cells_celltype'
N_READS_CT = 'n_reads_ct'
READS_TO_CELL_CT = 'reads_to_cell_ct'
N_GENES_YOUNG = 'n_genes_young'
N_CELLS_YOUNG = 'n_cells_young'
N_READS_YOUNG = 'n_reads_young'
READS_TO_CELL_YOUNG = 'reads_to_cell_young'
N_GENES_OLD = 'n_genes_old'
N_CELLS_OLD = 'n_cells_old'
N_READS_OLD = 'n_reads_old'
READS_TO_CELL_OLD = 'reads_to_cell_old'

GCL_YOUNG_BEFORE = 'GCL_young_before'
GCL_OLD_BEFORE = 'GCL_old_before'
GCL_YOUNG = 'GCL_young'
GCL_OLD = 'GCL_old'

KVK_YOUNG = 'kvk_young'
KVK_OLD = 'kvk_old'



data_dir = 'droplet_datasets'

filenames = [file for file in os.listdir(data_dir) if file.endswith('h5ad')]
filenames.sort(key=lambda filename: os.path.getsize(os.path.join(data_dir, filename)))



def filter_and_scale(adata):
    _, rows1 = sc.pp.subsample(adata.X, fraction=0.5, copy=True, random_state=np.random.randint(10 ** 6))
    rows2 = np.setdiff1d(np.arange(adata.X.shape[0]), rows1)

    # col_mask1, _ = sc.pp.filter_genes(adata.X[rows1], min_cells=int(adata.shape[0] / 100))
    # row_mask1, _ = sc.pp.filter_cells(adata.X[rows1], min_genes=int(adata.shape[0] / 100))
    col_mask1, _ = sc.pp.filter_genes(adata.X[rows1], min_cells=50)
    row_mask1, _ = sc.pp.filter_cells(adata.X[rows1], min_genes=50)
    rows1 = rows1[row_mask1]
    cols1 = np.arange(adata.shape[1])[col_mask1]

    col_mask_again, _ = sc.pp.filter_genes(adata.X[rows1][:, cols1], min_cells=1)
    row_mask_again, _ = sc.pp.filter_cells(adata.X[rows1][:, cols1], min_genes=1)
    cols1 = cols1[col_mask_again]
    rows1 = rows1[row_mask_again]

    # col_mask2, _ = sc.pp.filter_genes(adata.X[rows2], min_cells=int(adata.shape[0] / 100))
    # row_mask2, _ = sc.pp.filter_cells(adata.X[rows2], min_genes=int(adata.shape[0] / 100))
    col_mask2, _ = sc.pp.filter_genes(adata.X[rows2], min_cells=50)
    row_mask2, _ = sc.pp.filter_cells(adata.X[rows2], min_genes=50)
    rows2 = rows2[row_mask2]
    cols2 = np.arange(adata.shape[1])[col_mask2]

    col_mask_again, _ = sc.pp.filter_genes(adata.X[rows2][:, cols2], min_cells=1)
    row_mask_again, _ = sc.pp.filter_cells(adata.X[rows2][:, cols2], min_genes=1)
    cols2 = cols2[col_mask_again]
    rows2 = rows2[row_mask_again]

    alpha, beta = find_alpha_beta(adata.X[rows1][:, cols1]
                                  if adata.X[rows1][:, cols1].shape[0] < adata.X[rows1][:, cols1].shape[1]
                                  else adata.X[rows1][:, cols1].T)

    Y = adata.X[rows2][:, cols2] if rows2.shape < cols2.shape else adata.X[rows2][:, cols2].T
    transposed = False if rows2.shape < cols2.shape else True
    if transposed:
        tmp = rows2.copy()
        rows2 = cols2.copy()
        cols2 = tmp

    var_Y = alpha * ((1 - beta) * Y + beta * Y ** 2)
    x, y = sinkhorn_knopp(var_Y)
    return x, y, rows2, cols2, transposed



def find_k_hat(adata, name_to_save=None, plot=False):
    data1, rows1 = sc.pp.subsample(adata.X, fraction=0.5, copy=True, random_state=np.random.randint(10 ** 6))
    rows2 = np.setdiff1d(np.arange(adata.X.shape[0]), rows1)
    data2 = adata.X[rows2, :]

    adata_part1 = sc.AnnData(data1, obs=adata.obs.iloc[rows1], var=adata.var)
    adata_part2 = sc.AnnData(data2, obs=adata.obs.iloc[rows2], var=adata.var)

    # sc.pp.filter_genes(adata_part1, min_cells=int(adata.shape[0] / 100))
    # sc.pp.filter_cells(adata_part1, min_genes=int(adata.shape[0] / 100))
    sc.pp.filter_genes(adata_part1, min_cells=50)
    sc.pp.filter_cells(adata_part1, min_genes=50)
    sc.pp.filter_genes(adata_part1, min_cells=1)
    sc.pp.filter_cells(adata_part1, min_genes=1)

    # sc.pp.filter_genes(adata_part2, min_cells=int(adata.shape[0] / 100))
    # sc.pp.filter_cells(adata_part2, min_genes=int(adata.shape[0] / 100))
    sc.pp.filter_genes(adata_part2, min_cells=50)
    sc.pp.filter_cells(adata_part2, min_genes=50)
    sc.pp.filter_genes(adata_part2, min_cells=1)
    sc.pp.filter_cells(adata_part2, min_genes=1)

    if adata_part1.shape[0] < 120 or adata_part2.X.shape[0] < 120:
        return None, None

    print(adata_part1.shape, adata_part2.shape)
    alpha, beta = find_alpha_beta(adata_part1.X, plot=False)

    k_hat, ks_dist = run_on_adata(adata_part2, alpha=alpha, beta=beta, plot=plot,
                                     name_to_save=f"MP_fit_{name_to_save}" if name_to_save else None)

    return k_hat, ks_dist


def find_k_hat_bootstrap(adata, adata_to_subsample=None, name_to_save=None, plot=False, n_iters=2, subsample=False):
    assert isinstance(adata, sc.AnnData)
    k_list = []
    outliers = []

    for _ in range(n_iters):
        if subsample:
            assert adata_to_subsample is not None
            adata = sc.pp.subsample(adata, n_obs=adata_to_subsample.shape[0], copy=True) \
                if adata_to_subsample.shape[0] < adata.shape[0] else adata
            assert adata.shape[0] <= adata_to_subsample.shape[0]

        k_hat, ks_dist, outliers_genes = find_k_hat(adata, name_to_save=name_to_save, plot=plot)

        if k_hat is None or ks_dist is None:
            return None, None, None

        if ks_dist.pvalue > 0.1:
            k_list.append(k_hat)
            inter_before = len(list(set.intersection(*map(set, outliers)))) if len(outliers) else 0
            outliers.append(outliers_genes)
            inter_after = len(list(set.intersection(*map(set, outliers)))) if len(outliers) else 0
            print(inter_before, inter_after)
            # if inter_before == inter_after:
            #     return np.mean(k_list), np.std(k_list), list(set.intersection(*map(set, outliers)))

        # else:
        #     return None, None, None

    if not len(k_list):
        return None, None, None
    return np.mean(k_list), np.std(k_list), list(set.intersection(*map(set, outliers)))



def scale_once(csv_filename=None):
    df_cols = [TISSUE, CELLTYPE, K_HAT, KS, KS_PVAL, K_HAT_CT, KS_CT, KS_PVAL_CT,
               K_HAT_YOUNG, KS_YOUNG, KS_PVAL_YOUNG, K_HAT_OLD, KS_OLD, KS_PVAL_OLD]
    output_df = pd.DataFrame([], columns=df_cols)
    for filename in filenames[:]:
        tissue_name = filename.split('-')[-1].split('.')[0]
        adata = sc.read(os.path.join(data_dir, filename))

        if len(adata.obs.age.value_counts().keys()) == 1:
            print(f"{tissue_name} contains only one age group. Skipping...")
            continue

        if '1m' not in adata.obs.age.value_counts().keys() and '3m' not in adata.obs.age.value_counts().keys():
            print(f"{tissue_name} contains only old cells. Skipping...")
            continue

        print(tissue_name)

        X = adata.raw.X.todense()[:, adata.var['highly_variable']]
        adata = sc.AnnData(X, obs=adata.obs, var=adata.var[adata.var['highly_variable']])

        x, y, rows, cols, transposed = filter_and_scale(adata)
        col_to_y = dict(zip(cols, y))
        k_hat, ks = BW_after_scaling(adata.X[rows][:, cols] if not transposed else adata.X[cols][:, rows].T, x, y)

        for celltype in adata.obs['cell_ontology_class'].value_counts().keys():
            print(celltype)
            cols_celltype = np.intersect1d(np.where(adata.obs['cell_ontology_class'] == celltype), cols)
            if not len(cols_celltype):
                continue
            sub_y = [col_to_y[col] for col in cols_celltype]
            k_hat_ct, ks_ct = BW_after_scaling(adata.X[rows][:, cols_celltype]
                                               if not transposed else adata.X[cols_celltype][:, rows].T, x, sub_y,
                                               plot=True)
            print(f"{celltype} : {k_hat_ct}, pval: {ks_ct.pvalue}")

            is_young = (adata.obs['age'] == '1m') | (adata.obs['age'] == '3m')
            cols_young = np.intersect1d(
                np.where(np.logical_and(adata.obs['cell_ontology_class'] == celltype, is_young)), cols)
            if not len(cols_young):
                continue
            sub_y_young = [col_to_y[col] for col in cols_young]
            k_hat_young, ks_young = BW_after_scaling(adata.X[rows][:, cols_young]
                                                     if not transposed else adata.X[cols_young][:, rows].T, x, sub_y_young,
                                                     plot=True)
            print(f"{celltype} young: {k_hat_young}, pval: {ks_young.pvalue}")

            cols_old = np.intersect1d(np.where(np.logical_and(adata.obs['cell_ontology_class'] == celltype, ~is_young)),
                                      cols)
            if not len(cols_old):
                continue
            sub_y_old = [col_to_y[col] for col in cols_old]
            k_hat_old, ks_old = BW_after_scaling(adata.X[rows][:, cols_old]
                                                 if not transposed else adata.X[cols_old][:, rows].T, x, sub_y_old,
                                                 plot=True)
            print(f"{celltype} old: {k_hat_old}, pval: {ks_old.pvalue}")

            output_df = output_df.append({TISSUE: tissue_name, CELLTYPE: celltype,

                                          K_HAT: k_hat, KS: ks.statistic, KS_PVAL: ks.pvalue,

                                          K_HAT_CT: k_hat_ct, KS_CT: ks_ct.statistic, KS_PVAL_CT: ks_ct.pvalue,

                                          K_HAT_YOUNG: k_hat_young, KS_YOUNG: ks_young.statistic,
                                          KS_PVAL_YOUNG: ks_young.pvalue,

                                          K_HAT_OLD: k_hat_old, KS_OLD: ks_old.statistic, KS_PVAL_OLD: ks_old.pvalue},
                                         ignore_index=True)

            print(output_df)
            if csv_filename is not None:
                output_df.to_csv(csv_filename)

def scale_celltype(csv_filename=None, save_figs=False):
    df_cols = [TISSUE, CELLTYPE, K_HAT, KS, KS_PVAL, K_HAT_CT, KS_CT, KS_PVAL_CT,
               K_HAT_YOUNG, KS_YOUNG, KS_PVAL_YOUNG, K_HAT_OLD, KS_OLD, KS_PVAL_OLD]
    output_df = pd.DataFrame([], columns=df_cols)
    for filename in filenames[:]:
        tissue_name = filename.split('-')[-1].split('.')[0]
        adata = sc.read(os.path.join(data_dir, filename))

        if len(adata.obs.age.value_counts().keys()) == 1:
            print(f"{tissue_name} contains only one age group. Skipping...")
            continue

        if '1m' not in adata.obs.age.value_counts().keys() and '3m' not in adata.obs.age.value_counts().keys():
            print(f"{tissue_name} contains only old cells. Skipping...")
            continue

        print(tissue_name)

        X = adata.raw.X.todense()[:, adata.var['highly_variable']]
        adata = sc.AnnData(X, obs=adata.obs, var=adata.var[adata.var['highly_variable']])

        # x, y, rows, cols, transposed = filter_and_scale(adata)
        # col_to_y = dict(zip(cols, y))
        # k_hat, ks = BW_after_scaling(adata.X[rows][:, cols] if not transposed else adata.X[cols][:, rows].T, x, y)
        k_hat, ks = find_k_hat(adata, name_to_save=tissue_name if save_figs else None, plot=True)

        for celltype in adata.obs['cell_ontology_class'].value_counts().keys():
            print(celltype)

            adata_ct = adata[adata.obs['cell_ontology_class'] == celltype]
            x, y, rows, cols, transposed = filter_and_scale(adata_ct)
            col_to_y = dict(zip(cols, y))
            k_hat_ct, ks_ct = BW_after_scaling(adata_ct.X[rows][:, cols] if not transposed else adata_ct.X[cols][:, rows].T, x, y, plot=True)
            print(f"{celltype} : {k_hat_ct}, pval: {ks_ct.pvalue}")

            is_young = (adata_ct.obs['age'] == '1m') | (adata_ct.obs['age'] == '3m')
            cols_young = np.intersect1d(np.where(is_young), cols)
            if not len(cols_young):
                continue
            sub_y_young = [col_to_y[col] for col in cols_young]
            k_hat_young, ks_young = BW_after_scaling(adata.X[rows][:, cols_young]
                                                     if not transposed else adata.X[cols_young][:, rows].T, x,
                                                     sub_y_young,
                                                     plot=True)
            print(f"{celltype} young: {k_hat_young}, pval: {ks_young.pvalue}")

            cols_old = np.intersect1d(np.where(~is_young), cols)
            if not len(cols_old):
                continue
            sub_y_old = [col_to_y[col] for col in cols_old]
            k_hat_old, ks_old = BW_after_scaling(adata.X[rows][:, cols_old]
                                                 if not transposed else adata.X[cols_old][:, rows].T, x, sub_y_old,
                                                 plot=True)
            print(f"{celltype} old: {k_hat_old}, pval: {ks_old.pvalue}")

            output_df = output_df.append({TISSUE: tissue_name, CELLTYPE: celltype,

                                          K_HAT: k_hat, KS: ks.statistic, KS_PVAL: ks.pvalue,

                                          K_HAT_CT: k_hat_ct, KS_CT: ks_ct.statistic, KS_PVAL_CT: ks_ct.pvalue,

                                          K_HAT_YOUNG: k_hat_young, KS_YOUNG: ks_young.statistic,
                                          KS_PVAL_YOUNG: ks_young.pvalue,

                                          K_HAT_OLD: k_hat_old, KS_OLD: ks_old.statistic, KS_PVAL_OLD: ks_old.pvalue},
                                         ignore_index=True)

            print(output_df)
            if csv_filename is not None:
                output_df.to_csv(csv_filename)


def simple_run(csv_filename=None, save_figs=False):
    df_cols = [TISSUE, CELLTYPE, K_HAT, KS, KS_PVAL,
               K_HAT_CT, KS_CT, KS_PVAL_CT,
               K_HAT_YOUNG, KS_YOUNG, KS_PVAL_YOUNG,
               K_HAT_OLD, KS_OLD, KS_PVAL_OLD]

    output_df = pd.DataFrame([], columns=df_cols)
    for filename in filenames[:]:
        tissue_name = filename.split('-')[-1].split('.')[0]
        adata = sc.read(os.path.join(data_dir, filename))

        if len(adata.obs.age.value_counts().keys()) == 1:
            print(f"{tissue_name} contains only one age group. Skipping...")
            continue

        if '1m' not in adata.obs.age.value_counts().keys() and '3m' not in adata.obs.age.value_counts().keys():
            print(f"{tissue_name} contains only old cells. Skipping...")
            continue

        print(tissue_name)

        X = adata.raw.X.todense()[:, adata.var['highly_variable']]
        adata = sc.AnnData(X, obs=adata.obs, var=adata.var[adata.var['highly_variable']])

        # k_hat, ks = find_k_hat(adata, name_to_save=tissue_name if save_figs else None, plot=True)

        # adata_random0 = sc.AnnData(np.apply_along_axis(np.random.permutation, 0, adata.X))
        # k_hat_random0, ks_random0 = find_k_hat(adata_random0, name_to_save=tissue_name if save_figs else None, plot=True)

        # sc.pp.pca(adata, n_comps=k_hat)
        # X_tag = adata.obsm['X_pca']
        # X_hat = np.dot(X_tag[:, :k_hat], adata.varm['PCs'].T)
        # X_hat += np.mean(adata.X, axis=0)

        # sc.pp.neighbors(adata)
        # sc.tl.umap(adata)
        # sc.pl.umap(adata, color='cell_ontology_class') #, save=f"_{tissue_name}_after_PCA_BW.png")

        for celltype in adata.obs['cell_ontology_class'].value_counts().keys():
            print(celltype)
            # v_d_hat = mean_squared_error(adata.X[adata.obs['cell_ontology_class'] == celltype], X_hat[adata.obs['cell_ontology_class'] == celltype])
            # v_k_hat = np.mean(np.var(X_tag[adata.obs['cell_ontology_class'] == celltype], axis=0))
            # v_k_hat_umap = np.mean(np.var(adata.obsm['X_umap'][adata.obs['cell_ontology_class'] == celltype], axis=0))

            adata_ct = adata[adata.obs['cell_ontology_class'] == celltype]
            # n_cells_ct, n_genes_ct = adata_ct.shape
            # k_hat_ct, ks_ct = find_k_hat(adata_ct, plot=True)
            # print(f"k_hat_celltype = {k_hat_ct}")

            is_young = (adata.obs['age'] == '1m') | (adata.obs['age'] == '3m')
            adata_young = adata[np.logical_and(adata.obs['cell_ontology_class'] == celltype, is_young)]
            adata_old = adata[np.logical_and(adata.obs['cell_ontology_class'] == celltype, ~is_young)]

            # n_cells_young, n_genes_young = adata_young.shape
            k_hat_young, ks_young = find_k_hat(adata_young, plot=True)
            print(f"k_hat_young = {k_hat_young}")
            if k_hat_young is None:
                continue

            pca = PCA(n_components=k_hat_young)
            X_tag = pca.fit_transform(adata_young.X)
            X_hat = pca.inverse_transform(X_tag)

            v_k_hat = np.mean(np.var(X_tag, axis=0))
            v_d_hat = mean_squared_error(adata_young.X, X_hat)
            print(v_k_hat, v_d_hat)



            k_hat_old, ks_old = find_k_hat(adata_old, plot=True)
            print(f"k_hat_old = {k_hat_old}")
            if k_hat_old is None:
                continue

            pca = PCA(n_components=k_hat_old)
            X_tag = pca.fit_transform(adata_old.X)
            X_hat = pca.inverse_transform(X_tag)

            v_k_hat = np.mean(np.var(X_tag, axis=0))
            v_d_hat = mean_squared_error(adata_old.X, X_hat)
            print(v_k_hat, v_d_hat)

            output_df = output_df.append({TISSUE: tissue_name, CELLTYPE: celltype,

                                          K_HAT_YOUNG: k_hat_young, KS_YOUNG: ks_young.statistic, KS_PVAL_YOUNG: ks_young.pvalue,

                                          K_HAT_OLD: k_hat_old, KS_OLD: ks_old.statistic, KS_PVAL_OLD: ks_old.pvalue},
                                         ignore_index=True)
            print(output_df)
            if csv_filename is not None:
                output_df.to_csv(csv_filename)

def func(adata):
    data1, cols1 = sc.pp.subsample(adata.X.T, fraction=0.5, copy=True, random_state=np.random.randint(10 ** 6))
    data1 = data1.T
    cols2 = np.setdiff1d(np.arange(adata.X.shape[1]), cols1)
    data2 = adata[:, cols2].X
    return data1, data2

def run_with_bootstrap(csv_filename=None, save_figs=False, subsample=False, plot=False):
    df_cols = [TISSUE, CELLTYPE, K_HAT, KS, KS_PVAL, N_CELLS, N_GENES, N_READS, READS_TO_CELL,
                K_HAT_CT, STD_CT, N_CELLS_CT, N_GENES_CT, N_READS_CT, READS_TO_CELL_CT,
                K_HAT_YOUNG, STD_YOUNG, N_CELLS_YOUNG, N_GENES_YOUNG, N_READS_YOUNG, READS_TO_CELL_YOUNG,
                K_HAT_OLD, STD_OLD, N_CELLS_OLD, N_GENES_OLD, N_READS_OLD, READS_TO_CELL_OLD, GREATER,
                V_K_ALL_YOUNG, V_K_ALL_OLD, V_K_HAT_YOUNG, V_K_HAT_OLD, V_K_HAT_UMAP_YOUNG, V_K_HAT_UMAP_OLD,
                V_D_HAT_YOUNG, V_D_HAT_OLD, GREATER_V_K, GREATER_V_D, GCL_YOUNG_BEFORE, GCL_OLD_BEFORE,
                GCL_YOUNG, GCL_OLD, KVK_YOUNG, KVK_OLD]

    output_df = pd.DataFrame([], columns=df_cols)
    for filename in filenames:
        tissue_name = filename.split('-')[-1].split('.')[0]
        adata = sc.read(os.path.join(data_dir, filename))

        if len(adata.obs.age.value_counts().keys()) == 1:
            print(f"{tissue_name} contains only one age group. Skipping...")
            continue

        if '1m' not in adata.obs.age.value_counts().keys() and '3m' not in adata.obs.age.value_counts().keys():
            print(f"{tissue_name} contains only old cells. Skipping...")
            continue

        print(tissue_name)

        X = adata.raw.X.todense()[:, adata.var['highly_variable']]
        adata = sc.AnnData(X, obs=adata.obs, var=adata.var[adata.var['highly_variable']])
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=10_000)
        sc.pp.log1p(adata)

        # adata = sc.AnnData(np.log1p(X), obs=adata.obs, var=adata.var[adata.var['highly_variable']])

        n_cells, n_genes = adata.shape
        n_reads = adata.X.sum().tolist()
        k_hat, ks, _ = find_k_hat(adata, name_to_save=tissue_name if save_figs else None, plot=plot)

        # sc.pp.pca(adata, n_comps=k_hat)
        # X_tag = adata.obsm['X_pca']
        # X_hat = np.dot(X_tag[:, :k_hat], adata.varm['PCs'].T)
        # X_hat += np.mean(adata.X, axis=0)

        # sc.pp.neighbors(adata)
        # sc.tl.umap(adata)
        # sc.pl.umap(adata, color='cell_ontology_class') #, save=f"_{tissue_name}_after_PCA_BW.png")

        for celltype in adata.obs['cell_ontology_class'].value_counts().keys():
            print(celltype)
            # v_d_hat = mean_squared_error(adata.X[adata.obs['cell_ontology_class'] == celltype], X_hat[adata.obs['cell_ontology_class'] == celltype])
            # v_k_hat = np.mean(np.var(X_tag[adata.obs['cell_ontology_class'] == celltype], axis=0))
            # v_k_hat_umap = np.mean(np.var(adata.obsm['X_umap'][adata.obs['cell_ontology_class'] == celltype], axis=0))

            adata_ct = adata[adata.obs['cell_ontology_class'] == celltype]
            n_cells_ct, n_genes_ct = adata_ct.shape
            n_reads_ct = adata_ct.X.sum().tolist()
            k_hat_ct, std_ct, outliers_ct = find_k_hat_bootstrap(adata_ct)
            print(f"k_hat_celltype = {k_hat_ct}")

            is_young = (adata.obs['age'] == '1m') | (adata.obs['age'] == '3m')
            adata_young = adata[np.logical_and(adata.obs['cell_ontology_class'] == celltype, is_young)]
            adata_old = adata[np.logical_and(adata.obs['cell_ontology_class'] == celltype, ~is_young)]

            n_cells_young, n_genes_young = adata_young.shape
            n_reads_young = adata_young.X.sum().tolist()
            k_hat_young, std_young, outliers_young = find_k_hat_bootstrap(adata_young, adata_to_subsample=adata_old, subsample=False)
            print(f"k_hat_young = {k_hat_young}")
            if k_hat_young is None:
                continue

            n_cells_old, n_genes_old = adata_old.shape
            n_reads_old = adata_old.X.sum().tolist()
            k_hat_old, std_old, outliers_old = find_k_hat_bootstrap(adata_old, adata_to_subsample=adata_young, subsample=False)
            print(f"k_hat_old = {k_hat_old}")
            if k_hat_old is None:
                continue

            greater = 'overlap'
            if np.abs(k_hat_young - k_hat_old) > 2 * np.sqrt(std_young ** 2 + std_old ** 2):
                greater = 'young' if k_hat_young > k_hat_old else 'old'

            v_k_hat_young, v_k_hat_old, greater_var = None, None, None
            v_k_all_young, v_k_all_old, v_k_hat_umap_young, v_k_hat_umap_old = None, None, None, None
            v_d_young, v_d_old = None, None

            # if greater == 'overlap':
            if True:
                non_ct = np.array([g for g in adata_ct.var.index if g not in outliers_ct])
                non_y = np.array([g for g in adata_young.var.index if g not in outliers_young])
                non_o = np.array([g for g in adata_old.var.index if g not in outliers_old])

                ad_ct = adata_ct[:, non_ct]
                ad_young = adata_young[:, non_y]
                ad_old = adata_old[:, non_o]

                # young
                v_k_all_young = np.mean(np.var(ad_young.X, axis=0))

                # sc.pp.pca(adata_ct, n_comps=np.round(k_hat_ct).astype(int))

                # is_young_ct = (adata_ct.obs['age'] == '1m') | (adata_ct.obs['age'] == '3m')
                # plt.scatter(adata_ct.obsm['X_pca'][~is_young_ct][:, 0], adata_ct.obsm['X_pca'][~is_young_ct][:, 1],
                #             label='old', alpha=0.2)
                # plt.scatter(adata_ct.obsm['X_pca'][is_young_ct][:, 0], adata_ct.obsm['X_pca'][is_young_ct][:, 1],
                #             label='young', alpha=0.2)
                # plt.title(celltype)
                # plt.legend()
                # plt.show()

                # TODO - need to center?
                # gcl_young_before = GCL(ad_young.X.T)
                # print(f"young GCL before PCA is {gcl_young_before}")

                sc.pp.pca(ad_young, n_comps=np.round(k_hat_ct).astype(int))
                X_tag_young = ad_young.obsm['X_pca']
                v_k_hat_young = np.mean(np.var(X_tag_young, axis=0))

                # gcl_young = GCL(X_tag_young.T)
                # print(f"young GCL is {gcl_young}")

                sc.pp.neighbors(ad_young)
                sc.tl.umap(ad_young)
                v_k_hat_umap_young = np.mean(np.var(ad_young.obsm['X_umap'], axis=0))

                X_hat_young = np.dot(X_tag_young, ad_young.varm['PCs'].T)
                X_hat_young += np.mean(ad_young.X, axis=0)
                v_d_young = mean_squared_error(ad_young.X, X_hat_young)

                # old
                v_k_all_old = np.mean(np.var(ad_old.X, axis=0))

                # gcl_old_before = GCL(ad_old.X.T)
                # print(f"old GCL before PCA is {gcl_old_before}")

                sc.pp.pca(ad_old, n_comps=np.round(k_hat_ct).astype(int))
                X_tag_old = ad_old.obsm['X_pca']
                v_k_hat_old = np.mean(np.var(X_tag_old, axis=0))

                # gcl_old = GCL(X_tag_old.T)
                # print(f"old GCL is {gcl_old}")

                sc.pp.neighbors(ad_old)
                sc.tl.umap(ad_old)
                v_k_hat_umap_old = np.mean(np.var(ad_old.obsm['X_umap'], axis=0))

                X_hat_old = np.dot(X_tag_old, ad_old.varm['PCs'].T)
                X_hat_old += np.mean(ad_old.X, axis=0)
                v_d_old = mean_squared_error(ad_old.X, X_hat_old)

                # plt.plot(np.cumsum(adata_old.uns['pca']['variance']), label='old')
                # plt.plot(np.cumsum(adata_young.uns['pca']['variance']), label='young')
                # plt.title(f"{tissue_name} - {celltype}")
                # plt.legend()
                # plt.show()

                # pca = PCA(n_components=np.round(k_hat_ct).astype(int))
                # pca.fit(adata_ct.X - adata_ct.X.mean())
                # cov_young = np.cov(adata_ct.X[is_young_ct].T)
                # cov_old = np.cov(adata_ct.X[~is_young_ct].T)
                # explained_young = np.real(np.linalg.eigvals(cov_young))[:pca.n_components]
                # explained_old = np.real(np.linalg.eigvals(cov_old))[:pca.n_components]

                # df = pd.DataFrame({'all': (pca.explained_variance_ - pca.explained_variance_.min()) /
                #          (pca.explained_variance_.max() - pca.explained_variance_.min()), 'young': (explained_young - explained_young.min()) /
                #          (explained_young.max() - explained_young.min()), 'old': (explained_old - explained_old.min()) /
                #          (explained_old.max() - explained_old.min())})
                # df.plot.bar()
                # plt.title(f"{tissue_name} - {celltype}")
                # plt.show()

                # plt.plot(pca.explained_variance_, label='all')
                # plt.plot(explained_young, label='young')
                # plt.plot(explained_old, label='old')
                #
                # plt.plot(np.cumsum(pca.explained_variance_), label='all cum')
                # plt.plot(np.cumsum(explained_young), label='young cum')
                # plt.plot(np.cumsum(explained_old), label='old cum')
                # plt.title(f"{tissue_name} - {celltype}")
                # plt.legend()
                # plt.show()

                kvk_young = (k_hat_ct * v_k_hat_young) / ((k_hat_ct * v_k_hat_young) + (n_genes - k_hat_ct) * v_d_young)
                kvk_old = (k_hat_ct * v_k_hat_old) / ((k_hat_ct * v_k_hat_old) + (n_genes - k_hat_ct) * v_d_old)
                print(kvk_young, kvk_old)

                greater_v_k = 'young' if v_k_hat_young > v_k_hat_old else 'old'
                greater_v_d = 'young' if v_d_young > v_d_old else 'old'


            output_df = output_df.append({TISSUE: tissue_name, CELLTYPE: celltype,

                                          K_HAT: k_hat, KS: ks.statistic, KS_PVAL: ks.pvalue,
                                          N_CELLS: n_cells, N_GENES: n_genes,
                                          N_READS: n_reads, READS_TO_CELL: n_reads / n_cells,

                                          K_HAT_CT: k_hat_ct, STD_CT: std_ct,
                                          N_CELLS_CT: n_cells_ct, N_GENES_CT: n_genes_ct,
                                          N_READS_CT: n_reads_ct, READS_TO_CELL_CT: n_reads_ct / n_cells_ct,

                                          K_HAT_YOUNG: k_hat_young, STD_YOUNG: std_young,
                                          N_CELLS_YOUNG: n_cells_young, N_GENES_YOUNG: n_genes_young,
                                          N_READS_YOUNG: n_reads_young, READS_TO_CELL_YOUNG: n_reads_young / n_cells_young,

                                          K_HAT_OLD: k_hat_old, STD_OLD: std_old,
                                          N_CELLS_OLD: n_cells_old, N_GENES_OLD: n_genes_old,
                                          N_READS_OLD: n_reads_young, READS_TO_CELL_OLD: n_reads_old / n_cells_old,

                                          GREATER: greater,

                                          V_K_ALL_YOUNG: v_k_all_young, V_K_ALL_OLD: v_k_all_old,
                                          V_K_HAT_YOUNG: v_k_hat_young, V_K_HAT_OLD: v_k_hat_old,
                                          V_K_HAT_UMAP_YOUNG: v_k_hat_umap_young, V_K_HAT_UMAP_OLD: v_k_hat_umap_old,
                                          V_D_HAT_YOUNG: v_d_young, V_D_HAT_OLD: v_d_old,
                                          GREATER_V_K: greater_v_k, GREATER_V_D: greater_v_d,

                                          KVK_YOUNG: kvk_young, KVK_OLD: kvk_old

                                          # GCL_YOUNG_BEFORE: gcl_young_before, GCL_OLD_BEFORE: gcl_old_before,
                                          # GCL_YOUNG: gcl_young, GCL_OLD: gcl_old},
                                          }, ignore_index=True)
            print(output_df)
            if csv_filename is not None:
                output_df.to_csv(csv_filename)



def main_loop(csv_filename=None, save_figs=False):
    # df_cols = [TISSUE, CELLTYPE, K_HAT, KS, KS_PVAL, K_HAT_CT, KS_CT, KS_PVAL_CT, K_HAT_YOUNG, KS_YOUNG, KS_PVAL_YOUNG,
    #            K_HAT_OLD, KS_OLD, KS_PVAL_OLD]
    df_cols = [TISSUE, CELLTYPE, K_HAT, KS, KS_PVAL, K_HAT_CT, STD_CT, K_HAT_YOUNG, STD_YOUNG, K_HAT_OLD, STD_OLD, GREATER]

    output_df = pd.DataFrame([], columns=df_cols)
    for filename in filenames[:]:
        tissue_name = filename.split('-')[-1].split('.')[0]
        adata = sc.read(os.path.join(data_dir, filename))

        if len(adata.obs.age.value_counts().keys()) == 1:
            print(f"{tissue_name} contains only one age group. Skipping...")
            continue

        if '1m' not in adata.obs.age.value_counts().keys() and '3m' not in adata.obs.age.value_counts().keys():
            print(f"{tissue_name} contains only old cells. Skipping...")
            continue

        print(tissue_name)

        X = adata.raw.X.todense()[:, adata.var['highly_variable']]
        adata = sc.AnnData(X, obs=adata.obs, var=adata.var[adata.var['highly_variable']])

        k_hat, ks = find_k_hat(adata, name_to_save=tissue_name if save_figs else None, plot=True)

        # sc.pp.pca(adata, n_comps=k_hat)
        # X_tag = adata.obsm['X_pca']
        # X_hat = np.dot(X_tag[:, :k_hat], adata.varm['PCs'].T)
        # X_hat += np.mean(adata.X, axis=0)

        # sc.pp.neighbors(adata)
        # sc.tl.umap(adata)
        # sc.pl.umap(adata, color='cell_ontology_class') #, save=f"_{tissue_name}_after_PCA_BW.png")

        for celltype in adata.obs['cell_ontology_class'].value_counts().keys():
            print(celltype)
            # v_d_hat = mean_squared_error(adata.X[adata.obs['cell_ontology_class'] == celltype], X_hat[adata.obs['cell_ontology_class'] == celltype])
            # v_k_hat = np.mean(np.var(X_tag[adata.obs['cell_ontology_class'] == celltype], axis=0))
            # v_k_hat_umap = np.mean(np.var(adata.obsm['X_umap'][adata.obs['cell_ontology_class'] == celltype], axis=0))

            k_hat_ct, std_ct = find_k_hat_bootstrap(adata[adata.obs['cell_ontology_class'] == celltype])
            # k_hat_ct, ks_ct = find_k_hat(adata[adata.obs['cell_ontology_class'] == celltype], plot=True)
            print(f"k_hat_celltype = {k_hat_ct}")

            is_young = (adata.obs['age'] == '1m') | (adata.obs['age'] == '3m')
            adata_young = adata[np.logical_and(adata.obs['cell_ontology_class'] == celltype, is_young)]
            adata_old = adata[np.logical_and(adata.obs['cell_ontology_class'] == celltype, ~is_young)]

            k_hat_young, std_young = find_k_hat_bootstrap(adata_young, adata_to_subsample=adata_old, subsample=False)
            # k_hat_young, ks_young = find_k_hat(adata_young, plot=True)
            print(f"k_hat_young = {k_hat_young}")
            if k_hat_young is None:
                continue

            k_hat_old, std_old = find_k_hat_bootstrap(adata_old, adata_to_subsample=adata_young, subsample=False)
            # k_hat_old, ks_old = find_k_hat(adata_old, plot=True)
            print(f"k_hat_old = {k_hat_old}")
            if k_hat_old is None:
                continue

            greater = 'overlap'
            if k_hat_young - std_young > k_hat_old + std_old:
                greater = 'young'
            elif k_hat_old - std_old > k_hat_young + std_young:
                greater = 'old'

            output_df = output_df.append({TISSUE: tissue_name, CELLTYPE: celltype,

                                          K_HAT: k_hat, KS: ks.statistic, KS_PVAL: ks.pvalue,

                                          K_HAT_CT: k_hat_ct, STD_CT: std_ct,

                                          K_HAT_YOUNG: k_hat_young, STD_YOUNG: std_young,

                                          K_HAT_OLD: k_hat_old, STD_OLD: std_old,

                                          GREATER:greater},
                                         ignore_index=True)
            print(output_df)
            if csv_filename is not None:
                output_df.to_csv(csv_filename)


def split_old_young():
    for filename in filenames[:]:
        tissue_name = filename.split('-')[-1].split('.')[0]
        adata = sc.read(os.path.join(data_dir, filename))
        if len(adata.obs.age.value_counts().keys()) == 1:
            print(f"{tissue_name} contains only one age group. Skipping...")
            continue

        print(tissue_name)
        df_young = pd.DataFrame([], columns=adata.obs['cell_ontology_class'].value_counts().keys())
        df_old = pd.DataFrame([], columns=adata.obs['cell_ontology_class'].value_counts().keys())

        X = adata.raw.X.todense()[:, adata.var['highly_variable']]
        adata = sc.AnnData(X, obs=adata.obs, var=adata.var[adata.var['highly_variable']])
        k_hat, ks_dist = find_k_hat(adata)

        # sc.pp.pca(adata, n_comps=k_hat * 2)
        # explained = adata.uns['pca']['variance_ratio']
        # X_tag = adata.obsm['X_pca']

        k_vals = np.arange(1, k_hat * 2)
        k_hat_by_celltype = {}
        for celltype in adata.obs['cell_ontology_class'].value_counts().keys():
            print(celltype)

            is_young = (adata.obs['age'] == '1m') | (adata.obs['age'] == '3m')
            adata_young = adata[np.logical_and(adata.obs['cell_ontology_class'] == celltype, is_young)]
            adata_old = adata[np.logical_and(adata.obs['cell_ontology_class'] == celltype, ~is_young)]

            k_hat_ct_young, ks_dist_young = find_k_hat(adata_young, plot=False)
            k_hat_ct_old, ks_dist_old = find_k_hat(adata_old, plot=False)

            if k_hat_ct_young is None or k_hat_ct_old is None:
                continue

            k_hat_by_celltype[celltype] = {'young': k_hat_ct_young, 'old': k_hat_ct_old}

            sc.pp.pca(adata_young, n_comps=k_hat * 2)
            X_tag_young = adata_young.obsm['X_pca']
            df_young[celltype] = [0] + [np.mean(np.var(X_tag_young[:, :k], axis=0)) for k in k_vals]


            sc.pp.pca(adata_old, n_comps=k_hat * 2)
            X_tag_old = adata_old.obsm['X_pca']
            df_old[celltype] = [0] + [np.mean(np.var(X_tag_old[:, :k], axis=0)) for k in k_vals]

        # plot
        for celltype in k_hat_by_celltype.keys():
            p = plt.plot(df_young.index, df_young[celltype].cumsum(), label=f"{celltype} - young")
            plt.axvline(x=k_hat_by_celltype[celltype]['young'], color=p[0].get_color(), ls='--')
            p = plt.plot(df_old.index, df_old[celltype].cumsum(), label=f"{celltype} - old")
            plt.axvline(x=k_hat_by_celltype[celltype]['old'], color=p[0].get_color(), ls='--')
        plt.axvline(x=k_hat, color='black', ls='--', label='BW k_hat')
        plt.xlabel('k')
        plt.ylabel('V_k_hat')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True)
        plt.show()

        print(1)


def check_data_shape():
    counts = {}
    for filename in filenames[:]:
        tissue_name = filename.split('-')[-1].split('.')[0]
        adata = sc.read(os.path.join(data_dir, filename))

        if len(adata.obs.age.value_counts().keys()) == 1:
            print(f"{tissue_name} contains only one age group. Skipping...")
            continue

        if '1m' not in adata.obs.age.value_counts().keys() and '3m' not in adata.obs.age.value_counts().keys():
            print(f"{tissue_name} contains only old cells. Skipping...")
            continue

        print(tissue_name)
        counts[tissue_name] = {}
        for celltype in adata.obs['cell_ontology_class'].value_counts().keys():
            print(celltype)
            age_dict = adata[adata.obs['cell_ontology_class'] == celltype].obs['age'].value_counts()
            counts[tissue_name][celltype] = dict(zip([int(k[:-1]) for k in age_dict.keys()], age_dict.values))
            # ned_dict =
            # counts[tissue_name][celltype] = defaultdict(int, adata[adata.obs['cell_ontology_class'] == celltype].obs['age'].value_counts())

    print(1)


def again():
    for filename in filenames[:]:
        tissue_name = filename.split('-')[-1].split('.')[0]
        adata = sc.read(os.path.join(data_dir, filename))

        if len(adata.obs.age.value_counts().keys()) == 1:
            print(f"{tissue_name} contains only one age group. Skipping...")
            continue

        if '1m' not in adata.obs.age.value_counts().keys() and '3m' not in adata.obs.age.value_counts().keys():
            print(f"{tissue_name} contains only old cells. Skipping...")
            continue

        print(tissue_name)

        X = adata.raw.X.todense()[:, adata.var['highly_variable']]
        adata = sc.AnnData(X, obs=adata.obs, var=adata.var[adata.var['highly_variable']])

        for celltype in adata.obs['cell_ontology_class'].value_counts().keys():
            print(celltype)

            is_young = (adata.obs['age'] == '1m') | (adata.obs['age'] == '3m')
            adata_young = adata[np.logical_and(adata.obs['cell_ontology_class'] == celltype, is_young)]
            adata_old = adata[np.logical_and(adata.obs['cell_ontology_class'] == celltype, ~is_young)]

            for _ in range(3):
                k_hat_ct_young, ks_dist_young = find_k_hat(adata_young, plot=False)
                print(k_hat_ct_young)
                print('-------------------------------')

            print(1)


def scale_one_time():
    output_df = pd.DataFrame([], columns=df_cols)
    for filename in filenames[:]:
        tissue_name = filename.split('-')[-1].split('.')[0]
        adata = sc.read(os.path.join(data_dir, filename))

        if len(adata.obs.age.value_counts().keys()) == 1:
            print(f"{tissue_name} contains only one age group. Skipping...")
            continue

        if '1m' not in adata.obs.age.value_counts().keys() and '3m' not in adata.obs.age.value_counts().keys():
            print(f"{tissue_name} contains only old cells. Skipping...")
            continue

        print(tissue_name)

        X = adata.raw.X.todense()[:, adata.var['highly_variable']]
        adata = sc.AnnData(X, obs=adata.obs, var=adata.var[adata.var['highly_variable']])

        k_hat, ks = find_k_hat(adata, name_to_save=tissue_name if save_figs else None)


# if __name__ == '__main__':
    # main_loop(csv_filename='real_data_PCW_BW_new_bootstrap10.csv', save_figs=False)
    # scale_once(csv_filename='real_data_PCW_BW_new_one_scale.csv')
    # split_old_young()
    # check_data_shape()
    # again()
    # simple_run(csv_filename=None)
    # run_with_bootstrap(subsample=False)
    # run_with_bootstrap(csv_filename='real_data_PCA_BW_log_kvk.csv', subsample=False)
    # scale_celltype(csv_filename='real_data_PCA_BW_new_scale_celltype.csv')