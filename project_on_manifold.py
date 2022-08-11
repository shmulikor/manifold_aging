import os

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import normalize

from BW_real_data import find_k_hat
from manapprox import ManApprox

from skdim.id import TwoNN

TISSUE = 'Tissue'
CELLTYPE = 'Celltype'


K_HAT_YOUNG = 'k_hat_young'
INT_DIM_YOUNG = 'int_dim_young'
V_K_YOUNG = 'v_k_young'
V_K_YOUNG_ENRICHED = 'v_k_young_enriched'
V_D1_YOUNG = 'v_d1_young'
V_D2_YOUNG = 'v_d2_young'

K_HAT_OLD = 'k_hat_old'
INT_DIM_OLD = 'int_dim_old'
V_K_OLD = 'v_k_old'
V_K_OLD_ENRICHED = 'v_k_old_enriched'
V_D1_OLD = 'v_d1_old'
V_D2_OLD = 'v_d2_old'


def TWO_NN(X):
    n = X.shape[0]
    distances = cdist(X, X)
    shortest = np.array([sorted(distances[i])[1:3] for i in range(n)])
    mu = shortest[:, 1] / shortest[:, 0]

    f = np.zeros(n)
    f[mu.argsort()] = np.arange(1, n+1) / n

    x = np.log(mu)
    y = -np.log(1 - f)

    taken_idx = mu.argsort()[:int(0.9 * n)]
    reg = LinearRegression(fit_intercept=False).fit(x[taken_idx].reshape(-1, 1), y[taken_idx])
    print(reg.coef_[0])
    return np.round(reg.coef_[0]).astype(int)


def denoise_manifold(manifold, simulation=False):
    prev_dim = np.inf
    dim = manifold.manifold_dim

    while dim < prev_dim:
        success = False
        while not success:
            try:
                manifold.manifold_dim = dim
                projected = manifold.projectManyPoints(manifold.data)
                projected = projected.T if not simulation else projected
                success = True
            except np.linalg.LinAlgError:
                if manifold.sparse_factor < 320:
                    print(f"Failed. Changing sparse factor from {manifold.sparse_factor} to {manifold.sparse_factor * 2}")
                    manifold.sparse_factor *= 2
                    manifold.calculateSigma()
                else:
                    return None, None

        prev_dim = dim
        dim = np.round(TwoNN().fit_transform(projected)).astype(int) if not simulation else np.round(TwoNN().fit_transform(projected.T)).astype(int)
        print(f"previous dim is {prev_dim}, current dim is {dim}")

    manifold.manifold_dim = dim
    return manifold, projected


def estimate_knn_v(X, dim, enriched=0):
    n_neighbors = dim + enriched
    knn = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance').toarray()
    g = nx.from_numpy_matrix(knn)
    while nx.number_connected_components(g) > 1:
        n_neighbors += 1
        knn = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance').toarray()
        g = nx.from_numpy_matrix(knn)

    fwn = nx.floyd_warshall_numpy(g)
    v = (fwn ** 2).sum().sum() / (2 * fwn.shape[0] ** 2)
    return v

def estimate_variance_over_enriched_manifold(X, radius, orig_shape):
    rng = radius_neighbors_graph(X, radius, mode='distance')
    n_comps = connected_components(rng)[0]
    print(f"n_connected_components: {n_comps}")
    comp_sizes = np.unique(connected_components(rng)[1], return_counts=True)[1]
    main_comp_size = np.unique(connected_components(rng)[1][:orig_shape], return_counts=True)[1].max()
    print(comp_sizes)
    print(main_comp_size)

    dij = dijkstra(rng, directed=False, indices=np.arange(orig_shape))
    dij = dij[:, :orig_shape]
    dij[np.isinf(dij)] = 0
    v = (dij ** 2).sum().sum() / (2 * main_comp_size ** 2)
    return v



def estimate_variance_over_manifold(X, simulation=False):
    gap = np.unique(cdist(X, X))[2]
    radius = gap
    rng = radius_neighbors_graph(X, radius, mode='distance')
    n_comps = connected_components(rng)[0]
    print(f"n_connected_components: {n_comps}")
    n_comps_prev1, n_comps_prev2 = 0, 0
    while n_comps != n_comps_prev1 or n_comps_prev1 != n_comps_prev2:
        radius = radius + 10 if not simulation else radius + gap
        # radius += gap
        rng = radius_neighbors_graph(X, radius, mode='distance')
        n_comps_prev2 = n_comps_prev1
        n_comps_prev1 = n_comps
        n_comps = connected_components(rng)[0]
        print(f"n_connected_components: {n_comps}")
        if n_comps == 1:
            break

    print(radius)
    comp_sizes = np.unique(connected_components(rng)[1], return_counts=True)[1]
    main_comp_size = comp_sizes.max()
    not_outliers = np.where(connected_components(rng)[1] == comp_sizes.argmax())[0]
    print(comp_sizes)
    print(main_comp_size)

    dij = dijkstra(rng[not_outliers][:, not_outliers], directed=False)

    dij[np.isinf(dij)] = 0
    v = (dij ** 2).sum().sum() / (2 * main_comp_size ** 2)
    return v, radius, not_outliers


def manifold_learning(X):
    print(X.shape)
    intrinsic_dim = np.round(TwoNN().fit_transform(X)).astype(int)
    print(f"intrinsic dimension: {intrinsic_dim}")

    manifold = ManApprox(X.T)
    manifold.manifold_dim = intrinsic_dim

    manifold.calculateSigma()
    manifold.createTree()

    manifold, projected = denoise_manifold(manifold)

    return manifold, projected



def enrich(manifold, projected, n_points_to_enrich=10):
    # window = [-0.5 * manifold.sigma, 0.5 * manifold.sigma]
    # all_directions = [window for _ in range(manifold.manifold_dim)]
    # mesh = np.meshgrid(*all_directions)
    # indices = np.array([e.flatten() for e in mesh])

    if n_points_to_enrich == 0: # TODO - what if n_points == 1?
        return projected

    approximated_points = []
    for i, p in enumerate(projected):
        if i % 100 == 0:
            print(i)
        # print(i)
        try:
            projected_p, q, coeffs, Base, U = manifold.projectPointsGetPoly(p)
        except Exception as ex:
            print(ex)
            continue
        D = coeffs[np.sum(Base, axis=1) == 1, :]
        Q, _ = np.linalg.qr(D.T, mode='reduced')

        for_ball = np.random.uniform(-1, 1, size=(manifold.manifold_dim, n_points_to_enrich))
        for_ball = normalize(for_ball, axis=1, norm='l2')
        for_ball *= (manifold.sigma / 2)
        points_ball = Q @ for_ball + projected_p[np.newaxis].T

        # points_grid = Q @ indices + projected_p[np.newaxis].T

        U0 = None
        for point in points_ball.T:
            try:
                pp, _, _, _, U0 = manifold.projectPointsGetPoly(point, U0)
                if np.linalg.norm(point - pp) <= manifold.sigma * 0.5:
                    approximated_points.append(pp)
            except:
                continue
    enriched = np.concatenate((projected, approximated_points)) if len(approximated_points) else projected
    return enriched


data_dir = 'droplet_datasets'

filenames = [file for file in os.listdir(data_dir) if file.endswith('h5ad')]
filenames.sort(key=lambda filename: os.path.getsize(os.path.join(data_dir, filename)))

def project(csv_filename=None, to_enrich=False):
    df_cols = [TISSUE, CELLTYPE, K_HAT_YOUNG, INT_DIM_YOUNG, V_K_YOUNG, V_K_YOUNG_ENRICHED, V_D1_YOUNG, V_D2_YOUNG,
                                 K_HAT_OLD, INT_DIM_OLD, V_K_OLD, V_K_OLD_ENRICHED, V_D1_OLD, V_D2_OLD]
    output_df = pd.DataFrame([], columns=df_cols)

    for filename in filenames[5:]:
        tissue_name = filename.split('-')[-1].split('.')[0]
        print(tissue_name)
        adata = sc.read(os.path.join(data_dir, filename))
        X = adata.raw.X.todense()[:, adata.var['highly_variable']]
        adata = sc.AnnData(X, obs=adata.obs, var=adata.var[adata.var['highly_variable']])

        for celltype in adata.obs.cell_ontology_class.values.unique():
            print(celltype)
            adata_ct = adata[adata.obs.cell_ontology_class == celltype]
            is_young_ct = (adata_ct.obs.age == '1m') | (adata_ct.obs.age == '3m')
            adata_young = adata_ct[is_young_ct]
            is_old_ct = ~is_young_ct
            adata_old = adata_ct[is_old_ct]
            if is_young_ct.sum() == 0 or is_old_ct.sum() == 0:
                continue

            print("Run Biwhitening...")
            k_hat_young, ks_dist_young = find_k_hat(adata_young)
            if k_hat_young is None:
                continue

            print("Run Biwhitening...")
            k_hat_old, ks_dist_old = find_k_hat(adata_old)
            if k_hat_old is None:
                continue

            if is_young_ct.sum() < max(k_hat_young, 100) or is_old_ct.sum() < max(k_hat_old, 100):
                continue

            pca_young = PCA(n_components=k_hat_young.astype(int))
            X_young = pca_young.fit_transform(adata_young.X)

            pca_old = PCA(n_components=k_hat_old.astype(int))
            X_old = pca_old.fit_transform(adata_old.X)

            manifold_young, projected_young = manifold_learning(X_young)
            if projected_young is None:
                print("couldn't project due to sparsity")
                return None

            v_d1_young = mean_squared_error(adata_young.X,  pca_young.inverse_transform(X_young))
            v_d2_young = mean_squared_error(X_young, projected_young)

            v_k_young, radius, not_outliers = estimate_variance_over_manifold(projected_young)

            projected_young = projected_young[not_outliers]
            v_k_young_enriched = None
            if to_enrich:
                print(f"enrich_factor: {np.round(10_000 / projected_young.shape[0]).astype(int)}")
                enriched_young = enrich(manifold_young, projected_young, n_points_to_enrich=np.round(10_000 / projected_young.shape[0]).astype(int))
                v_k_young_enriched = estimate_variance_over_enriched_manifold(enriched_young, radius, orig_shape=projected_young.shape[0])
                print(v_k_young, v_k_young_enriched)

            print(f"{tissue_name}, {celltype} - young: dim: {manifold_young.manifold_dim}, v_d1: {v_d1_young}, v_d2: {v_d2_young}, v_k_metric: {v_k_young}")

            manifold_old, projected_old = manifold_learning(X_old)
            if projected_old is None:
                print("couldn't project due to sparsity")
                return None

            v_d1_old = mean_squared_error(adata_old.X, pca_old.inverse_transform(X_old))
            v_d2_old = mean_squared_error(X_old, projected_old)

            v_k_old, radius, not_outliers = estimate_variance_over_manifold(projected_old)

            projected_old = projected_old[not_outliers]
            v_k_old_enriched = None
            if to_enrich:
                print(f"enrich_factor: {np.round(10_000 / projected_old.shape[0]).astype(int) - 1}")
                enriched_old = enrich(manifold_old, projected_old, n_points_to_enrich=np.round(10_000 / projected_old.shape[0]).astype(int))
                v_k_old_enriched = estimate_variance_over_enriched_manifold(enriched_old, radius, orig_shape=projected_old.shape[0])
                print(v_k_old, v_k_old_enriched)

            print(f"{tissue_name}, {celltype} - old: dim: {manifold_old.manifold_dim}, v_d1: {v_d1_old}, v_d2: {v_d2_old}, v_k_metric: {v_k_old}")

            output_df = output_df.append({TISSUE: tissue_name, CELLTYPE: celltype,

                                          K_HAT_YOUNG: k_hat_young.astype(int), INT_DIM_YOUNG: manifold_young.manifold_dim,
                                          V_K_YOUNG: v_k_young, V_K_YOUNG_ENRICHED: v_k_young_enriched,
                                          V_D1_YOUNG: v_d1_young, V_D2_YOUNG: v_d2_young,

                                          K_HAT_OLD: k_hat_old.astype(int), INT_DIM_OLD: manifold_old.manifold_dim,
                                          V_K_OLD: v_k_old, V_K_OLD_ENRICHED: v_k_old_enriched,
                                          V_D1_OLD: v_d1_old, V_D2_OLD: v_d2_old},

                                         ignore_index=True)

            if csv_filename:
                output_df.to_csv(csv_filename)

            print('---------------------------------------------------------------------------------------------------')


def poc():
    for filename in [filenames[::-1][0]]:
        tissue_name = filename.split('-')[-1].split('.')[0]
        print(tissue_name)
        adata = sc.read(os.path.join(data_dir, filename))
        X = adata.raw.X.todense()[:, adata.var['highly_variable']]
        adata = sc.AnnData(X, obs=adata.obs, var=adata.var[adata.var['highly_variable']])

        for celltype in [adata.obs.cell_ontology_class.values.unique()[4]]:
            print(celltype)
            adata_ct = adata[adata.obs.cell_ontology_class == celltype]
            is_young_ct = (adata_ct.obs.age == '1m') | (adata_ct.obs.age == '3m')
            adata_young = adata_ct[is_young_ct]
            is_old_ct = ~is_young_ct
            adata_old = adata_ct[is_old_ct]
            if is_young_ct.sum() == 0 or is_old_ct.sum() == 0:
                continue

            print("Run Biwhitening...")
            k_hat_old, ks_dist_old = find_k_hat(adata_old)
            if k_hat_old is None:
                continue

            pca_old = PCA(n_components=k_hat_old.astype(int))
            X_old = pca_old.fit_transform(adata_old.X)

            l = []
            for _ in range(3):
                X = X_old[np.random.choice(X_old.shape[0], size=1000, replace=False)]

                manifold, projected = manifold_learning(X)
                if projected is None:
                    print("couldn't project due to sparsity")
                    return None

                v_k, radius, not_outliers = estimate_variance_over_manifold(projected)

                projected = projected[not_outliers]

                enriched = enrich(manifold, projected,
                                      n_points_to_enrich=np.round(10_000 / projected.shape[0]).astype(int) - 1)

                v_k_enriched = estimate_variance_over_enriched_manifold(enriched, radius, orig_shape=projected.shape[0])


                print(v_k, v_k_enriched)
                l.append(v_k_enriched)
                print(l)
            print(l, np.mean(l))


# if __name__ == '__main__':
#     project('real_data_PCA_BW_project_liver1.csv')
    # project('real_data_PCA_BW_enrich1.csv', to_enrich=True)
    # poc()
