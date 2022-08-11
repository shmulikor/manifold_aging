import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from pyRMT import marcenkoPastur

warnings.filterwarnings('ignore')
from scipy.stats import kstest, rv_continuous, ks_2samp
from collections import namedtuple

KstestResult = namedtuple('KstestResult', ('statistic', 'pvalue'))

POISSON = 'poisson'
NB = 'NB'
NB_FAILURES = 'NB_failures'

simulation_types = [POISSON, NB, NB_FAILURES]

figures_dir_name = 'figures'

class MP(rv_continuous):
    def __init__(self, beta_range, rho, xtol=1e-14, seed=None):
        super().__init__(a=0, xtol=xtol, seed=seed)
        self.beta_range = beta_range
        self.rho = rho

    def _cdf(self, x):
        if isinstance(x, (list, np.ndarray)):
            return np.array([self._cdf(xx) for xx in x])
        pdf_range = np.linspace(self.beta_range[0], x, 1000)
        pdf_vals = np.array([self.rho(num) for num in pdf_range])
        return 0 if x < self.beta_range[0] else 1 if x > self.beta_range[1] else np.sum(np.diff(pdf_range) * pdf_vals[:-1])

    def use_cdf(self, x):
        if isinstance(x, (list, np.ndarray)):
            return np.array([self._cdf(xx) for xx in x])
        return self._cdf(x)


def get_MP_cdf(beta_range, rho):

    def MP_cdf(x):
        if isinstance(x, (list, np.ndarray)):
            return np.array([MP_cdf(xx) for xx in x])

        pdf_range = np.linspace(beta_range[0], x, 1000)
        pdf_vals = np.array([rho(num) for num in pdf_range])
        return 0 if x < beta_range[0] else 1 if x > beta_range[1] else np.sum(np.diff(pdf_range) * pdf_vals[:-1])

    return MP_cdf


def sinkhorn_knopp(A, epsilon=1e-9):
    assert isinstance(A, np.ndarray)
    m, n = A.shape
    x = np.ones(m)
    y = np.ones(n)
    r = n * np.ones(m)
    c = m * np.ones(n)
    iters = 0

    while np.max(np.abs(np.sum(np.multiply(np.multiply(x[:, None], A), y), axis=1) - r)) > epsilon or \
            np.max(np.abs(np.sum(np.multiply(np.multiply(x[:, None], A), y), axis=0) - c)) > epsilon and iters < 100:
        y = c / np.sum(np.multiply(A.T, x), axis=1)
        x = r / np.sum(np.multiply(A, y), axis=1)

        iters += 1
        # print(iters)

    if iters >= 100:
        print("Sinkhorn-Knopp did not converge,terminates. Bad results")

    return x, y


def generate_poisson_data(m=300, n=1000, r=10, log_mean=0, log_sigma=2):
    # assert m < n
    B = np.random.lognormal(mean=log_mean, sigma=log_sigma, size=(m, r))
    C = np.random.uniform(size=(r, n))
    X = B @ C
    X /= np.mean(X)
    Y = np.random.poisson(X)

    return Y


def find_alpha_beta(Y, plot=False):

    if Y.shape[0] > Y.shape[1]:
        Y = Y.T

    ks_dist = []
    alphas = []
    betas = np.arange(0, 1.05, 0.05)
    for beta in betas:
        # print(beta)
        alpha = 1
        var_Y = alpha * ((1 - beta) * Y + beta * Y ** 2)
        u_hat, v_hat = sinkhorn_knopp(var_Y)

        Y_hat = np.multiply(np.multiply(np.sqrt(u_hat)[:, None], Y), np.sqrt(v_hat))
        sigma_hat = (Y_hat @ Y_hat.T) / Y.shape[1]
        sigma_hat_eigs = np.linalg.eigvals(sigma_hat)
        lambda_med = np.median(sigma_hat_eigs)

        beta_range, marchenko_pdf = marcenkoPastur(Y)
        pdf_range = np.linspace(beta_range[0], beta_range[1], 1000)
        pdf_vals = np.array([marchenko_pdf(num) for num in pdf_range])
        mu_idx = np.where(np.cumsum(np.diff(pdf_range) * pdf_vals[:-1]) > 0.5)[0][0]
        mu = pdf_range[mu_idx]
        alpha = lambda_med / mu

        alpha_inv_sigma_hat_eigs = sigma_hat_eigs / alpha
        esd = lambda x: (alpha_inv_sigma_hat_eigs <= x).sum() / Y.shape[0]
        esd_vals = np.array([esd(num) for num in pdf_range])
        pdf_integral = [np.sum(np.diff(pdf_range)[:i] * pdf_vals[:i]) for i in range(len(pdf_range))]
        # ks = np.max(np.abs(esd_vals - pdf_integral))
        ks = ks_2samp(esd_vals, pdf_integral).statistic
        ks_dist.append(ks)
        alphas.append(alpha)

    beta = np.min(betas[np.where(ks_dist == np.min(ks_dist))])
    alpha = alphas[np.min(np.where(betas == beta))]

    print(f"beta = {beta}, alpha = {alpha}")

    if plot:
        plt.plot(betas, ks_dist)
        plt.axvline(x=beta, c='red', linestyle='--', label='estimated beta')
        plt.title("KS distance vs. beta")
        plt.xlabel('beta')
        plt.ylabel('KS distance')
        plt.legend()
        # todo - enable save of the figure?
        plt.show()

    return alpha, beta


def compare_uv(Y, alpha, beta):
    var_Y = alpha * ((1 - beta) * Y + beta * Y ** 2)
    u_hat, v_hat = sinkhorn_knopp(var_Y)
    u, v = sinkhorn_knopp(Y)
    u /= np.linalg.norm(u, ord=1)
    v /= np.linalg.norm(v, ord=1)
    uv = np.concatenate((u, v))
    u_hat /= np.linalg.norm(u_hat, ord=1)
    v_hat /= np.linalg.norm(v_hat, ord=1)
    uv_hat = np.concatenate((u_hat, v_hat))

    plt.scatter(uv, uv, alpha=0.5, label='True')
    plt.scatter(uv, uv_hat, alpha=0.5, label='Estimator')
    plt.title(f"u,v true vs. estimator - alpha = {alpha}, beta = {beta}")
    plt.legend()
    plt.show()


def generate_NB_data(m=1000, n=2000, r=10, n_failures=3):
    U = np.exp(2 * np.random.normal(size=(m, r)))
    V = np.exp(np.random.normal(size=(r, n)))
    X = U @ V
    X /= np.mean(X)
    if n_failures == 0:
        n_failures = np.random.randint(1, 11, size=(m, n))
    p = X / (n_failures + X)
    Y = np.random.negative_binomial(n_failures, 1-p)

    return Y


def plot_eig_values_and_marchenko(Y, beta_range, singular_values=None, plot=False, name_to_save=None, after_BW=False):

    if singular_values is None:
        _, singular_values, _ = np.linalg.svd(Y / np.sqrt(Y.shape[1]))

    w = singular_values ** 2
    w.sort()

    if plot or name_to_save:
        plt.bar(range(len(w)), sorted(w)[::-1])
        plt.axhline(y=beta_range[1], color='red', label='MP upper edge')
        plt.legend()
        plt.ylim(0, sorted(w)[-10])
        # plt.xlim(0, len(w) / 2)
        plt.title("Sorted eigenvalues")
        plt.xlabel("k")
        ylabel = "$\lambda_{k}(n^{-1} \hat{Y} \hat{Y}^{T})$" if after_BW else "$\lambda_{k}(n^{-1} Y Y^{T})$"
        plt.ylabel(ylabel)
        if isinstance(name_to_save, str):
            if not os.path.isdir(figures_dir_name):
                os.mkdir(figures_dir_name)
            plt.savefig(f"{figures_dir_name}\\{name_to_save}")
        plt.show() if plot else plt.close()

    return w


def plot_density_and_marchenko(w, beta_range, rho, name_to_save=None, plot=False, after_bw=False):
    restricted_w = [eig for eig in w if beta_range[0] < eig < beta_range[1]]

    if plot or name_to_save:
        rho_range = np.linspace(beta_range[0], beta_range[1], 100)
        hist_label = 'Eigenvalues of $n^{-1} \hat{Y} \hat{Y}^{T}$ in the MP bulk' if after_bw else 'Eigenvalues of $n^{-1} Y Y^{T}$ in the MP bulk'
        plt.hist([eig for eig in w if beta_range[0] < eig < beta_range[1]], bins=30, density=True, label=hist_label)
        plt.scatter(rho_range, [rho(num) for num in rho_range], color='red', label='MP density')
        plt.legend()
        plt.title("Eigenvalue density")
        plt.xlabel("Eigenvalues")
        plt.ylabel("Density")
        if isinstance(name_to_save, str):
            if not os.path.isdir(figures_dir_name):
                os.mkdir(figures_dir_name)
            plt.savefig(f"{figures_dir_name}\\{name_to_save}")
        plt.show() if plot else plt.close()

    cdf = get_MP_cdf(beta_range, rho)
    ks_dist = kstest(restricted_w, cdf) if len(restricted_w) else KstestResult(np.inf, 0)
    print(ks_dist)

    return ks_dist


def mp_process(Y, plot=False):
    beta_range, rho = marcenkoPastur(Y)
    w = plot_eig_values_and_marchenko(Y, beta_range, plot=plot)
    ks_dist = plot_density_and_marchenko(w, beta_range, rho, plot=plot)

    return ks_dist



def run_Biwhitening(Y=None, simulation_type=None, alpha=None, beta=None, name_to_save=None, plot=False):
    assert Y is None or isinstance(Y, np.ndarray)
    # transposed = False
    if Y is None:
        assert simulation_type in simulation_types
        if simulation_type == POISSON:
            Y = generate_poisson_data()
        elif simulation_type == NB:
            Y = generate_NB_data()
        else:
            Y = generate_NB_data(n_failures=0)

    else:
        if Y.shape[0] > Y.shape[1]:
            Y = Y.T
            # transposed = True
    if alpha is not None and beta is not None:
        var_Y = alpha * ((1 - beta) * Y + beta * Y ** 2)
    else:
        alpha, beta = find_alpha_beta(Y, plot=plot)
        var_Y = alpha * ((1 - beta) * Y + beta * Y ** 2)

    beta_range, rho = marcenkoPastur(Y)

    w = plot_eig_values_and_marchenko(Y, beta_range, plot=plot,
                                      name_to_save=f"{name_to_save}_eigvals_before" if name_to_save else None)
    plot_density_and_marchenko(w, beta_range, rho, plot=plot,
                               name_to_save=f"{name_to_save}_MP_before" if name_to_save else None)

    # main algorithm
    x, y = sinkhorn_knopp(var_Y)
    beta_range, rho = marcenkoPastur(Y)
    Y_hat = np.multiply(np.multiply(np.sqrt(x)[:, None], Y), np.sqrt(y))
    u, s, v_t = np.linalg.svd(Y_hat)
    threshold = np.sum(np.sqrt(Y.shape))
    rank_hat = (s > threshold).sum()
    print(f"rank_hat = {rank_hat}")

    s /= np.sqrt(Y_hat.shape[1])
    w_hat = plot_eig_values_and_marchenko(Y_hat, beta_range, singular_values=s, plot=plot, after_BW=True,
                                          name_to_save=f"{name_to_save}_eigvals_after" if name_to_save else None)
    ks_dist = plot_density_and_marchenko(w_hat, beta_range, rho, plot=plot, after_bw=True,
                                         name_to_save=f"{name_to_save}_MP_after" if name_to_save else None)

    return rank_hat, ks_dist

def run_on_adata(adata, alpha=None, beta=None, plot=False, name_to_save=False):
    assert isinstance(adata, sc.AnnData)
    Y = adata.X
    rank_hat, ks_dist = run_Biwhitening(Y, alpha=alpha, beta=beta, name_to_save=name_to_save, plot=plot)
    return rank_hat, ks_dist



# if __name__ == '__main__':
#     k_hat, ks_dist = run_Biwhitening(simulation_type=POISSON, plot=True)
#     k_hat, ks_dist = run_Biwhitening(simulation_type=NB, plot=True)
