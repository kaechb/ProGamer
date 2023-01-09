import jetnet
from os.path import exists
from scipy import linalg
from scipy.stats import iqr
from scipy.optimize import curve_fit
import numpy as np
def linear(x, intercept, slope):
    return intercept + slope * x
def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        )
        logging.debug(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def frechet_gaussian_distance(X, Y, normalise: bool = True) -> float:
    if normalise:
        X, Y = normalise_features(X, Y)

    mu1 = np.mean(X, axis=0)
    sigma1 = np.cov(X, rowvar=False)
    mu2 = np.mean(Y, axis=0)
    sigma2 = np.cov(Y, rowvar=False)

    return _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def normalise_features(X ,Y=None):
    maxes = np.max(np.abs(X), axis=0)

    return (X / maxes, Y / maxes) if Y is not None else X / maxes
def fpd_infinity(
    X,
    Y,
    min_samples: int = 5_000,
    max_samples: int = 50_000,
    num_batches: int = 10,
    num_points: int = 200,
    seed: int = 42,
    normalise: bool = True,
    n_jobs=None,
):
    if normalise:
        X, Y = normalise_features(X, Y)

    # Choose the number of images to evaluate FID_N at regular intervals over N
    batches = (1 / np.linspace(1.0 / min_samples, 1.0 / max_samples, num_points)).astype("int32")
    # batches = np.linspace(min_samples, max_samples, num_points).astype("int32")

    np.random.seed(seed)

    vals = []

    for i, batch_size in enumerate(batches):
        vals_point = []
        for _ in range(num_batches):
            rand1 = np.random.choice(len(X), size=batch_size)
            rand2 = np.random.choice(len(Y), size=batch_size)

            rand_sample1 = X[rand1]
            rand_sample2 = Y[rand2]

            val = frechet_gaussian_distance(rand_sample1, rand_sample2, normalise=False)
            vals_point.append(val)

        vals.append(np.mean(vals_point))

    vals = np.array(vals)

    params, covs = curve_fit(linear, 1 / batches, vals, bounds=([0, 0], [np.inf, np.inf]))

    return (params[0], np.sqrt(np.diag(covs)[0]))
def fgd(real_scaled,fake_scaled,n,p):
    gen_efps = jetnet.utils.efps(fake_scaled.detach().numpy(), efpset_args=[("d<=", 4)], efp_jobs=None)
    efp_file =f"/beegfs/desy/user/kaechben/efps/"+p+"_"+str(n)+".npy"
    if exists(efp_file):
        real_efps = np.load(efp_file)
    else:
        real_efps = jetnet.utils.efps(
            real_scaled.detach().numpy(), efpset_args=[("d<=", 4)], efp_jobs=None
        )
        np.save(efp_file, real_efps)

    return fpd_infinity(real_efps, gen_efps, n_jobs=None)
