"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.

Derivations and discussion are in derivations.pdf; "Section x.y" below
refers to that file.
"""
import numpy as np
import matplotlib.pyplot as plt


# PART 1 (20 points), Q1: Show computations for the given A and B.
# See PDF, Section 1.1.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    A = np.array([[3, 4], [2, 1]]) if A is None else np.asarray(A)
    B = np.array([[1, 2], [0, -1]]) if B is None else np.asarray(B)
    return A @ B, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(A @ B)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# See PDF, Section 1.2.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    M = np.array([[2., -1.], [-1., 2.]]) if M is None else np.asarray(M, dtype=float)
    return np.linalg.eigh(M)


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# See PDF, Section 1.3.

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# See PDF, Section 1.4.
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    C = np.array([[4., 7.], [2., 6.]]) if C is None else np.asarray(C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# See PDF, Section 2.1.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # X ~ Poisson(rate): c = e^-rate, mean = variance = rate
    return float(np.exp(-rate)), float(rate), float(rate)


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# See PDF, Section 2.2.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    p_ab = p_a_given_b * p_b
    return p_ab, p_ab / p_a, p_a + p_b - p_ab


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# See PDF, Section 2.3.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # MLE = h/n; MAP = (h+1)/(n+1) (see PDF); both also hold at heads=0 and heads=tosses
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# See PDF, Section 2.4.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    x = np.array([1., 3., 5., 7.]) if data is None else np.asarray(data, dtype=float)
    return float(x.mean()), float(np.mean((x - x.mean()) ** 2))


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# See PDF, Section 2.5.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # posterior mode = average of x and prior_mean weighted by precision (1/variance)
    w_x, w_0 = 1. / observation_variance, 1. / prior_variance
    return (w_x * x + w_0 * prior_mean) / (w_x + w_0)


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# See PDF, Section 2.6.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# See PDF, Section 3.1.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# See PDF, Section 3.2.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    mean = np.array([2., -1.]) if mean is None else np.array(mean, dtype=float)
    cov = (np.array([[3., 1.2], [1.2, 1.]]) if covariance is None
           else np.array(covariance, dtype=float))
    return np.random.default_rng(seed).multivariate_normal(mean, cov, n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# See PDF, Section 3.3.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (len(X) - 1)
    values, vectors = np.linalg.eigh(covariance)      # ascending order
    order = np.argsort(values)[::-1]
    eigenvalues = np.clip(values[order], 0., None)    # round-off can give -1e-17
    components = vectors[:, order]
    return {'mean': mean, 'centered': centered, 'covariance': covariance,
            'eigenvalues': eigenvalues, 'components': components,
            'scores': centered @ components}


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# See PDF, Section 3.4.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# See PDF, Sections 3.2-3.4.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    X = np.asarray(X, dtype=float)
    mean, V, lam = result['mean'], result['components'], result['eigenvalues']
    r = np.corrcoef(X.T)[0, 1]

    # Q2: the data
    fig1, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=.6, label='samples')
    ax.plot(*mean, 'kx', ms=10, mew=2, label='sample mean')
    ax.set(title=f'Synthetic data (sample correlation r = {r:.3f})',
           xlabel='$x_1$', ylabel='$x_2$', aspect='equal')
    ax.legend()

    # Q3: principal directions through the mean, each spanning +-2 std devs
    fig2, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=.6, label='samples')
    for k in range(V.shape[1]):
        d = 2 * np.sqrt(lam[k]) * V[:, k]            # half-length along PC k
        ax.plot([mean[0] - d[0], mean[0] + d[0]], [mean[1] - d[1], mean[1] + d[1]],
                lw=2.5, label=f'PC{k + 1} (variance {lam[k]:.3f})')
    ax.plot(*mean, 'kx', ms=10, mew=2)
    ax.set(title='Principal directions through the mean (±2σ)',
           xlabel='$x_1$', ylabel='$x_2$', aspect='equal')
    ax.legend()

    # Q4: fraction of total variance explained by each component
    fig3, ax = plt.subplots(figsize=(3.5, 2.6))
    frac = explained_variance(lam)
    bars = ax.bar([f'PC{k + 1}' for k in range(len(frac))], frac)
    ax.bar_label(bars, labels=[f'{100 * f:.1f}%' for f in frac])
    ax.set(title='Proportion of variance explained', xlabel='principal component',
           ylabel='fraction of total variance', ylim=(0, 1.1))

    for fig in (fig1, fig2, fig3):
        fig.tight_layout()
    return fig1, fig2, fig3


if __name__ == '__main__':
    # This block does not run when the grader imports your functions.
    import os

    np.set_printoptions(precision=4, suppress=True)

    AB, BA, ApBT, ABT, trA, trAB = matrix_operations()
    print('AB =', AB, 'BA =', BA, '(A+B)^T =', ApBT, 'AB^T =', ABT, sep='\n')
    print('trace(A) =', trA, ' trace(AB) =', trAB)
    values, vectors = eigensystem()
    print('eigenvalues =', values, 'eigenvectors (columns) =', vectors, sep='\n')
    print('C^-1 =', matrix_inverse(), sep='\n')

    print('(c, E[X], Var X) =', distribution_moments())
    print('(P(A∩B), P(B|A), P(A∪B)) =', event_probabilities())
    print('(MLE, MAP) =', coin_estimates())
    print('(mu, sigma^2) MLE =', gaussian_mle())
    print('mu MAP =', gaussian_map())

    X = generate_data()
    result = pca(X)
    print('sample mean =', result['mean'])
    print('sample covariance =', result['covariance'], sep='\n')
    print('sample correlation =', np.corrcoef(X.T)[0, 1])
    print('eigenvalues =', result['eigenvalues'])
    print('components (columns) =', result['components'], sep='\n')
    print('explained variance =', explained_variance(result['eigenvalues']))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(out, exist_ok=True)
    for name, fig in zip(('data', 'pca', 'explained_variance'), make_plots(X, result)):
        fig.savefig(os.path.join(out, name + '.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(out, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close('all')
