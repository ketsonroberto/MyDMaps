import unittest
import numpy as np
import scipy as sp
import scipy.spatial.distance as sd
from sklearn.datasets import make_swiss_roll


class DiffusionMaps:
    """
    Diffusion maps is a nonlinear dimensionality reduction technique for embedding high-dimensional data into a
    low-dimensional Euclidean space revealing their intrinsic geometric structure.
    """

    def __init__(self):
        # Attributes of ``DiffusionMaps``.
        self.kernel_matrix = None  # Kernel matrix.
        self.transition_matrix = None  # Kernel matrix.
        self.X = None  # Input data.
        self.diffusion_coordinates = None  # Diffusion Coordinates.

    def fit(self, X=None, epsilon=None):
        """
        Compute the diffusion coordinates using fit.

        **Input:**
        * **X** (`ndarray`)
            Matrix of input data: row (samples) and columns (features).

        * **epsilon** (`float`)
            Lenght-scale parameter of the kernel used to determine the similarity of paris of data points (default = 1).

        **Output/Returns:**
            Instantiate the attributes of ``DiffusionMaps``.
        """

        # `X` must be a `ndarray` matrix (X.shape == 2) where the rows are the samples and columns the coordinates.
        if isinstance(X, np.ndarray):
            if len(X.shape) != 2:
                raise ValueError('Not acceptable shape for `X`.')
        else:
            # Show an error message if `X` is not a ndarray.
            raise TypeError('`X` must be `ndarray`.')

        if epsilon is None:
            raise TypeError('`epsilon` cannot be `NoneType`.')
        else:
            # Check the provided value of `epsilon` which must be either `float` or `int` larger than zero.
            if isinstance(epsilon, (float, int)):
                epsilon = float(epsilon)
                if epsilon < 0:
                    raise ValueError('`epsilon` must be larger than zero.')
            else:
                raise TypeError('`epsilon` must be either `float` or `int`.')

        # Construct the Gaussian Kernel Matrix (`self.kernel_matrix`).
        self.X = X
        dX = sd.squareform(sd.pdist(X))  # Compute the distance matrix for the elements in `X`.
        self.kernel_matrix = np.exp(-np.square(dX) / (4 * epsilon))  # Compute the kernel matrix.

        # Compute the transition matrix (`self.transition_matrix`).
        self._get_transition_matrix()

        # Get the diffusion coordinates (`self.diffusion_coordinates`): coordinates in a low-dimensional space.
        self._get_diffusion_coordinates()

    def _get_transition_matrix(self):
        """
        Private: Get the Markov matrix used to determine the low-dimensional representation of the input data in `X`.

        **Output/Returns:**
            The instantiation of `self.transition_matrix`.
        """

        # If no `kernel_matrix` is construct, show an error message.
        if self.kernel_matrix is None:
            raise TypeError('`kernel_matrix` not found!')

        # Compute d^{-1/2}, where the matrix d is given by - d(i,i) = sum(kernel_matrix(i,j),j)
        d_alpha = np.diag(np.power(self.kernel_matrix.sum(axis=1).flatten(), -0.5))

        # Get the Laplacian normalization: compute L^alpha = D^(-alpha)*L*D^(-alpha).
        l_star = d_alpha.dot(self.kernel_matrix.dot(d_alpha))

        # Compute the inverse of d_star(i,i) = sum(L_star(i,j),j).
        d_star_inv = np.diag(np.power(l_star.sum(axis=1).flatten(), -1))

        # Compute the transition matrix (Markov matrix).
        self.transition_matrix = d_star_inv.dot(l_star)

    def _get_diffusion_coordinates(self):
        """
        Private: Get the diffusion coordinates via the eigendecomposition of of `self.transition_matrix`.

        **Output/Returns:**
            The instantiation of `self.diffusion_coordinates`.
        """

        # Show an error message `self.kernel_matrix` is not provided.
        if self.transition_matrix is None:
            raise TypeError('`transition_matrix` not found!')

        # Perform the eigendecomposition of the transition matrix (`self.transition_matrix`) using Scipy.
        eigenvalues, eigenvectors = sp.linalg.eig(self.transition_matrix)

        # Reverse sort of eigenvalues: from larger to smaller.
        indices = np.argsort(np.abs(eigenvalues))[::-1]

        # Ensure that the eigenvectors and eigenvalues are real-valued and compute the diffusion coordinates
        self.diffusion_coordinates = np.real(eigenvectors[:, indices]) * np.real(eigenvalues[indices])


# ============================================ Unit Test of DiffusionMaps ==============================================
class MyTestCase(unittest.TestCase):
    # Problem: Finding the low-dimensional representation of a point cloud in 3-D.
    # Solution: Diffusion Maps is a technique used to find low-dimensional representation of high-dimensional data.
    # Test 1: test the if the number of diffusion coordinates is equal to the number of data points in `X`.
    # Test 2: test if the code correctly raise an exception for the shape of the input data `X`.

    # Test 1: test the if the number of diffusion coordinates is equal to the number of data points in `X`.
    def test_length_coordinates(self):
        X, _ = make_swiss_roll(n_samples=1000)  # Sample `n_samples` points from the Swiss Roll manifold.
        dfm = DiffusionMaps()  # Object of `DiffusionMaps`.
        dfm.fit(X=X, epsilon=1.0)  # Instantiate the attributes of `DiffusionMaps` with `epsilon` = 1.

        # Test if the number of data points in `X` is equal to the length of diffusion_coordinates.
        self.assertEqual(np.shape(X)[0], np.shape(dfm.diffusion_coordinates)[0])

    # Test 2: test if the code correctly raise an exception for the shape of the input data `X`.
    def test_input_shape_exception(self):
        # Get a random array with shape (100, 3, 1). DiffusionMaps only accepts len(np.shape(`X`)) = 2.
        X = np.random.rand(100, 3, 1)
        dfm = DiffusionMaps()  # Object of `DiffusionMaps`.

        # Get the exception for raise ValueError.
        with self.assertRaises(ValueError) as exception_context:
            dfm.fit(X=X, epsilon=1.0)  # Instantiate the attributes of `DiffusionMaps` with `epsilon` = 1.

        # The code will pass the test when it will raise the following exception.
        self.assertEqual(str(exception_context.exception), 'Not acceptable shape for `X`.')


if __name__ == '__main__':
    unittest.main()
