# MyDMaps
This repository contains the Python files and Jupiter notebooks running several examples of Diffusion Maps.

1. Introduction

Objective and methods
Although the codes in TestDiffusionMaps.py are self-explanatory due to the included comments, this supplementary document contains a more detailed description of the unsupervised learning technique (Diffusion Maps) implemented in ``TestDiffusionMaps.py``.

The main objective of the presented code is to show how the implementation of Diffusion Maps can be simple and powerful. Moreover, two simple examples of unit tests are implemented to verify the code reliability. The classes in ``TestDiffusionMaps.py`` were implemented in Python 3.9 using the oriented-object programming (OOP) paradigm, and the examples were run on a computer with macOS. Further, the code requires the following Python toolboxes numpy, scipy, and scikit-learn. 

2. Theory of Diffusion Maps

Nonlinear dimensionality reduction techniques consider that high-dimensional data can lie on a low-dimensional manifold. To reveal this embedded low-dimensional structure, one can resort to kernel-based techniques such as Diffusion Maps [coifman 2006]; where the spectral decomposition of the transition matrix of a random walk performed on the data is used to determine a new set of coordinates, also known as diffusion coordinates, embedding this manifold into a space of reduced dimension. For example, data observed in a 3-D space can be constrained to a 2-D structure that can be revealed by the diffusion coordinates.


3. Class DiffusionMaps

The Diffusion Maps framework is implemented as a python class in ``TestDiffusionMaps.py``. Next, some elements of ``DiffusionMaps`` are discussed. First, the attributes of ``DiffusionMaps`` are the following:

* Kernel matrix (`kernel_matrix`)
* Transition matrix (`transition_matrix`)
* Input data (`X`) 
* Diffusion coordinates (`diffusion_coordinates`)

as presented in the following piece of code.

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


This class attributes are instantiated by using the method ``fit``; where `X` is the input data, and `epsilon` is the length-scale parameter of the Gaussiann Kernel.

    def fit(self, X=None, epsilon=None):

Once the class ``DiffusionMaps`` is instantiated, the attributes and methods are accessible from the object using OOP paradigm with Python.

4. Class MyTestCase

The file ``TestDiffusionMaps.py`` also contain the class ``MyTestCase`` for performing the unit tests in two distinct cases. The first test (presented next) is used to compare the number of points in `diffusion_coordinates` and the number of points in the the input dataset `X`. Therefore, the diffusion coordinates must be consistent with the input dataset.

    def test_length_coordinates(self):
        # Test 1 test the if the number of diffusion coordinates is equal to the number of data points in `X`.

        X, _ = make_swiss_roll(n_samples=1000)  # Sample `n_samples` points from the Swiss Roll manifold.
        dfm = DiffusionMaps()  # Object of `DiffusionMaps`.
        dfm.fit(X=X, epsilon=1.0)  # Instantiate the attributes of `DiffusionMaps` with `epsilon` = 1.

        # Test if the number of data points in `X` is equal to the length of diffusion_coordinates.
        self.assertEqual(np.shape(X)[0], np.shape(dfm.diffusion_coordinates)[0])


The second unit test (presented next) check the raise of exceptions in the code. In particular, it verifies if `raise ValueError` is called when the shape of `X` is not acceptable. In this implementation of Diffusion Maps `X` must be an array with two dimensions (row and columns), otherwise the code shows the following error message: ``Not acceptable shape for `X` ``. Therefore, the code will pass this test only if it raises an exception for this particular condition.

    def test_input_shape_exception(self):
        # Test 2: test if the code correctly raise an exception for the shape of the input data `X`.
        
        # Get a random array with shape (100, 3, 1). DiffusionMaps only accepts len(np.shape(`X`)) = 2.
        X = np.random.rand(100, 3, 1)
        dfm = DiffusionMaps()  # Object of `DiffusionMaps`.

        # Get the exception for raise ValueError.
        with self.assertRaises(ValueError) as exception_context:
            dfm.fit(X=X, epsilon=1.0)  # Instantiate the attributes of `DiffusionMaps` with `epsilon` = 1.

        # The code will pass the test when it will raise the following exception.
        self.assertEqual(str(exception_context.exception), 'Not acceptable shape for `X`.')
      
5. Running DiffusionMaps

This section shows how to run one example using ``DiffusionMaps``. The problem consist of unwrapping the Swiss Roll manifold, which is defined in a 3-D space, and it is implicitly included in the first unit test in ``TestDiffusionMaps.py``. Herein, the Swiss Roll manifold is sampled, and a 2,000 point cloud of points in the 3-D space represent this surface. Thus, one can use the following scikit-learn command to get samples from the Swiss Roll manifold:

    from sklearn.datasets import make_swiss_roll
    X, color = make_swiss_roll(n_samples=2000, random_state=1)

One can easily see that the Swiss Roll manifold is defined in 3-D, but it has an intrinsic 2-D structure. Thus, one can use Diffusion Maps to unwrap this 3-D structure. Using the code presented herein, one can obtain the representation of the Swiss Roll manifold in a 2-D space. To this aim, one can use the following commands to instantiate ``DiffusionMaps``. 

    dfm = DiffusionMaps()
    dfm.fit(X=X, epsilon=1.0)

One can observe that the an object of ``DiffusionMaps`` (`dfm`) is created without input arguments. To instantiate the attributes one can use the method ``fit``, which receives the input dataset `X` and the value of `epsilon` equal to 1.0 (which is selected by the user).

6. Running MyTestCase

To run the unit tests, one can use the following command in the directory containing ``TestDiffusionMaps.py``:

    $ python -m unittest TestDiffusionMaps

Therefore, one can expect the following outcome:

    ..
    --------------------------------------------------------------------
    Ran 2 tests in 0.514s

    OK

[coifman 2006]  R. R. Coifman and S. Lafon.  Diffusion maps.Applied and ComputationalHarmonic Analysis, 21(1):5 – 30, 2006.  Special Issue:  Diffusion Maps andWavelets.

