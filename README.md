# MyDMaps
This repository contains the Python files and Jupiter notebooks running several examples of Diffusion Maps.

Introduction

Objective and methods
Although the codes in TestDiffusionMaps.py are self-explanatory due to the included comments, this supplementary document contains a more detailed description of the unsupervised learning technique (Diffusion Maps) implemented in TestDiffusionMaps.py.

The main objective of the presented code is to show how the implementation of Diffusion Maps can be simple and powerful. Moreover, two simple examples of unit tests are implemented to verify the code reliability. The classes in \texttt{TestDiffusionMaps.py} were implemented in Python 3.9 using the oriented-object programming (OOP) paradigm, and the examples were run on a computer with macOS. Further, the code requires the following Python toolboxes numpy, scipy, and scikit-learn. 

Theory of Diffusion Maps

Nonlinear dimensionality reduction techniques consider that high-dimensional data can lie on a low-dimensional manifold. To reveal this embedded low-dimensional structure, one can resort to kernel-based techniques such as Diffusion Maps [coifman 2006]; where the spectral decomposition of the transition matrix of a random walk performed on the data is used to determine a new set of coordinates, also known as diffusion coordinates, embedding this manifold into a space of reduced dimension. For example, data observed in a 3-D space can be constrained to a 2-D structure that can be revealed by the diffusion coordinates.


Class DiffusionMaps

The Diffusion Maps framework is implemented as a python class in TestDiffusionMaps.py. Next, some elements of the ``DiffusionMaps`` are discussed. First, the attributes of \texttt{DiffusionMaps} are the following:
\begin{itemize}
    \item Kernel matrix $\mathbf{K}$ (\texttt{kernel\underline{\hspace{.1in}}matrix})
    \item Transition matrix $\mathbf{P}$ (\texttt{transition\underline{\hspace{.1in}}matrix})
    \item Input data $S_{\mathbf{X}}$ (\texttt{X}) 
    \item Diffusion coordinates $\boldsymbol{\psi}_i$ (\texttt{diffusion\underline{\hspace{.1in}}coordinates})
\end{itemize}
\noindent
as presented in the following piece of code.

\begin{lstlisting}[language=Python, caption=class DiffusionMaps.]
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
\end{lstlisting}\label{ls:dmaps1}

This class attributes are instantiated by using the method \texttt{fit}; where \texttt{X} is the input data equivalent to $S_{\mathbf{X}}$, and \texttt{epsilon} is the length-scale parameter $\epsilon$ in Eq. \ref{eq:gaussian_kernel}.
\begin{lstlisting}[language=Python, caption=Instantiating the attributes of DiffusionMaps]
    def fit(self, X=None, epsilon=None):
\end{lstlisting}\label{ls:dmaps2}
\noindent
Once the class \texttt{DiffusionMaps} is instantiated, the attributes and methods are accessible from the object using OOP paradigm with Python.

\section{Class MyTestCase}
\label{class_my_test}
The file \texttt{TestDiffusionMaps.py} also contain the class \texttt{MyTestCase} for performing the unit tests in two distinct cases. The first test (Listing \ref{ls:test1}) is used to compare the number of points in \texttt{diffusion\underline{\hspace{.1in}}coordinates} and the number of points in the the input dataset \texttt{X}. Therefore, the diffusion coordinates must be consistent with the input dataset.

\begin{lstlisting}[language=Python, caption=Unit test 1.]
# Test 1: test the if the number of diffusion coordinates is equal to the number of data points in `X`.
def test_length_coordinates(self):
    X, _ = make_swiss_roll(n_samples=1000)  # Sample `n_samples` points from the Swiss Roll manifold.
    dfm = DiffusionMaps()  # Object of `DiffusionMaps`.
    dfm.fit(X=X, epsilon=1.0)  # Instantiate the attributes of `DiffusionMaps` with `epsilon` = 1.

    # Test if the number of data points in `X` is equal to the length of diffusion_coordinates.
    self.assertEqual(np.shape(X)[0], np.shape(dfm.diffusion_coordinates)[0])
\end{lstlisting}\label{ls:test1}

The second unit test (Listing 4) check the raise of exceptions in the code. In particular, it verifies if \texttt{raise ValueError} is called when the shape of \texttt{X} is not acceptable. In this implementation of Diffusion Maps \texttt{X} must be an array with two dimensions (row and columns), otherwise the code shows the following error message: \texttt{Not acceptable shape for `X`}. Therefore, the code will pass this test only if it raises an exception for this particular condition.
\begin{lstlisting}[language=Python, caption=Unit test 2.]
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
\end{lstlisting}\label{ls:test2}

\section{Running DiffusionMaps}
\label{run_dmaps}
This section shows how to run one example using \texttt{DiffusionMaps}. The problem consist of unwrapping the Swiss Roll manifold, which is defined in a 3-D space, and it is implicitly included in the unit test presented in Listing \ref{ls:test1}. Herein, the Swiss Roll manifold is sampled, and a point cloud of points in the 3-D space represent this surface, as observed in Fig. \ref{fig:swiss_roll} when 2,000 samples are generated using the following scikit-learn command:
\begin{lstlisting}[language=Python, caption=Sampling the Swiss Roll manifold.]
from sklearn.datasets import make_swiss_roll
X, color = make_swiss_roll(n_samples=2000, random_state=1)
\end{lstlisting}

One can easily see that the Swiss Roll manifold is defined in 3-D, but it has an intrinsic 2-D structure. Thus, one can use Diffusion Maps to unwrap this 3-D structure. Using the code presented herein, one can obtain the representation of the Swiss Roll manifold in a 2-D space. To this aim, one can use the following commands to instantiate the \texttt{DiffusionMaps} class. 
\begin{lstlisting}[language=Python, caption=Instantiating the class DiffusionMaps.]
dfm = DiffusionMaps()
dfm.fit(X=X, epsilon=1.0)
\end{lstlisting}
One can observe that the an object of \texttt{DiffusionMaps} (\texttt{dfm}) is created without input arguments. To instantiate the attributes one can use the method \texttt{fit}, which receives the input dataset \texttt{X} and the value of \texttt{epsilon} equal to 1.0 (which is selected by the user).
\begin{figure}[!ht]
	\centering
	\captionsetup{justification=centering}
	\includegraphics[scale=0.35]{images/swiss_roll.pdf}  
% 	\vspace{-1.5em}
	\caption{Example 1: Grassmannian diffusion manifold: a) training set for GH, and b) predicted Grassmannian diffusion manifold for 3,000 additional samples.}
% 	\vspace{-0.5em}
	\label{fig:swiss_roll}
\end{figure}

One of the attributes of \texttt{DiffusionMaps} is \texttt{diffusion\underline{\hspace{.1in}}coordinates} (presented as $\boldsymbol{\psi}_i$ in Section \ref{dmaps}), which stores the diffusion coordinates used to embed the Swiss Roll manifold into a 2-D space. This embedding is presented in Fig. \ref{fig:dmaps}, where the diffusion coordinates $\psi_2$ (\texttt{diffusion\underline{\hspace{.1in}}coordinates}[:, 2]), $\psi_3$ (\texttt{diffusion\underline{\hspace{.1in}}coordinates}[:, 3]), $\psi_4$ (\texttt{diffusion\underline{\hspace{.1in}}coordinates}[:, 4]), and $\psi_5$ (\texttt{diffusion\underline{\hspace{.1in}}coordinates}[:, 5]) are plotted with respect to $\psi_1$ (\texttt{diffusion\underline{\hspace{.1in}}coordinates}[:, 1]). \textbf{It is important mentioning that the 0th diffusion coordinates ($\psi_0$) are not used in this kind of embedding because they represent the trivial eigendirection. Therefore, when plotting the diffusion coordinates there is no need to show \texttt{diffusion\underline{\hspace{.1in}}coordinates}[:, 0]}. Based on Fig. \ref{fig:dmaps} one can observe that $\psi_1$ and $\psi_5$ are the directions that unwrap the Swiss Roll manifold.
\begin{figure}[!ht]
	\centering
	\captionsetup{justification=centering}
	\includegraphics[scale=0.45]{images/diffmaps.pdf}  
% 	\vspace{-1.5em}
	\caption{Example 1: Grassmannian diffusion manifold: a) training set for GH, and b) predicted Grassmannian diffusion manifold for 3,000 additional samples.}
% 	\vspace{-0.5em}
	\label{fig:dmaps}
\end{figure}

\section{Running MyTestCase}
To run the unit tests, one can use the following command in the directory containing \texttt{TestDiffusionMaps.py}:
\begin{lstlisting}[language=bash, caption=Running the unit test.]
$ python -m unittest TestDiffusionMaps
\end{lstlisting}
Therefore, one can expect the following outcome:
\begin{lstlisting}[language=bash, caption=Outcome of the unit test.]
..
--------------------------------------------------------------------
Ran 2 tests in 0.514s

OK
\end{lstlisting}

[coifman 2006]  R. R. Coifman and S. Lafon.  Diffusion maps.Applied and ComputationalHarmonic Analysis, 21(1):5 – 30, 2006.  Special Issue:  Diffusion Maps andWavelets.

