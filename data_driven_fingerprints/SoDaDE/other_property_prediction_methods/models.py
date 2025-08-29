from gauche.kernels.fingerprint_kernels import TanimotoKernel
import gpytorch
import torch

from botorch import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

class Model(object):
    """
    Base class for all models.
    """

    def __init__(self, **kwargs):
        """
        Initialize the model with the given parameters.

        :param kwargs: Keyword arguments for model parameters.
        """
        self.params = kwargs

    def train(self, data):
        """
        Train the model on the given data.

        :param data: The training data.
        """
        raise NotImplementedError("Train method not implemented.")

    def predict(self, data):
        """
        Predict using the model on the given data.

        :param data: The data to predict on.
        :return: The predictions.
        """
        raise NotImplementedError("Predict method not implemented.")

class TreeModel(Model):
    """
    Decision tree model for predicting solvent properties.
    """

    def __init__(self, **kwargs):
        """
        Initialize the tree model with the given parameters.

        :param kwargs: Keyword arguments for model parameters.
        """
        super().__init__(**kwargs)
        self.tree = None  # Placeholder for the decision tree

class MultiTaskTaniamotoGP(Model):
    """
    Multi-task Gaussian Process model for predicting solvent properties.
    """

    def __init__(self, n_tasks, rank = None, **kwargs):
        """
        Initialize the multi-task Gaussian Process model with the given parameters.

        :param kwargs: Keyword arguments for model parameters.
        """
        self.rank = rank
        self.n_tasks = n_tasks

        self.gp = None  # Placeholder for the Gaussian Process
        # self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_tasks)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    def train(self, train_x, train_y):
        """
        Train the multi-task Gaussian Process model.

        :param train_x: Training input data.
        :param train_y: Training output data.
        :param likelihood: Likelihood function for the GP.
        """
        self.gp = MultitaskGPModel(n_tasks=self.n_tasks, train_x=train_x, train_y=train_y, likelihood=self.likelihood, rank=self.rank)

        self.gp.train()
        self.likelihood.train()

        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)
        

    def predict(self, x):
        if self.gp is None:
            raise ValueError("Model has not been trained yet.")
        
        self.gp.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Make predictions
            observed_pred = self.gp(x[0], x[1])
            return observed_pred.mean, observed_pred.variance

#### Helper models

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, n_tasks, train_x, train_y, likelihood, rank = None):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = TanimotoKernel()

        # We learn an IndexKernel for 4 tasks
        if rank is None:
            rank = n_tasks
        # The rank of the task covariance matrix
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=n_tasks, rank=rank)

    def forward(self, x, i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

    def transform_inputs(self, X, input_transform=None):
        return X
