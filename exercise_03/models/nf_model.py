import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import jammy_flows
from scipy.stats import norm


# Define the CNN encoder model. The output of the model is the input to the normalizing flow.
# The latent dimension is the number of parameters in the normalizing flow.
class TinyCNNEncoder(nn.Module):
    def __init__(self, input_dim, latent_dimension):
        super(TinyCNNEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, latent_dimension),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ConvNNModel(nn.Module):
    def __init__(self, input_dim, latent_dimension):
        super(ConvNNModel, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=20, kernel_size=64, stride=16, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=20, out_channels=10, kernel_size=16, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=2, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(latent_dimension),
        )

    def forward(self, x):
        x.unsqueeze_(1)
        return self.net(x)


class CombinedModel(nn.Module):
    """
    A combined model that integrates a normalizing flow with a CNN encoder.
    """

    def __init__(self, encoder, input_dim, nf_type="diagonal_gaussian"):
        """
        Initializes the normalizing flow model.

        Parameters
        ----------
        encoder : callable
            A function or callable object that returns an encoder model. The encoder model
            should take the number of flow parameters as input and output the latent dimension.
        nf_type : str, optional
            The type of normalizing flow to use. Options are "diagonal_gaussian", "full_gaussian",
            and "full_flow". Default is "diagonal_gaussian".
        Raises
        ------
        Exception
            If an unknown `nf_type` is provided.
        Notes
        -----
        This method sets up a 3-dimensional probability density function (PDF) over Euclidean space (e3)
        using the specified normalizing flow type. The flow structure and options are configured based on
        the provided `nf_type`. The PDF is created using the `jammy_flows` library, and the number of flow
        parameters is determined and printed. The encoder is initialized with the number of flow parameters.
        """

        super().__init__()
        
        # we define a 3-d PDF over Euclidean space (e3)
        # using recommended settings (https://github.com/thoglu/jammy_flows/issues/5 scroll down)
        opt_dict = {}
        opt_dict["t"] = {}
        if nf_type == "diagonal_gaussian":
            opt_dict["t"]["cov_type"] = "diagonal"
            flow_defs = "t"
        elif nf_type == "full_gaussian":
            opt_dict["t"]["cov_type"] = "full"
            flow_defs = "t"
        elif nf_type == "full_flow":
            opt_dict["t"]["cov_type"] = "full"
            flow_defs = "gggt"
        else:
            raise Exception("Unknown nf type ", nf_type)

        opt_dict["g"] = dict()
        opt_dict["g"]["fit_normalization"] = 1
        opt_dict["g"]["upper_bound_for_widths"] = 1.0
        opt_dict["g"]["lower_bound_for_widths"] = 0.01

        self.nf_type = nf_type

        # 3d PDF (e3) with ggggt flow structure. Four Gaussianation-flow (https://arxiv.org/abs/2003.01941) layers ("g") and an affine flow ("t")
        self.pdf = jammy_flows.pdf(
            "e3",
            flow_defs,
            options_overwrite=opt_dict,
            amortize_everything=True,
            amortization_mlp_use_custom_mode=True,
        )

        # get the number of flow parameters
        num_flow_parameters = self.pdf.total_number_amortizable_params

        print("The normalizing flow has ", num_flow_parameters, " parameters...")

        # latent dimension (output of the CNN encoder) is set to 128
        self.encoder = encoder(input_dim, num_flow_parameters)

    def log_pdf_evaluation(self, target_labels, input_data):
        """
        Evaluate the log probability density function (PDF) for the given target labels and input data.

        The normalizing flow parameters are predicted by the encoder network based on the input data.
        Then, the log PDF is evaluated at the position of the label.

        Parameters:
        -----------
        target_labels : torch.Tensor
            The target labels for which the log PDF is to be evaluated.
        input_data : torch.Tensor
            The input data to be encoded and used for evaluating the log PDF.
        Returns:
        --------
        log_pdf : torch.Tensor
            The evaluated log PDF for the given target labels and input data.
        """
        latent_intermediate = self.encoder(
            input_data
        )  # get the flow parameters from the CNN encoder
        fp64_on_cpu = False
        if self.nf_type == "full_flow" and next(self.parameters()).device.type == "mps":
            # MPS does not support double precision, therefore we need to run the flow on the CPU
            fp64_on_cpu = True
        # convert to double. Double precision is needed for the Gaussianization flow. This is for numerical stability.
        if fp64_on_cpu:  # MPS does not support double precision, therefore we need to run the flow on the CPU
            latent_intermediate = latent_intermediate.cpu().to(torch.float64)
            target_labels = target_labels.cpu().to(torch.float64)
        elif self.nf_type == "full_flow":
            latent_intermediate = latent_intermediate.to(torch.float64)
            target_labels = target_labels.to(torch.float64)

        # evaluate the log PDF at the target labels
        log_pdf, _, _ = self.pdf(
            target_labels,
            amortization_parameters=latent_intermediate,
        )
        return log_pdf

    def sample(self, flow_params, samplesize_per_batchitem=1000):
        """
        Sample new points from the PDF given input data.

        Parameters
        ----------
        flow_params : tensor
            Parameters for the normalizing flow, must be of shape (B, L) where B is the batch size and L is the latent dimension.
        samplesize_per_batchitem : int, optional
            Number of samples to draw per batch item. Defaults to 1000.

        Returns
        -------
        tensor
            A tensor of shape (B, S, D) where B is the batch dimension, S is the number of samples,
            and D is the dimension of the target space for the samples.
        """
        # for full flow we need to convert to double precision for the normalizing flow
        # for numerical stability
        fp64_on_cpu = False
        if self.nf_type == "full_flow" and next(self.parameters()).device.type == "mps":
            fp64_on_cpu = True
            # convert to double
        if fp64_on_cpu:  # MPS does not support double precision, therefore we need to run the flow on the CPU
            flow_params = flow_params.cpu().to(torch.float64)
        elif self.nf_type == "full_flow":
            flow_params = flow_params.to(torch.float64)

        batch_size = flow_params.shape[0]  # get the batch size
        # sample from the normalizing flow
        repeated_samples, _, _, _ = self.pdf.sample(
            amortization_parameters=flow_params.repeat_interleave(
                samplesize_per_batchitem, dim=0
            ),
            allow_gradients=False,
        )

        # reshape the samples to be grouped by batch item
        reshaped_samples = repeated_samples[:, None, :].view(
            batch_size, samplesize_per_batchitem, -1
        )

        return reshaped_samples
    

    def forward(self, input_data, samplesize_per_batchitem=1000):
        """
        Perform a forward pass through the model, predicting the mean and standard deviation of the samples.

        Normalizing flows do not directly predict the target labels. Instead, they predict the parameters of the flow that
        transforms the base distribution to the target distribution. Often, we still want to predict the target labels.
        Then, we can sample from the distribution and form the mean of the samples and their standard deviations.
        This is what this function does.

        Parameters
        ----------
        input_data : torch.Tensor
            The input data tensor.
        Returns
        -------
        torch.Tensor
            A tensor of size (B, D*2) where the first half (size D) are the means,
            the second half (another D) are the standard deviations.
        """
        flow_params = self.encoder(input_data)
        samples = self.sample(
            flow_params,
            samplesize_per_batchitem=samplesize_per_batchitem,
        )

        # form mean along dim 1 (samples)
        means = samples.mean(dim=1)
        # form std along dim 1 (samples)
        std_deviations = samples.std(dim=1)

        # return means and std deviations as a concatenated tensor along dim 1
        return torch.cat([means, std_deviations], dim=1)

    def visualize_pdf(
        self, input_data, filename, samplesize=1000, batch_index=0, truth=None, labelNames=None,
    ):
        """
        Visualizes the probability density function (PDF) of the given input data using a normalizing flow model.

        The function generates samples from the normalizing flow (using the sample() function)
        and plots the histogram of the samples together with a Gaussian approximation.

        Parameters
        ----------
        input_data : torch.Tensor
            The input data tensor from which to pick one batch item for visualization.
        filename : str
            The filename where the resulting plot will be saved.
        samplesize : int, optional
            The number of samples to generate for the PDF visualization (default is 10000).
        batch_index : int, optional
            The index of the batch item to visualize (default is 0).
        truth : torch.Tensor, optional
            The true values of the labels, used for comparison in the plot (default is None).

        Returns
        -------
        None
        """
        # pick out one input from batch
        input_bitem = input_data[batch_index : batch_index + 1]

        # get the flow parameters (by passing the input data through the CNN encoder network)
        flow_params = self.encoder(input_bitem)

        # sample from the normalizing flow (i.e. samples are drawn from the base distribution and transformed by the flow
        # using the change-of-variable formula)
        samples = self.sample(flow_params, samplesize_per_batchitem=samplesize)
        # the rest of the code is just plotting.

        # we only have 1 batch item
        samples = samples.squeeze(0)

        # plot three 1-dimensional distributions together with normal approximation,
        # so we calculate the mean and std of the samples
        mean = samples.mean(dim=0).cpu().numpy()
        std = samples.std(dim=0).cpu().numpy()
        samples = samples.cpu().numpy()

        fig, axs = plt.subplots(3, 1)
        for dim_ind, ax in enumerate(axs):
            # plot the histogram of the samples
            ax.hist(
                samples[:, dim_ind],
                color="k",
                density=True,
                bins=50,
                alpha=0.5,
                label="density based on samples",
            )

            # plot the Gaussian approximation
            min_sample = samples[:, dim_ind].min()
            max_sample = samples[:, dim_ind].max()
            xvals = np.linspace(min_sample, max_sample, 1000)
            yvals = norm.pdf(xvals, loc=mean[dim_ind], scale=std[dim_ind])
            ax.plot(
                xvals, yvals, color="green", label="Gaussian approximation"
            )
            if labelNames is not None:
                ax.set_xlabel(f"{labelNames[dim_ind]}")
            # plot the true value if it is given
            if truth is not None:
                true_value = truth[batch_index : batch_index + 1, dim_ind].cpu().item()
                ax.axvline(true_value, color="red", label="true value")
                
            # plot the legend only for the first panel
            if dim_ind == 0:
                ax.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
