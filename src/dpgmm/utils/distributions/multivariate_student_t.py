import torch
from torch import Tensor
from torch.distributions import Chi2, Distribution, constraints
from torch.distributions.multivariate_normal import _batch_mv
from torch.distributions.utils import _standard_normal


class MultivariateStudentT(Distribution):
    """
    Creates a multivariate Student-t distribution parameterized by a mean vector,
    a lower Cholesky factor of the scale matrix, and degrees of freedom.
    """

    @property
    def arg_constraints(self):
        """Returns a dictionary of constraints for the distribution's arguments."""
        return {
            "loc": constraints.real_vector,
            "scale_tril": constraints.lower_cholesky,
        }

    @property
    def support(self):
        """Returns the support of the distribution (real vectors)."""
        return constraints.real_vector

    def __init__(
        self,
        loc: Tensor,
        scale_tril: Tensor,
        df: Tensor,
    ):
        """
        Initializes the Multivariate Student-t distribution.

        Args:
            loc (Tensor): The mean vector of the distribution. Must be at least 1D.
            scale_tril (Tensor): The lower triangular Cholesky factor of the scale matrix.
                Must be at least 2D.
            df (Tensor): The degrees of freedom for the distribution.

        Raises:
            ValueError: If `loc` is less than 1-dimensional, or if `scale_tril` is
                less than 2-dimensional.
        """
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")

        if scale_tril.dim() < 2:
            raise ValueError(
                "precision_matrix must be at least two-dimensional, "
                "with optional leading batch dimensions"
            )

        self.loc = loc
        self.scale_tril = scale_tril
        self.df = df

        self.dim = self.loc.shape[-1]

        batch_shape = torch.broadcast_shapes(scale_tril.shape[:-2], loc.shape[:-1])
        self.loc = loc.expand(batch_shape + (-1,))
        event_shape = self.loc.shape[-1:]

        super(MultivariateStudentT, self).__init__(
            batch_shape, event_shape, validate_args=None
        )

    def log_prob(self, value: Tensor):
        """
        Evaluates the log probability density of the given values.

        Args:
            value (Tensor): The values to evaluate.

        Returns:
            Tensor: A tensor of log probabilities matching the batch shape.
        """
        x_batch_shape = value.shape[:-1]
        value_reshaped = torch.reshape(value, (-1, self.dim))
        # log_probs.shape -> (*batch_shape, dim)
        log_probs = self.log_student_t_pdf(value_reshaped)
        return torch.reshape(log_probs, x_batch_shape + self.batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates reparameterized samples from the distribution.

        Uses the reparameterization trick by sampling from a standard normal,
        scaling by the Cholesky factor, and dividing by the square root of a
        normalized Chi-squared sample.

        Args:
            sample_shape (torch.Size, optional): The shape of the samples to generate.
                Defaults to an empty Size.

        Returns:
            Tensor: A tensor of generated samples.
        """
        extended_shape = self._extended_shape(sample_shape)
        eps = _standard_normal(
            extended_shape, dtype=self.loc.dtype, device=self.loc.device
        )
        chi_distr = Chi2(self.df)
        # zero_mean_mvn_samples.shape -> (sample_shape, batch_shape, dim)
        zero_mean_mvn_samples = _batch_mv(self.scale_tril, eps)
        # chi2_samples.shape -> (sample_shape, batch_shape)
        chi2_samples = chi_distr.sample(sample_shape=sample_shape)
        # dfs.shape -> (sample_shape, batch_shape)
        dfs = self.df.view((-1,) * len(sample_shape) + self.batch_shape)
        # multiplier.shape -> (sample_shape, batch_shape, 1)
        multiplier = torch.sqrt(dfs / chi2_samples).unsqueeze(-1)
        # loc_reshaped.shape -> (sample_shape, batch_shape, dim)
        loc_reshaped = self.loc.view(
            (-1,) * len(sample_shape) + self.batch_shape + (self.dim,)
        )
        return loc_reshaped + multiplier * zero_mean_mvn_samples

    def log_student_t_pdf(self, xs: Tensor) -> Tensor:
        """
        Computes the log probability density function for the Student-t distribution
        using heavily vectorized tensor operations.

        Args:
            xs (Tensor): A reshaped tensor of data points to evaluate.

        Returns:
            Tensor: The computed log probabilities.
        """
        # xs.shape -> ((batch_size,), dim)
        # means.shape -> ((batch_shapes,), dim)
        # scale_trils.shape -> ((batch_shapes,), dim, dim)
        # nus.shape -> (batch_shapes,)
        batch_size, dim = xs.shape
        params_batch_shape = self.loc.shape[:-1]
        xs_reshaped = torch.reshape(
            xs, (batch_size,) + (1,) * len(params_batch_shape) + (dim,)
        )
        # xs_normalized.shape -> (batch_size, (batch_shapes,), dim)
        xs_normalized = xs_reshaped - self.loc
        #  xs_transposed.shape -> ((batch_shapes,), dim, batch_size)
        xs_transposed = torch.permute(
            xs_normalized, tuple(range(1, len(params_batch_shape) + 1, 1)) + (-1, 0)
        )
        # vecs.shape -> ((batch_shapes,), dim, batch_size)
        vecs = torch.linalg.solve_triangular(
            self.scale_tril, xs_transposed, upper=False
        )
        # mah_dists.shape ((batch_shapes,), batch_size)
        mah_dists = torch.norm(vecs, dim=-2, p=2.0)
        # It's necessary to mitigate the square root from norm function
        mah_dists_sq = mah_dists * mah_dists
        # mah_dists.shape -> (batch_size, (batch_shapes,))
        mah_dists_rearranged = torch.permute(
            mah_dists_sq, (-1,) + tuple(range(len(params_batch_shape)))
        )

        # scale_trils_diagonal.shape -> ((batch_shapes,), dim)
        scale_trils_diagonal = torch.diagonal(self.scale_tril, dim1=-2, dim2=-1)
        # log_dets_sqrt.shape -> (batch_shapes,)
        log_dets_sqrt = torch.sum(torch.log(scale_trils_diagonal), dim=-1)
        # nus_expanded.shape -> (batch_size, (batch_shapes,))
        nus_expanded = torch.unsqueeze(self.df, dim=0)

        t = (nus_expanded + dim) / 2
        num = torch.lgamma(t)
        denom = torch.lgamma(nus_expanded / 2.0) + (dim / 2.0) * torch.log(
            nus_expanded * torch.pi
        )
        denom += log_dets_sqrt
        denom = denom + (t * torch.log1p(mah_dists_rearranged / nus_expanded))
        return num - denom

    @constraints.dependent_property
    def df_constraint(self):
        """Returns the dynamic constraint for degrees of freedom (must be > dim - 1)."""
        return constraints.greater_than(self.dim - 1.0)


if __name__ == "__main__":
    dist = MultivariateStudentT(
        df=torch.tensor([5.0, 10.0, 15.0]),
        loc=torch.zeros(3, 2),
        scale_tril=torch.eye(2).repeat(3, 1, 1),
    )

    samples = dist.sample((500,))
    print(samples.shape)
