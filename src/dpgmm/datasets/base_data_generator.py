from abc import ABC, abstractmethod
from typing import Optional

from dpgmm.datasets.data_generator_types import SyntheticDataset
from dpgmm.visualisation import DataVisualizer


class BaseDataGenerator(ABC):
    """
    Abstract base class for synthetic data generators.

    Defines the interface that all specific data generation strategies (e.g., Gaussian,
    Uniform, Non-linear) must implement. It also provides a convenience static method
    for quickly visualizing the generated datasets.
    """

    @abstractmethod
    def generate(
        self, n_points: int = 500, data_dim: int = 2, num_components: int = 10
    ) -> SyntheticDataset:
        """
        Generates a synthetic dataset.

        This method must be implemented by subclasses to define the specific
        logic for creating data points, cluster centers, and assignments.

        Args:
            n_points (int, optional): The total number of data points to generate.
                Defaults to 500.
            data_dim (int, optional): The dimensionality of the data (number of features).
                Defaults to 2.
            num_components (Optional[int], optional): The number of clusters or components
                to generate. Defaults to 10.

        Returns:
            SyntheticDataset: A dictionary-like object containing the generated 'data',
            'centers', and 'assignment'.
        """
        pass

    @staticmethod
    def plot(
        gen_data: SyntheticDataset,
        title: str = "Generated data",
        out_file: Optional[str] = None,
    ) -> None:
        """
        A convenience wrapper to visualize the generated dataset.

        Delegates the plotting logic to the `DataVisualizer` class. This allows
        users to visualize the output of any generator instance directly from the class.

        Args:
            gen_data (SyntheticDataset): The dataset dictionary returned by `generate()`.
            title (str, optional): The title for the plot. Defaults to "Generated data".
            out_file (Optional[str], optional): Path to save the figure. If None,
                the plot is displayed interactively.
        """
        return DataVisualizer.plot(gen_data, title, out_file)
