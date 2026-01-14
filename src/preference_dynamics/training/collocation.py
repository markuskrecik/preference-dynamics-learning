# """
# Collocation strategy utilities for physics-informed neural networks.

# Provides utilities for selecting time points where physics residual is evaluated.
# """

# import numpy as np
# import torch


# def _observed_times_collocation_points(
#     time_points: np.ndarray | torch.Tensor,
# ) -> np.ndarray | torch.Tensor:
#     """
#     Return observed time points as collocation points.

#     Args:
#         time_points: Observed time points of shape (T,)

#     Returns:
#         Collocation time points (same as input), shape (T,)
#     """
#     return time_points


# def _uniform_collocation_points(
#     time_span: tuple[float, float],
#     n_points: int,
# ) -> np.ndarray:
#     """
#     Generate uniformly spaced collocation points.

#     Args:
#         time_span: Time span (t_start, t_end)
#         n_points: Number of collocation points

#     Returns:
#         Collocation time points of shape (n_points,)
#     """
#     return np.linspace(time_span[0], time_span[1], n_points)


# def _random_collocation_points(
#     time_span: tuple[float, float],
#     n_points: int,
#     seed: int | None = None,
# ) -> np.ndarray:
#     """
#     Generate random collocation points.

#     Args:
#         time_span: Time span (t_start, t_end)
#         n_points: Number of collocation points
#         seed: Random seed (optional)

#     Returns:
#         Collocation time points of shape (n_points,)
#     """
#     rng = np.random.default_rng(seed)
#     return rng.uniform(time_span[0], time_span[1], size=n_points)


# def get_collocation_points(
#     time_points: np.ndarray | torch.Tensor,
#     time_span: tuple[float, float],
#     strategy: str = "observed_times",
#     n_collocation_points: int | None = None,
#     seed: int | None = None,
# ) -> np.ndarray | torch.Tensor:
#     """
#     Get collocation points based on strategy.

#     Args:
#         time_points: Observed time points of shape (T,)
#         time_span: Time span (t_start, t_end)
#         strategy: Collocation strategy ("observed_times", "uniform", "random")
#         n_collocation_points: Number of additional collocation points (if strategy != "observed_times")
#         seed: Random seed for random strategy (optional)

#     Returns:
#         Collocation time points

#     Raises:
#         ValueError: If strategy is invalid or n_collocation_points is missing when required
#     """
#     if strategy == "observed_times":
#         return _observed_times_collocation_points(time_points)
#     elif strategy == "uniform":
#         if n_collocation_points is None:
#             raise ValueError("n_collocation_points must be provided for uniform strategy")
#         return _uniform_collocation_points(time_span, n_collocation_points)
#     elif strategy == "random":
#         if n_collocation_points is None:
#             raise ValueError("n_collocation_points must be provided for random strategy")
#         return _random_collocation_points(time_span, n_collocation_points, seed)
#     else:
#         raise ValueError(f"Unknown collocation strategy: {strategy}")
