# """
# Evaluation pipeline for inverse PINN model.

# Produces InversePINNEvaluationReport with metrics required by the spec.
# """

# import numpy as np
# import torch
# from torch.utils.data import DataLoader

# from preference_dynamics.schemas import (
#     ICVector,
#     InversePINNEvaluationReport,
#     ObjectiveConfig,
#     ParameterVector,
#     TimeSeriesSample,
# )
# from preference_dynamics.training.loss import compute_inverse_pinn_loss
# from preference_dynamics.training.metrics import time_series_match_metrics


# def evaluate_inverse_pinn(  # noqa: PLR0912, PLR0915
#     model: torch.nn.Module,
#     dataloader: DataLoader,
#     objective_config: ObjectiveConfig,
#     n_actions: int,
#     device: str = "cpu",
# ) -> InversePINNEvaluationReport:
#     """
#     Evaluate inverse PINN model on a dataset split.

#     Computes metrics required by the spec:
#         - trajectory_error: Error between observed and reconstructed [u,a]
#         - physics_residual_score: Summary statistic of ODE residual
#         - parameter_error: Error in parameter estimates (if labels available)
#         - ic_error: Error in initial condition estimates (if labels available)
#         - time_series_match: Metrics comparing simulated trajectories

#     Args:
#         model: Trained inverse PINN model
#         dataloader: DataLoader yielding TimeSeriesSample objects
#         objective_config: Objective configuration for loss computation
#         n_actions: Number of actions (n)
#         device: Device to run evaluation on

#     Returns:
#         Evaluation report with aggregated metrics
#     """
#     model.eval()
#     model.to(device)

#     trajectory_errors = []
#     physics_residuals = []
#     parameter_errors = []
#     ic_errors = []
#     # time_series_match_metrics = []

#     has_labels = False

#     with torch.no_grad():
#         for batch in dataloader:
#             if isinstance(batch, dict):
#                 samples = batch.get("samples", [])
#             elif isinstance(batch, list):
#                 samples = batch
#             else:
#                 samples = [batch]

#             for sample in samples:
#                 if not isinstance(sample, TimeSeriesSample):
#                     continue

#                 observed_trajectory = torch.from_numpy(sample.time_series).float().to(device)
#                 observed_time_points = torch.from_numpy(sample.time_points).float().to(device)

#                 if observed_trajectory.ndim == 2:
#                     observed_trajectory = observed_trajectory.unsqueeze(0)
#                 if observed_time_points.ndim == 0:
#                     observed_time_points = observed_time_points.unsqueeze(0)

#                 outputs = model(
#                     observed_trajectory, observed_time_points, return_intermediates=True
#                 )
#                 predicted_observables = outputs["observables"]
#                 params_constrained = outputs["params_constrained"]
#                 ic = outputs["ic"]

#                 trajectory_error = torch.nn.functional.mse_loss(
#                     predicted_observables, observed_trajectory, reduction="mean"
#                 ).item()
#                 trajectory_errors.append(trajectory_error)

#                 physics_loss_dict = compute_inverse_pinn_loss(
#                     model=model,
#                     observed_trajectory=observed_trajectory,
#                     observed_time_points=observed_time_points,
#                     objective_config=objective_config,
#                     n_actions=n_actions,
#                 )
#                 physics_residual = physics_loss_dict["physics_loss"].mean().item()
#                 physics_residuals.append(physics_residual)

#                 if sample.parameters is not None and sample.initial_conditions is not None:
#                     has_labels = True
#                     target_params = torch.from_numpy(sample.parameters.values).float().to(device)
#                     target_ic = (
#                         torch.from_numpy(sample.initial_conditions.values).float().to(device)
#                     )

#                     if target_params.ndim == 1:
#                         target_params = target_params.unsqueeze(0)
#                     if target_ic.ndim == 1:
#                         target_ic = target_ic.unsqueeze(0)

#                     param_error = torch.nn.functional.mse_loss(
#                         params_constrained, target_params, reduction="mean"
#                     ).item()
#                     ic_error = torch.nn.functional.mse_loss(ic, target_ic, reduction="mean").item()

#                     parameter_errors.append(param_error)
#                     ic_errors.append(ic_error)

#                     params_vec = ParameterVector(
#                         values=params_constrained[0].cpu().numpy(), n_actions=n_actions
#                     )
#                     ic_vec = ICVector(values=ic[0].cpu().numpy(), n_actions=n_actions)

#                     consistency_metrics = time_series_match_metrics(
#                         inferred_params=params_vec,
#                         inferred_ic=ic_vec,
#                         observed_time_series=sample.time_series,
#                         observed_time_points=sample.time_points,
#                     )
#                     time_series_match_metrics.append(consistency_metrics)

#     trajectory_error_mean = float(np.mean(trajectory_errors)) if trajectory_errors else 0.0
#     physics_residual_score = float(np.mean(physics_residuals)) if physics_residuals else 0.0

#     parameter_error_mean = None
#     ic_error_mean = None
#     if has_labels and parameter_errors:
#         parameter_error_mean = float(np.mean(parameter_errors))
#         ic_error_mean = float(np.mean(ic_errors))

#     time_series_match = {}
#     if time_series_match_metrics:
#         consistency_keys = time_series_match_metrics[0].keys()
#         for key in consistency_keys:
#             values = [m[key] for m in time_series_match_metrics if isinstance(m[key], (int, float))]
#             if values:
#                 time_series_match[key] = float(np.mean(values))

#     return InversePINNEvaluationReport(
#         trajectory_error=trajectory_error_mean,
#         physics_residual_score=physics_residual_score,
#         parameter_error=parameter_error_mean,
#         ic_error=ic_error_mean,
#         time_series_match=time_series_match,
#     )
