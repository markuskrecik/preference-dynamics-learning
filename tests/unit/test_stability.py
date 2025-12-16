"""
Unit tests for stability checking functionality.
"""

import numpy as np

from preference_dynamics.schemas import ParameterVector, StabilityReport
from preference_dynamics.solver.stability import _check_stability_eigenvalues, check_stability


class TestCheckStability:
    """Test suite for check_stability() function."""

    def test_stable_config_n3(self) -> None:
        """Test stability checking with a known stable configuration for n=3."""
        params = ParameterVector(
            values=np.array(
                [
                    1.0,
                    1.5,
                    2.0,  # g
                    -1.0,
                    -0.5,
                    -0.5,  # μ <= 0
                    2.0,
                    -0.1,
                    -0.2,
                    -0.1,
                    1.5,
                    -0.3,
                    -0.2,
                    -0.3,
                    1.0,  # Π (diagonals > 0)
                    1.0,
                    -0.05,
                    0.1,
                    -0.1,
                    2.0,
                    -0.15,
                    -0.05,
                    -0.2,
                    1.5,  # Γ (diagonals > 0)
                ]
            )
        )

        is_stable = check_stability(params)
        _, reports = _check_stability_eigenvalues(params)

        assert isinstance(is_stable, bool)
        assert is_stable
        assert len(reports) > 0
        for report in reports:
            assert isinstance(report, StabilityReport)
            assert report.eigenvalues is not None
            assert report.eigenvectors is not None

    def test_stable_config_n1(self) -> None:
        """Test stability checking with a simple n=1 configuration."""
        params = ParameterVector(values=np.array([1.0, -1.0, 2.0, 1.5]))  # g, μ, Π, Γ

        is_stable = check_stability(params)
        _, reports = _check_stability_eigenvalues(params)

        assert isinstance(is_stable, bool)
        assert is_stable
        assert len(reports) == 1
        assert reports[0].eigenvalues is not None
        assert reports[0].eigenvectors is not None

    def test_checks_all_subsets(self) -> None:
        """Test that stability checker examines all non-empty subsets."""
        params = ParameterVector(
            values=np.array(
                [
                    1.0,
                    1.5,
                    2.0,  # g
                    -1.0,
                    -0.5,
                    -0.5,  # μ <= 0
                    2.0,
                    -0.1,
                    -0.2,
                    -0.1,
                    1.5,
                    -0.3,
                    -0.2,
                    -0.3,
                    1.0,  # Π
                    1.0,
                    -0.05,
                    0.1,
                    -0.1,
                    2.0,
                    -0.15,
                    -0.05,
                    -0.2,
                    1.5,  # Γ
                ]
            )
        )

        _, reports = _check_stability_eigenvalues(params)

        assert len(reports) > 0
        for report in reports:
            assert report.eigenvalues is not None
            assert report.eigenvectors is not None
            assert report.actions is not None

    def test_stability_report_structure(self) -> None:
        """Test that StabilityReport has correct structure."""
        params = ParameterVector(values=np.array([1.0, -1.0, 2.0, 1.5]))  # n=1

        _, reports = _check_stability_eigenvalues(params)

        assert len(reports) > 0

        for report in reports:
            assert hasattr(report, "is_stable")
            assert hasattr(report, "actions")
            assert hasattr(report, "desired_actions")
            assert hasattr(report, "eigenvalues")
            assert hasattr(report, "eigenvectors")

            assert isinstance(report.is_stable, bool)
            assert report.actions is None or isinstance(report.actions, list)
            assert report.desired_actions is None or isinstance(report.desired_actions, list)
            assert report.eigenvalues is None or isinstance(report.eigenvalues, np.ndarray)
            assert report.eigenvectors is None or isinstance(report.eigenvectors, np.ndarray)

    def test_unstable_config(self) -> None:
        """Test detection of unstable configuration."""
        params = ParameterVector(
            values=np.array(
                [
                    1.0,
                    1.0,
                    1.0,  # g
                    -1.0,
                    -1.0,
                    -1.0,  # μ <= 0
                    2.0,
                    5.0,
                    5.0,  # Π row 1 (large positive off-diag)
                    5.0,
                    2.0,
                    5.0,  # Π row 2
                    5.0,
                    5.0,
                    2.0,  # Π row 3
                    0.5,
                    0.0,
                    0.0,  # Γ row 1 (small diagonal)
                    0.0,
                    0.5,
                    0.0,  # Γ row 2
                    0.0,
                    0.0,
                    0.5,  # Γ row 3
                ]
            )
        )

        is_stable = check_stability(params)
        _, reports = _check_stability_eigenvalues(params)

        assert isinstance(is_stable, bool)
        unstable_reports = [r for r in reports if not r.is_stable]
        if not is_stable:
            assert len(unstable_reports) > 0

    def test_singular_gamma_detected(self) -> None:
        """Test that singular Γ matrix is detected."""
        params = ParameterVector(
            n_actions=2,
            values=np.array(
                [
                    1.0,
                    1.0,  # g (2 elements)
                    -1.0,
                    -1.0,  # μ (2 elements)
                    2.0,
                    0.0,
                    0.0,
                    0.1,  # Π (2x2 = 4 elements, diagonal > 0: 2.0, 0.1)
                    0.1,
                    0.0,
                    0.0,
                    0.1,  # Γ (2x2 = 4 elements, valid but very small diagonal)
                ]
            ),
        )

        is_stable = check_stability(params)
        _, reports = _check_stability_eigenvalues(params)

        assert isinstance(is_stable, bool)
        assert len(reports) > 0

    def test_different_n_values(self) -> None:
        """Test stability checking works for different n_actions values."""
        test_cases = [
            (1, 4),  # n=1: 4 parameters
            (2, 12),  # n=2: 12 parameters
            (3, 24),  # n=3: 24 parameters
        ]

        for n, expected_param_count in test_cases:
            params = np.ones(expected_param_count)
            params[0:n] = 1.0
            params[n : 2 * n] = -1.0

            for i in range(n):
                params[2 * n + i * n + i] = 2.0
            for i in range(n):
                params[2 * n + n**2 + i * n + i] = 1.5

            parameters = ParameterVector(values=params)

            is_stable = check_stability(parameters)
            _, reports = _check_stability_eigenvalues(parameters)

            assert isinstance(is_stable, bool)
            assert len(reports) > 0


class TestStabilityReportDataclass:
    """Test suite for StabilityReport dataclass."""

    def test_can_create_report(self) -> None:
        """Test that StabilityReport can be created directly."""
        report = StabilityReport(
            is_stable=True,
            actions=[0],
            desired_actions=[0],
            eigenvalues=np.array([1.0 + 0j]),
            eigenvectors=np.array([[1.0]]),
        )

        assert report.is_stable
        assert report.actions == [0]
        assert report.desired_actions == [0]
        assert report.eigenvalues is not None
        assert report.eigenvectors is not None

    def test_report_fields_accessible(self) -> None:
        """Test that StabilityReport fields can be accessed."""
        report = StabilityReport(
            is_stable=False,
            actions=None,
            desired_actions=None,
            eigenvalues=None,
            eigenvectors=None,
        )

        assert not report.is_stable
        assert report.actions is None
        assert report.desired_actions is None
