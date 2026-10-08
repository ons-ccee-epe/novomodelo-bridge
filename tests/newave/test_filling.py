"""Unit tests for the stage-math and ζ pure helpers in ``novomodelo_bridge.newave.filling``."""

from __future__ import annotations

from datetime import date

from novomodelo_bridge.newave.filling import (
    expansion_machine_counts,
    filling_completion_date,
    filling_min_rate_m3s,
    filling_schedule,
    online_machines,
    stage_id,
    zeta,
)


def test_filling_completion_date_adds_duration_months() -> None:
    # JURUENA: Oct 2024 start + 1 month filling -> enters service Nov 2024.
    assert filling_completion_date(2024, 10, 1) == date(2024, 11, 1)


def test_filling_completion_date_wraps_across_year_boundary() -> None:
    # Nov 2024 + 3 months -> Feb 2025.
    assert filling_completion_date(2024, 11, 3) == date(2025, 2, 1)


def test_filling_completion_date_zero_duration_is_start_month() -> None:
    # A degenerate (no-filling) window completes in its own start month.
    assert filling_completion_date(2025, 6, 0) == date(2025, 6, 1)


def test_stage_id_study_start_is_zero() -> None:
    assert stage_id(2024, 9, 2024, 9) == 0


def test_stage_id_jan_2025_is_four() -> None:
    assert stage_id(2025, 1, 2024, 9) == 4


def test_stage_id_pre_start_is_negative() -> None:
    # No clamping in this helper: dates before the study start go negative.
    assert stage_id(2024, 8, 2024, 9) == -1


def test_zeta_october_31_days() -> None:
    # October has 31 days -> 744 h -> 744 * 3600 / 1e6 == 2.6784.
    assert abs(zeta(2024, 10) - 2.6784) < 1e-9


def test_zeta_leap_february() -> None:
    # 2024 is a leap year: February has 29 days -> 696 h -> 2.5056.
    assert abs(zeta(2024, 2) - 2.5056) < 1e-9


def test_zeta_non_leap_february() -> None:
    # 2023 is not a leap year: February has 28 days -> 672 h -> 2.4192.
    assert abs(zeta(2023, 2) - 2.4192) < 1e-9


def test_fill_rate_juruena_single_stage() -> None:
    # JURUENA single-stage window [1, 2) = Oct 2024: 2.93 / 2.6784 ≈ 1.09.
    result = filling_min_rate_m3s(2.93, 0.0, 1, 2, lambda t: (2024, 10))
    assert abs(result - 2.93 / 2.6784) < 1e-6


def test_fill_rate_empty_window_is_zero() -> None:
    # Empty window (start >= entry): "no Filling phase" signal, not ZeroDivision.
    assert filling_min_rate_m3s(2.93, 0.0, 2, 2, lambda t: (2024, 10)) == 0.0


def test_fill_rate_nonzero_volume_morto() -> None:
    # 50% dead volume halves remaining: (2.93 * 0.5) / 2.6784.
    result = filling_min_rate_m3s(2.93, 50.0, 1, 2, lambda t: (2024, 10))
    assert abs(result - (2.93 * 0.5) / 2.6784) < 1e-6


def test_filling_schedule_juruena() -> None:
    # JURUENA: Oct 2024 start, duracao 1, study start Sep 2024 -> (1, 2).
    assert filling_schedule(2024, 10, 1, 2024, 9) == (1, 2)


def test_filling_schedule_pre_start_clamp() -> None:
    # Start before study start (raw_start = -2): start clamps to 0,
    # entry = raw_start + duracao = -2 + 3 = 1 (pre-start clamp).
    assert filling_schedule(2024, 7, 3, 2024, 9) == (0, 1)


def test_filling_schedule_zero_duration_equal_pair() -> None:
    # duracao 0: equal pair, the empty-window "no Filling phase" signal.
    assert filling_schedule(2024, 10, 0, 2024, 9) == (1, 1)


def test_filling_schedule_multi_month_window() -> None:
    # Jan 2025 start (raw_start = 4), duracao 6 -> entry = 4 + 6 = 10.
    assert filling_schedule(2025, 1, 6, 2024, 9) == (4, 10)


def test_online_machines_none_before_unit_entry() -> None:
    # Both units enter stage 4; querying stage 3 -> no group online.
    assert online_machines([(1, 4), (1, 4)], 2, 3) == {}


def test_online_machines_full_count_at_entry() -> None:
    # At stage 4 both conjunto-1 units are online.
    assert online_machines([(1, 4), (1, 4)], 2, 4) == {1: 2}


def test_online_machines_clamps_to_entry_stage() -> None:
    # Unit nominally enters stage 1 but is clamped up to entry_stage_id=2,
    # so it is not online when querying stage 1.
    assert online_machines([(1, 1)], 2, 1) == {}


def test_online_machines_partial_groups() -> None:
    # Two conjuntos entering different stages; query stage 5 -> only group 1.
    assert online_machines([(1, 5), (2, 7)], 2, 5) == {1: 1}


class TestExpansionMachineCounts:
    """``expansion_machine_counts`` resolves the three machine sources of an
    expanding plant: the final registry, the study-start configuration, and the
    dated entries."""

    _REGISTRY = {1: 4, 2: 3}
    _UNITS = ((2, 48), (2, 48), (2, 48))

    def test_declared_start_configuration_holds_before_entry(self) -> None:
        # modif.dat states conjunto 2 is empty at the study start.
        start = {1: 4, 2: 0}
        assert expansion_machine_counts(self._REGISTRY, start, self._UNITS, 47) == {
            1: 4,
            2: 0,
        }

    def test_registry_is_reached_at_the_entry_stage(self) -> None:
        start = {1: 4, 2: 0}
        assert expansion_machine_counts(self._REGISTRY, start, self._UNITS, 48) == {
            1: 4,
            2: 3,
        }

    def test_undeclared_start_falls_back_to_registry_minus_entering(self) -> None:
        # No NUMMAQ, so the start configuration equals the final registry; the
        # floor keeps the entering machines from being counted twice.
        assert expansion_machine_counts(
            self._REGISTRY, dict(self._REGISTRY), self._UNITS, 47
        ) == {1: 4, 2: 0}
        assert expansion_machine_counts(
            self._REGISTRY, dict(self._REGISTRY), self._UNITS, 48
        ) == {1: 4, 2: 3}

    def test_never_exceeds_the_registry(self) -> None:
        # A deck whose entries would overshoot the registry is capped, not trusted.
        overshoot = ((2, 0), (2, 0), (2, 0), (2, 0), (2, 0))
        counts = expansion_machine_counts(self._REGISTRY, {1: 4, 2: 0}, overshoot, 0)
        assert counts[2] == 3

    def test_partial_entry_counts_only_what_has_entered(self) -> None:
        units = ((2, 10), (2, 20), (2, 30))
        start = {1: 4, 2: 0}
        assert expansion_machine_counts(self._REGISTRY, start, units, 20)[2] == 2
