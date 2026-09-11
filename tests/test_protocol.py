from collections import Counter
import pytest

from bsense_p300_pilot.protocol import Config, COMMANDS, build_plan, identifier, timing_summary


def test_single_group_budget_and_balanced_splits():
    plan = build_plan("P01", "S01")
    summary = timing_summary()
    assert summary["run_count"] == 1
    assert summary["protocol_seconds"] == 690.6
    assert summary["formal_trials"] == 60
    for role, repeats in Config().role_repetitions:
        targets = [s.metadata["target_command"] for s in plan if s.event == "p300_trial_start" and s.metadata["role"] == role]
        assert Counter(targets) == Counter({c: repeats for c in COMMANDS})


def test_short_and_multigroup_configs_enforce_time_limit():
    assert timing_summary(Config(1, 6, 4))["protocol_seconds"] == 384
    assert timing_summary(Config(2, 6, 6))["protocol_seconds"] < 1080
    with pytest.raises(ValueError, match="18 分钟"):
        Config(3, 10, 6).validate()
    with pytest.raises(ValueError):
        Config(1, 5, 6).validate()


def test_randomization_reproducible_no_target_cue_during_flashes():
    a = build_plan("P1", "S1")
    assert a == build_plan("P1", "S1")
    assert a != build_plan("P1", "S2")
    previous = None
    trial = None
    for step in a:
        if step.event != "p300_flash":
            continue
        assert step.cue is None
        assert not any(label in step.message for label in ("前进", "后退", "左转", "右转", "急停", "待机"))
        if trial == step.metadata["global_trial"]:
            assert step.flash != previous
        previous, trial = step.flash, step.metadata["global_trial"]


@pytest.mark.parametrize("value", ["../bad", "", "/absolute", "姓名", "a"*41])
def test_identifiers_reject_paths_and_real_names(value):
    with pytest.raises(ValueError):
        identifier(value)
