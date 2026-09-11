import copy
import json
import joblib
import numpy as np
import pytest

from bsense_p300_pilot.protocol import Config
from bsense_p300_pilot.training import (choose_sequences, epoch_features, fit_personal, predict_trials,
                                       read_session, score_sequences, structure_trials, train)
from conftest import event_rows


def test_missing_duplicate_and_wrong_labels_are_not_silently_dropped():
    config = Config(1,6,4)
    rows = event_rows(config)
    flash = next(i for i,r in enumerate(rows) if r["event"] == "p300_flash" and r["role"] == "train")
    for mutation in ("missing", "duplicate", "wrong_label"):
        bad = copy.deepcopy(rows)
        if mutation == "missing":
            bad.pop(flash)
        elif mutation == "duplicate":
            bad.insert(flash, bad[flash])
        else:
            bad[flash]["is_target"] = not bad[flash]["is_target"]
        with pytest.raises(ValueError):
            structure_trials(bad, config)


def test_bad_timing_is_preserved_as_failed_trial():
    rows = event_rows()
    flash = next(r for r in rows if r["event"] == "p300_flash" and r["role"] == "test")
    flash["time"] += 0.06
    trials = structure_trials(rows, Config(1,6,4))
    assert len(trials) == 36
    assert sum(not t["timing_ok"] for t in trials) == 1


def test_epoch_rejects_gaps_and_flat_signal():
    times = np.arange(0, 3, 0.004)
    values = np.zeros((len(times), 2))
    assert epoch_features(times, values, 1, 250)[2] == "flat_channel"
    keep = (times < 1.1) | (times > 1.2)
    assert epoch_features(times[keep], values[keep], 1, 250)[2] == "eeg_gap"


def test_training_excludes_validation_test_and_practice(synthetic_session):
    trials, _, _ = read_session(synthetic_session, "P01", "S01")
    model1, n1 = fit_personal(trials)
    changed = copy.deepcopy(trials)
    for t in changed:
        if t["role"] != "train":
            t["features"] = [np.ones_like(x)*1000 if x is not None else None for x in t["features"]]
            for f in t["flashes"]:
                f["is_target"] = not f["is_target"]
    model2, n2 = fit_personal(changed)
    assert n1 == n2 == 12*4*6
    x = np.asarray([trials[0]["features"][0]])
    np.testing.assert_array_equal(model1.predict_proba(x), model2.predict_proba(x))


def test_failed_trials_count_in_denominator_and_sequence_choice():
    curves = [dict(sequences=2, accepted_correct_over_all=.5, accepted_accuracy=1),
              dict(sequences=3, accepted_correct_over_all=.85, accepted_accuracy=.95),
              dict(sequences=4, accepted_correct_over_all=.9, accepted_accuracy=.95)]
    assert choose_sequences(curves) == 3
    assert choose_sequences(curves[:1]) is None
    trial = dict(key=[1,1], run=1, target="forward", flashes=[dict(sequence=1, flash_command="forward")])
    report = score_sequences([(trial, np.array([np.nan]))], 2)
    assert report["trials"] == 1
    assert report["command_accuracy"] == report["coverage"] == 0


def test_xdf_to_personal_artifact_and_no_overwrite(synthetic_session):
    output = synthetic_session/"model-output"
    report = train(synthetic_session, "P01", "S01", output)
    artifact = joblib.load(output/"model.joblib")
    assert artifact["task"] == "m7_p300"
    assert artifact["artifact_schema_version"] == 2
    assert artifact["training_subjects"] == ["P01"]
    assert artifact["feature_names"][0] == "ch1_erp_t-0.200"
    assert len(artifact["feature_names"]) == 120
    assert report["test"]["trials"] == 12
    assert report["provenance"]["quality"]["ok"] == 36*4*6
    assert report["split"]["history_train_only"]
    with pytest.raises(FileExistsError):
        train(synthetic_session, "P01", "S01", output)
    context = synthetic_session/"sub-P01"/"ses-S01"/"context.json"
    data = json.loads(context.read_text())
    data["status"] = "aborted"
    context.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="完整完成"):
        read_session(synthetic_session, "P01", "S01")
