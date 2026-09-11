"""Personal LDA calibration with separate, complete trial blocks for validation and test."""

import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pyxdf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import BANDPASS, OFFSET, SFREQ, WINDOW, erp_features, preprocess_window, signal_quality
from .protocol import COMMANDS, CUE, SOA, TAIL, VERSION, Config, identifier


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _one_stream(streams, kind):
    matches = [s for s in streams if s["info"]["type"][0].lower() == kind]
    if len(matches) != 1:
        raise ValueError(f"需要恰好一个 {kind} 流，实际 {len(matches)}。")
    return matches[0]


def read_session(root: Path, subject: str, session: str):
    folder = root.expanduser().resolve()/f"sub-{identifier(subject)}"/f"ses-{identifier(session)}"
    context = json.loads((folder/"context.json").read_text(encoding="utf-8"))
    if (context.get("subject_id"), context.get("session_id")) != (subject, session):
        raise ValueError("目录编号与采集身份不一致。")
    if context.get("status") != "complete" or context.get("protocol_version") != VERSION:
        raise ValueError("只训练本项目完整完成的 Session；中止记录保留但不自动用于训练。")
    if context.get("clock_offset_count", 0) < 1 or context.get("timestamp_inversions", 0):
        raise ValueError("记录缺少时钟校正或存在时间戳倒序。")
    config = Config(**context["config"])
    config.validate()
    files = list(folder.glob("*task-m7_p300.xdf"))
    if len(files) != 1:
        raise ValueError("每个 Session 必须对应一个 XDF。")
    streams, _ = pyxdf.load_xdf(str(files[0]), synchronize_clocks=True, dejitter_timestamps=False)
    eeg, markers = _one_stream(streams, "eeg"), _one_stream(streams, "markers")
    times = np.asarray(eeg["time_stamps"], dtype=float)
    values = np.asarray(eeg["time_series"], dtype=float)
    order = context.get("channel_order")
    if order not in (["FP1", "FP2"], ["FP2", "FP1"]):
        raise ValueError("缺少经过确认的 FP1/FP2 通道顺序。")
    if values.ndim != 2 or values.shape[1] != 2 or len(times) < 2:
        raise ValueError("EEG 数据形状错误。")
    values = values[:, [order.index("FP1"), order.index("FP2")]]
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("EEG 时间戳必须有限且严格递增。")
    rows = []
    for sample, stamp in zip(markers["time_series"], markers["time_stamps"], strict=True):
        row = json.loads(sample[0])
        if row.get("subject_id") != subject or row.get("session_id") != session:
            raise ValueError("Marker 身份与 Session 不一致。")
        row["time"] = float(stamp)
        rows.append(row)
    ends = {r["run"] for r in rows if r["event"] == "p300_run_end"}
    if ends != set(range(config.groups+1)) or sum(r["event"] == "session_end" for r in rows) != 1:
        raise ValueError("缺少完整 Run/Session 结束事件。")
    trials = structure_trials(rows, config)
    sfreq = float(context["nominal_srate"])
    names = None
    audit = Counter()
    for trial in trials:
        trial["features"] = []
        trial["valid"] = []
        for flash in trial["flashes"]:
            if not trial["timing_ok"]:
                feature, reason = None, "stimulus_timing"
            else:
                feature, feature_names, reason = epoch_features(times, values, flash["time"], sfreq)
                if feature is not None:
                    names = feature_names
            trial["features"].append(feature)
            trial["valid"].append(reason == "ok")
            audit[reason] += 1
    if names is None:
        raise ValueError("没有有效 ERP 窗口；先检查采集信号与事件时序。")
    digest = hashlib.sha256()
    with files[0].open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    provenance = dict(xdf=files[0].name, sha256=digest.hexdigest(), subject_id=subject,
                      session_id=session, quality=dict(audit), channel_order=order, config=context["config"])
    return trials, names, provenance


def structure_trials(rows, config: Config):
    starts, flashes, offsets = {}, defaultdict(list), {}
    for row in rows:
        if row.get("role") == "practice" or not row.get("run"):
            continue
        key = (row["run"], row.get("global_trial"))
        if row["event"] == "p300_trial_start":
            if key in starts:
                raise ValueError("重复 Trial 起始事件。")
            starts[key] = row
        elif row["event"] == "p300_flash":
            flashes[key].append(row)
        elif row["event"] == "p300_flash_off":
            offkey = (*key, row["sequence"], row["flash_command"])
            if offkey in offsets:
                raise ValueError("重复闪烁结束事件。")
            offsets[offkey] = row["time"]
    if set(starts) != set(flashes):
        raise ValueError("Trial 起始事件与闪烁记录不匹配。")
    result = []
    for key, start in starts.items():
        fs = flashes[key]
        if start.get("role") not in dict(config.role_repetitions) or len(fs) != config.sequences*6:
            raise ValueError("Trial 角色错误或闪烁不完整。")
        if start.get("target_command") not in COMMANDS:
            raise ValueError("未知目标指令。")
        seen = set()
        for f in fs:
            pair = (f["sequence"], f["flash_command"])
            expected_label = f["flash_command"] == start["target_command"]
            if (pair in seen or f["sequence"] not in range(1, config.sequences+1)
                    or f["flash_command"] not in COMMANDS or f.get("role") != start["role"]
                    or f.get("target_command") != start["target_command"]
                    or type(f.get("is_target")) is not bool or f["is_target"] != expected_label):
                raise ValueError("闪烁标签、序列或角色不一致。")
            seen.add(pair)
        expected_order = [s for s in range(1, config.sequences+1) for _ in COMMANDS]
        if [f["sequence"] for f in fs] != expected_order:
            raise ValueError("闪烁序列顺序错误。")
        intervals = np.diff([f["time"] for f in fs])
        durations = [offsets.get((*key, f["sequence"], f["flash_command"]), -1)-f["time"] for f in fs]
        # These are project-internal software timing gates, not display-onset validation.
        timing_ok = bool(np.all((intervals >= 0.155) & (intervals <= 0.215))
                         and np.all((np.asarray(durations) >= 0.08) & (np.asarray(durations) <= 0.14)))
        result.append(dict(key=list(key), run=key[0], role=start["role"],
                           target=start["target_command"], flashes=fs, timing_ok=timing_ok))
    if {t["run"] for t in result} != set(range(1, config.groups+1)):
        raise ValueError("组编号不完整。")
    for run in range(1, config.groups+1):
        subset = [t for t in result if t["run"] == run]
        for role, repeats in config.role_repetitions:
            group = [t for t in subset if t["role"] == role]
            if Counter(t["target"] for t in group) != Counter({c:repeats for c in COMMANDS}):
                raise ValueError(f"第 {run} 组 {role} 不完整或目标不平衡。")
        expected = [role for role, repeats in config.role_repetitions for _ in range(repeats*6)]
        if [t["role"] for t in subset] != expected:
            raise ValueError("训练、验证、测试必须按完整连续 Trial 分段。")
    return result


def epoch_features(times, values, onset, source_rate):
    target = onset+OFFSET+np.arange(round(WINDOW*SFREQ))/SFREQ
    lo, hi = np.searchsorted(times, [target[0], target[-1]])
    lo = max(0, lo-1)
    hi = min(len(times), hi+1)
    if target[0] < times[0] or target[-1] > times[-1] or hi-lo < 2:
        return None, None, "incomplete_epoch"
    ts, xs = times[lo:hi], values[lo:hi]
    if np.max(np.diff(ts)) > max(3/source_rate, 0.02):
        return None, None, "eeg_gap"
    if not np.isfinite(xs).all():
        return None, None, "non_finite"
    raw = np.vstack([np.interp(target, ts, xs[:, c]) for c in range(2)])
    processed = preprocess_window(raw, SFREQ, BANDPASS)
    good, reason = signal_quality(processed)
    if not good:
        return None, None, reason
    features, names = erp_features(processed, SFREQ)
    return features, names, "ok"


def fit_personal(trials):
    xs, ys = [], []
    command_counts = Counter()
    for trial in trials:
        if trial["role"] != "train":
            continue
        if sum(trial["valid"]) >= len(trial["valid"])*0.8:
            command_counts[trial["target"]] += 1
        for f, x, ok in zip(trial["flashes"], trial["features"], trial["valid"], strict=True):
            if ok:
                xs.append(x)
                ys.append(int(f["is_target"]))
    if any(command_counts[c] < 2 for c in COMMANDS) or set(ys) != {0, 1}:
        raise ValueError("每个指令至少需要 2 个质量合格的训练 Trial，请重新试采。")
    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]))
    model.fit(np.asarray(xs), np.asarray(ys))
    return model, len(ys)


def predict_trials(model, trials):
    results = []
    for trial in trials:
        p = np.full(len(trial["features"]), np.nan)
        good = np.flatnonzero(trial["valid"])
        if len(good):
            p[good] = model.predict_proba(np.asarray([trial["features"][i] for i in good]))[:, list(model.classes_).index(1)]
        results.append((trial, p))
    return results


def score_sequences(predictions, sequences):
    rows = []
    for trial, ps in predictions:
        by_command = {c: [] for c in COMMANDS}
        for flash, p in zip(trial["flashes"], ps, strict=True):
            if flash["sequence"] <= sequences and np.isfinite(p):
                by_command[flash["flash_command"]].append(float(p))
        complete = all(len(v) == sequences for v in by_command.values())
        scores = {c: float(np.mean(v)) if v else 0.0 for c, v in by_command.items()}
        rank = sorted(COMMANDS, key=scores.get, reverse=True)
        confidence, margin = scores[rank[0]], scores[rank[0]]-scores[rank[1]]
        accepted = complete and confidence >= 0.55 and margin >= 0.08
        rows.append(dict(run=trial["run"], trial=trial["key"][1], target=trial["target"],
                         prediction=rank[0] if complete else None, complete=complete,
                         correct=complete and rank[0] == trial["target"], accepted=accepted,
                         accepted_correct=accepted and rank[0] == trial["target"],
                         confidence=confidence, margin=margin))
    n = len(rows)
    seconds = CUE+6*SOA*sequences+TAIL
    correct = sum(r["correct"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    ac = sum(r["accepted_correct"] for r in rows)
    matrix = [[0]*7 for _ in COMMANDS]
    for row in rows:
        column = COMMANDS.index(row["prediction"]) if row["accepted"] else 6
        matrix[COMMANDS.index(row["target"] )][column] += 1
    fraction = correct/n if n else 0
    z = 1.96
    center = (fraction+z*z/(2*n))/(1+z*z/n) if n else 0
    radius = z*np.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/(1+z*z/n) if n else 0
    return dict(sequences=sequences, trials=n, complete_trials=sum(r["complete"] for r in rows),
                command_accuracy=correct/n if n else 0, coverage=accepted/n if n else 0,
                command_accuracy_wilson95=[max(0, float(center-radius)), min(1, float(center+radius))],
                accepted_wrong_commands=accepted-ac, rejected_trials=n-accepted,
                confusion_matrix_with_reject=matrix, confusion_columns=[*COMMANDS, "reject"],
                accepted_accuracy=ac/accepted if accepted else None,
                accepted_correct_over_all=ac/n if n else 0,
                estimated_selection_seconds=round(seconds, 3),
                estimated_correct_selections_per_minute=60*ac/n/seconds if n else 0,
                rows=rows)


def choose_sequences(curves):
    # Select on validation only; never tune sequence count on the held-out test block.
    eligible = [c for c in curves if c["accepted_correct_over_all"] >= 0.8
                and (c["accepted_accuracy"] or 0) >= 0.9]
    return min(eligible, key=lambda c: c["sequences"])["sequences"] if eligible else None


def train(root: Path, subject: str, session: str, output: Path, history_sessions=()):
    if output.exists():
        raise FileExistsError("输出目录已存在，请选新目录以保留之前的模型和评估。")
    trials, names, provenance = read_session(root, subject, session)
    fitting_trials = list(trials)
    history = []
    if session in history_sessions or len(set(history_sessions)) != len(history_sessions):
        raise ValueError("历史 Session 不得重复或包含当前 Session。")
    for previous in history_sessions:
        old, old_names, old_source = read_session(root, subject, previous)
        if old_names != names:
            raise ValueError("历史数据特征不兼容。")
        fitting_trials.extend(t for t in old if t["role"] == "train")
        history.append(old_source)
    model, count = fit_personal(fitting_trials)
    validation = predict_trials(model, [t for t in trials if t["role"] == "validation"])
    maximum = provenance["config"]["sequences"]
    curves = [score_sequences(validation, n) for n in range(2, maximum+1)]
    recommended = choose_sequences(curves)
    # Freeze this model and selection policy before any test predictions.
    selected = recommended or maximum
    test = score_sequences(predict_trials(model, [t for t in trials if t["role"] == "test"]), selected)
    confirmed = recommended is not None and test["accepted_correct_over_all"] >= 0.8 and (test["accepted_accuracy"] or 0) >= 0.9
    versions = {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "pyxdf")}
    artifact = dict(artifact_schema_version=2, task="m7_p300", target_kind="classification",
                    model_family="sklearn", model_name="personal_shrinkage_lda", model=model,
                    feature_names=names, feature_mode="erp", sfreq=SFREQ, window_seconds=WINDOW,
                    window_offset_seconds=OFFSET, stride_seconds=SOA, bandpass_hz=list(BANDPASS),
                    channel_count=2, channel_order=["FP1", "FP2"], label_mapping={0:"non_target", 1:"target"},
                    deployment_mode="event_locked_marker_required", default_confidence_threshold=0.55,
                    unknown_state=-1, experimental_model=True, control_output_enabled=False,
                    training_subjects=[subject], training_session=session, training_history=list(history_sessions),
                    software_versions=versions,
                    training_role="train", evaluation="within_subject_held_out_trial_blocks", recommended_sequences=recommended,
                    p300_control_defaults=dict(sequences_per_trial=selected, minimum_flashes_per_command=selected,
                                               confidence_threshold=0.55, margin_threshold=0.08, minimum_quality_ratio=0.8))
    report = dict(status="offline_pilot_pass" if confirmed else "needs_more_validation", provenance=provenance, history=history,
                  software_versions=versions,
                  training_epochs=count, split=dict(unit="contiguous_complete_trials", validation_session=session,
                                                    test_session=session, history_train_only=True, practice_excluded=True),
                  selected_sequences=selected, recommended_sequences=recommended, validation_curves=curves,
                  test=test, chance_accuracy=1/6,
                  limitations=["同人最新 Session 的独立 Trial 分段评估，不是跨人验证；测试仅 12 次/组，统计不确定性很大。",
                               "速度是长序列前缀离线估计；实际短序列、自由选择及无意图场景仍需在线验证。",
                               "FP1/FP2 可能包含眼动和面肌伪迹；软件质检不能证明信号来自 P300。",
                               "现有机器狗控制台固定 10 轮闪烁，且不强制执行 control_output_enabled 字段；不要把此字段当作真机联锁。",
                               "只有候选模型，尚无机械狗速度或成功率的物理验证。"])
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(artifact, output/"model.joblib")
    write_json(output/"training_report.json", report)
    write_json(output/"speed_profile.json", dict(subject_id=subject, session_id=session,
                                                status=report["status"], recommended_sequences=recommended,
                                                tested_sequences=selected, requires_controller_integration=True))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-session", action="append", default=[], help="仅合并该同人历史 Session 的训练段，可重复指定。")
    args = parser.parse_args(argv)
    report = train(args.data_root, args.subject, args.session, args.output_dir, args.history_session)
    print(json.dumps({"status": report["status"], "recommended_sequences": report["recommended_sequences"],
                      "test_accuracy": report["test"]["command_accuracy"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
