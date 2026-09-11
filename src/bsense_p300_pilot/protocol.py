"""Balanced repeated M7 runs with an explicit participant time budget."""

from dataclasses import dataclass, field
import hashlib
import random
import re

COMMANDS = ("forward", "backward", "left", "right", "stop", "idle")
LABELS = ("前进", "后退", "左转", "右转", "急停", "待机")
ICONS = ("↑", "↓", "←", "→", "■", "●")
VERSION = "m7_personal_short_v1"
HIGHLIGHT = 0.100
SOA = 0.175
SEQUENCES = 6
CUE = 1.5
TAIL = 1.3
REST = 45.0
HARD_LIMIT = 1200.0

@dataclass(frozen=True)
class Config:
    groups: int = 1
    trials_per_target: int = 10
    sequences: int = 6

    def validate(self):
        if not 1 <= self.groups <= 3:
            raise ValueError("组数需为 1–3。")
        if not 6 <= self.trials_per_target <= 20:
            raise ValueError("每指令重复次数需为 6–20，保留独立验证和测试。")
        if not 2 <= self.sequences <= 10:
            raise ValueError("每次选择的闪烁轮数需为 2–10。")
        if self.total_seconds > 18*60:
            raise ValueError("当前配置超过 18 分钟，请减少组数、重复次数或闪烁轮数；至少预留 2 分钟余量。")

    @property
    def trial_seconds(self):
        return CUE+6*SOA*self.sequences+TAIL

    @property
    def total_seconds(self):
        return 70+6*self.trial_seconds+20+self.groups*6*self.trials_per_target*self.trial_seconds+(self.groups-1)*REST

    @property
    def role_repetitions(self):
        return (("train", self.trials_per_target-4), ("validation", 2), ("test", 2))


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", value):
        raise ValueError("编号限 1–40 位字母、数字、下划线或连字符，不填写姓名。")
    return value


def seed_for(subject: str, session: str) -> int:
    token = f"{VERSION}:{identifier(subject)}:{identifier(session)}"
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Step:
    event: str
    seconds: float
    message: str
    metadata: dict = field(default_factory=dict)
    cue: int | None = None
    flash: int | None = None


def shuffled_sequence(rng: random.Random, previous: int | None) -> list[int]:
    order = list(range(6))
    rng.shuffle(order)
    if order[0] == previous:
        order[0], order[1] = order[1], order[0]
    return order


def build_plan(subject: str, session: str, config: Config | None = None) -> list[Step]:
    config = config or Config()
    config.validate()
    rng = random.Random(seed_for(subject, session))
    common = dict(task="m7_p300", protocol_version=VERSION, subject_id=subject,
                  session_id=session, seed=seed_for(subject, session),
                  sequences_per_trial=config.sequences, soa_s=SOA,
                  highlight_duration_s=HIGHLIGHT, layout="controller_grid_2x3")
    plan = [Step("device_qc", 30, "设备检查：放松，减少头部运动"),
            Step("baseline_open", 30, "睁眼静息：看屏幕中央，正常呼吸"),
            Step("instruction", 10, "先看指定目标；闪烁时默数目标亮起次数，不追随其他格子")]
    global_trial = 0
    for run in range(config.groups+1):
        meta = {**common, "run": run, "block_number": run}
        plan.append(Step("p300_run_start", 0, "练习" if run == 0 else f"第 {run}/{config.groups} 组", meta))
        schedule = []
        for role, repeats in (("practice", 1),) if run == 0 else config.role_repetitions:
            targets = list(range(6))*repeats
            rng.shuffle(targets)
            schedule.extend((role, target) for target in targets)
        for trial, (role, target) in enumerate(schedule, 1):
            global_trial += 1
            tm = {**meta, "role": role, "trial": trial, "global_trial": global_trial,
                  "target_position": target, "target_command": COMMANDS[target]}
            plan.append(Step("p300_trial_start", CUE, f"请注视：{LABELS[target]}", tm, cue=target))
            previous = None
            for sequence in range(1, config.sequences + 1):
                order = shuffled_sequence(rng, previous)
                previous = order[-1]
                for position, flash in enumerate(order, 1):
                    fm = {**tm, "sequence": sequence, "position_in_sequence": position,
                          "flash_position": flash, "flash_command": COMMANDS[flash],
                          "is_target": flash == target}
                    plan.append(Step("p300_flash", SOA, "保持注视，默数目标闪烁", fm, flash=flash))
            # Preserve the last ERP response before presenting the next target cue.
            plan.append(Step("p300_trial_end", TAIL, "保持注视，默数目标闪烁", tm))
        plan.append(Step("p300_run_end", 0, "本轮已完成", meta))
        if run < config.groups:
            duration = 20.0 if run == 0 else REST
            plan.append(Step("rest", duration, "休息：放松视线，倒计时结束后继续", meta))
    plan.append(Step("session_end", 0, "采集完成，可以摘下设备", common))
    return plan


def timing_summary(config: Config | None = None) -> dict:
    config = config or Config()
    plan = build_plan("P001", "S001", config)
    seconds = sum(step.seconds for step in plan)
    return dict(protocol_version=VERSION, run_count=config.groups, trials_per_run=6*config.trials_per_target,
                sequences_per_trial=config.sequences, run_seconds=round(6*config.trials_per_target*config.trial_seconds, 2),
                practice_seconds=round(6*config.trial_seconds, 2),
                protocol_seconds=round(seconds, 2), hard_limit_seconds=HARD_LIMIT,
                preparation_budget_seconds=round(HARD_LIMIT-seconds, 2),
                formal_trials=config.groups*6*config.trials_per_target,
                formal_flashes=config.groups*6*config.trials_per_target*config.sequences*6)
