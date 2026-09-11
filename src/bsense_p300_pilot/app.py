"""Single-group acquisition UI; all robot control stays outside this application."""

import argparse
import json
import os
from pathlib import Path
import queue
import threading
import time
import sys
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pylsl import local_clock

from .protocol import Config, HARD_LIMIT, HIGHLIGHT, ICONS, LABELS, build_plan, identifier, timing_summary
from .recording import Recording, discover

BG, CARD, TEXT, MUTED, CYAN = "#101B2B", "#1C2D43", "#F1F5F9", "#ACC0D7", "#63DCE6"


def create_root():
    # Relocatable Python builds may retain build-machine Tcl search paths.
    # Configure only this process, using libraries belonging to this interpreter.
    for variable, folder, sentinel in (("TCL_LIBRARY", f"tcl{tk.TclVersion}", "init.tcl"),
                                      ("TK_LIBRARY", f"tk{tk.TkVersion}", "tk.tcl")):
        path = Path(sys.base_prefix)/"lib"/folder
        if variable not in os.environ and (path/sentinel).is_file():
            os.environ[variable] = str(path)
    return tk.Tk()


def mmss(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds//60:02d}:{seconds%60:02d}"


class App:
    def __init__(self, root, data_root: Path):
        self.root = root
        root.title("BSense P300 · 个体短时采集")
        root.geometry("1120x800")
        root.minsize(980, 740)
        root.configure(bg=BG)
        self.messages = queue.Queue()
        self.infos = []
        self.recording = None
        self.busy = False
        self.starting = False
        self.finishing = False
        self.save_failed = False
        self.closing = False
        self.cancel_start = False
        self.current = None
        self.health_warnings = ()
        self.after_step = self.after_off = None
        self.index = 0
        self.plan = []
        self.subject = tk.StringVar(value="P001")
        self.session = tk.StringVar(value=datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.data_root = tk.StringVar(value=str(data_root))
        self.groups = tk.StringVar(value="1")
        self.repeats = tk.StringVar(value="10")
        self.sequences = tk.StringVar(value="6")
        self.order = tk.StringVar(value="auto")
        self.notes = tk.StringVar()
        self.history = tk.StringVar()
        self.status = tk.StringVar(value="默认只采一组；同一人再次采集时更换 Session 编号。")
        self.task_text = tk.StringVar(value="准备开始")
        self.progress_text = tk.StringVar(value="设备检查 → 练习 → 一组正式采集 → 保存")
        self.clock_text = tk.StringVar(value="尚未开始")
        self.summary = tk.StringVar()
        self.inputs = []
        self._build()
        self._summary()
        root.bind("<Escape>", lambda _: self.abort())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self._poll)

    def _label(self, parent, text=None, variable=None, size=12, color=TEXT):
        return tk.Label(parent, text=text, textvariable=variable, bg=parent["bg"], fg=color,
                        font=("Arial", size), anchor="w")

    def _build(self):
        header = tk.Frame(self.root, bg=BG, padx=22, pady=16)
        self.header = header
        header.pack(fill="x")
        self._label(header, "P300 个体采集", size=23).pack(anchor="w")
        self._label(header, "一次一组 · 可重复测量 · 原始脑电与事件同步保存", color=MUTED).pack(anchor="w", pady=(5,0))
        body = tk.Frame(self.root, bg=BG, padx=22)
        body.pack(fill="both", expand=True)
        sidebar = tk.Frame(body, bg=CARD, width=345)
        self.sidebar = sidebar
        sidebar.pack(side="left", fill="y", padx=(0,18))
        sidebar.pack_propagate(False)
        canvas = tk.Canvas(sidebar, bg=CARD, highlightthickness=0, width=330)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        controls = tk.Frame(canvas, bg=CARD, padx=16, pady=12)
        window = canvas.create_window((0,0), window=controls, anchor="nw")
        controls.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        fields = [("匿名受试者编号", self.subject), ("本次 Session（重复采集需换号）", self.session)]
        for label, variable in fields:
            self._label(controls, label, size=11).pack(fill="x", pady=(5,2))
            entry = ttk.Entry(controls, textvariable=variable)
            entry.pack(fill="x")
            self.inputs.append(entry)
        line = tk.Frame(controls, bg=CARD)
        line.pack(fill="x", pady=(10,0))
        for label, var, values in [("组数", self.groups, (1,2,3)), ("每指令次数", self.repeats, (6,8,10,12,15,20)),
                                   ("闪烁轮数", self.sequences, (2,3,4,5,6,8,10))]:
            cell = tk.Frame(line, bg=CARD)
            cell.pack(side="left", expand=True, fill="x", padx=2)
            self._label(cell, label, size=10).pack()
            combo = ttk.Combobox(cell, textvariable=var, values=values, state="readonly", width=7)
            combo.pack()
            combo.bind("<<ComboboxSelected>>", lambda _: self._summary())
            self.inputs.append(combo)
        estimate = self._label(controls, variable=self.summary, size=11, color=CYAN)
        estimate.configure(wraplength=285, justify="left")
        estimate.pack(fill="x", pady=10)
        presets = tk.Frame(controls, bg=CARD)
        presets.pack(fill="x")
        for label, config in [("标准单组", Config()), ("现场短测", Config(1,6,4))]:
            button = ttk.Button(presets, text=label, command=lambda c=config: self.preset(c))
            button.pack(side="left", expand=True, fill="x", padx=2)
            self.inputs.append(button)
        self._label(controls, "EEG 流（先启动设备发布软件）", size=11).pack(fill="x", pady=(12,2))
        self.stream = ttk.Combobox(controls, state="readonly", width=25)
        self.stream.pack(fill="x")
        self.inputs.append(self.stream)
        self.scan_button = ttk.Button(controls, text="扫描 EEG", command=self.scan)
        self.scan_button.pack(fill="x", pady=4)
        self.inputs.append(self.scan_button)
        self._label(controls, "缺少通道标签时：核实后指定顺序", size=10).pack(fill="x", pady=(6,2))
        combo = ttk.Combobox(controls, textvariable=self.order, values=("auto", "FP1,FP2", "FP2,FP1"), state="readonly")
        combo.pack(fill="x")
        self.inputs.append(combo)
        self._label(controls, "佩戴/参考电极/单位/异常备注", size=10).pack(fill="x", pady=(6,2))
        note = ttk.Entry(controls, textvariable=self.notes)
        note.pack(fill="x")
        self.inputs.append(note)
        folder = ttk.Button(controls, text="选择数据保存目录", command=self.choose_root)
        folder.pack(fill="x", pady=(10,4))
        self.inputs.append(folder)
        self._label(controls, "同人历史 Session（逗号分隔，可留空）", size=10).pack(fill="x", pady=(6,2))
        history = ttk.Entry(controls, textvariable=self.history)
        history.pack(fill="x")
        self.inputs.append(history)
        self.train_button = ttk.Button(controls, text="训练当前受试者模型", command=self.train)
        self.train_button.pack(fill="x", pady=8)
        self.inputs.append(self.train_button)
        right = tk.Frame(body, bg=BG)
        self.task_panel = right
        right.pack(side="left", fill="both", expand=True)
        title = self._label(right, variable=self.task_text, size=21)
        title.configure(wraplength=680)
        title.pack(fill="x", pady=(4,10))
        self._label(right, variable=self.progress_text, size=11, color=MUTED).pack(fill="x")
        grid = tk.Frame(right, bg=BG)
        grid.pack(fill="both", expand=True, pady=18)
        self.tiles = []
        for i, label in enumerate(LABELS):
            tile = tk.Label(grid, text=f"{ICONS[i]}\n{label}", bg=CARD, fg=TEXT,
                            font=("Arial", 25), relief="flat", highlightthickness=3,
                            highlightbackground=CARD)
            tile.grid(row=i//3, column=i%3, sticky="nsew", padx=6, pady=6)
            self.tiles.append(tile)
        for i in range(3):
            grid.columnconfigure(i, weight=1, uniform="tiles")
        for i in range(2):
            grid.rowconfigure(i, weight=1, uniform="rows")
        self._label(right, variable=self.clock_text, size=14, color=CYAN).pack(fill="x")
        actions = tk.Frame(right, bg=BG)
        actions.pack(fill="x", pady=12)
        self.start_button = ttk.Button(actions, text="开始一组采集", command=self.start)
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0,8))
        ttk.Button(actions, text="结束并保存（Esc）", command=self.abort).pack(side="left", fill="x", expand=True)
        footer = self._label(self.root, variable=self.status, size=11, color=MUTED)
        footer.configure(wraplength=1060, justify="left", padx=22, pady=12)
        footer.pack(fill="x")

    def config(self):
        config = Config(int(self.groups.get()), int(self.repeats.get()), int(self.sequences.get()))
        config.validate()
        return config

    def _summary(self):
        try:
            config = self.config()
            self.summary.set(f"正式 {config.groups} 组，每组 {mmss(6*config.trials_per_target*config.trial_seconds)}\n"
                             f"含练习和休息约 {mmss(config.total_seconds)}；20 分钟自动结束")
            self.start_button.configure(text=f"开始 {config.groups} 组采集")
        except ValueError as error:
            self.summary.set(str(error))

    def preset(self, config):
        self.groups.set(str(config.groups))
        self.repeats.set(str(config.trials_per_target))
        self.sequences.set(str(config.sequences))
        self._summary()

    def choose_root(self):
        folder = filedialog.askdirectory(initialdir=self.data_root.get())
        if folder:
            self.data_root.set(folder)
            self.status.set(f"保存位置：{folder}")

    def scan(self):
        self.scan_button.configure(state="disabled")
        self.status.set("正在扫描 EEG 流……")
        self.background("scan", discover)

    def background(self, kind, callback):
        def work():
            try:
                self.messages.put((kind, callback(), None))
            except Exception as error:
                self.messages.put((kind, None, str(error)))
        threading.Thread(target=work, daemon=True).start()

    def _lock(self, locked):
        for widget in self.inputs:
            state = "disabled" if locked else ("readonly" if isinstance(widget, ttk.Combobox) else "normal")
            widget.configure(state=state)
        self.start_button.configure(state="disabled" if locked else "normal")

    def start(self):
        if self.busy:
            return
        try:
            config = self.config()
            subject, session = identifier(self.subject.get()), identifier(self.session.get())
            index = self.stream.current()
            if index < 0:
                raise ValueError("请先扫描并选择一个 EEG 流。")
            self.plan = build_plan(subject, session, config)
            root, info, order, notes = Path(self.data_root.get()), self.infos[index], self.order.get(), self.notes.get()
        except (ValueError, IndexError) as error:
            messagebox.showerror("无法开始", str(error))
            return
        self.busy = self.starting = True
        self.cancel_start = False
        self.started = time.monotonic()
        self._lock(True)
        self.status.set("连接 EEG 并创建本次数据目录……")
        self.background("start", lambda: Recording(root, subject, session, info, order, notes, config))

    def _clear_tiles(self):
        for tile in self.tiles:
            tile.configure(bg=CARD, fg=TEXT, highlightbackground=CARD)

    def _acquisition_display(self, active):
        self.root.attributes("-fullscreen", active)
        if active:
            self.header.pack_forget()
            self.sidebar.pack_forget()
        else:
            self.header.pack(before=self.sidebar.master, fill="x")
            self.sidebar.pack(before=self.task_panel, side="left", fill="y", padx=(0,18))

    def _next(self):
        self.after_step = None
        if not self.recording or self.finishing:
            return
        if time.monotonic()-self.started >= HARD_LIMIT:
            self.finish("time_limit", "已达到 20 分钟上限")
            return
        step = self.plan[self.index]
        if self.current and self.current.event == "device_qc":
            health = self.recording.health()
            if not health["ok"] or health["samples"] < 100:
                self.finish("quality_failed", health["reason"] if not health["ok"] else "设备检查结束时 EEG 样本不足 100 个")
                return
        self.current = step
        self.index += 1
        try:
            self._clear_tiles()
            self.task_text.set(step.message)
            if step.cue is not None:
                self.tiles[step.cue].configure(highlightbackground=CYAN)
            if step.flash is not None:
                self.tiles[step.flash].configure(bg=CYAN, fg=BG)
            self.root.update_idletasks()
            onset = local_clock()
            self.step_started = time.monotonic()
            self.recording.mark(step.event, step.metadata, onset)
            if step.event == "session_end":
                self.finish("complete", "协议完成")
                return
            if step.flash is not None:
                self.after_off = self.root.after(round(HIGHLIGHT*1000), self._off, step)
            if step.metadata.get("run") is not None:
                run = step.metadata["run"]
                trial = step.metadata.get("trial", "—")
                self.progress_text.set(f"{'练习' if run == 0 else f'第 {run} 组'} · 第 {trial} 次选择")
            self.after_step = self.root.after(max(0, round((self.step_started+step.seconds-time.monotonic())*1000)), self._next)
        except Exception as error:
            self.finish("recording_error", str(error))

    def _off(self, step):
        self.after_off = None
        if self.finishing or not self.recording:
            return
        try:
            self._clear_tiles()
            self.root.update_idletasks()
            self.recording.mark("p300_flash_off", step.metadata, local_clock())
        except Exception as error:
            self.finish("recording_error", str(error))

    def abort(self):
        if self.save_failed:
            self.save_failed = False
            self.finish(*self.pending_completion)
            return
        if self.starting:
            self.cancel_start = True
            self.status.set("已请求取消，正在完成连接清理……")
        elif self.recording:
            self.finish("aborted", "操作者结束")

    def finish(self, status, reason):
        if self.finishing:
            return
        self.finishing = True
        self.pending_completion = (status, reason)
        for token in (self.after_step, self.after_off):
            if token:
                self.root.after_cancel(token)
        self.after_step = self.after_off = None
        self._acquisition_display(False)
        self._clear_tiles()
        self.task_text.set("正在保存记录")
        recorder = self.recording
        try:
            recorder.mark("recording_end", {"status": status, "reason": reason})
        except Exception as error:
            status, reason = "recording_error", f"结束事件保存失败：{error}"
        self.status.set(f"{reason}；等待原始数据安全落盘……")
        def stop():
            recorder.stop(status, reason)
            return str(recorder.path), recorder.context["status"]
        self.background("stop", stop)

    def train(self):
        if self.busy:
            return
        from .training import train
        subject, session = self.subject.get(), self.session.get()
        root = Path(self.data_root.get()).expanduser()
        history = [s.strip() for s in self.history.get().split(",") if s.strip()]
        output = root/"models"/f"sub-{subject}"/f"{session}_{datetime.now():%Y%m%d_%H%M%S}"
        self.busy = True
        self._lock(True)
        self.status.set("正在训练该受试者模型并评估速度，测试段不会参与训练或参数选择……")
        self.background("train", lambda: (train(root, subject, session, output, history), str(output)))

    def _poll(self):
        try:
            while True:
                kind, result, error = self.messages.get_nowait()
                if kind == "scan":
                    self.scan_button.configure(state="normal")
                    self.infos = result or []
                    self.stream["values"] = [f"{i.name()} [{i.uid()[:8]}]" for i in self.infos]
                    if self.infos:
                        self.stream.current(0)
                    self.status.set(error or f"发现 {len(self.infos)} 个 EEG 流；请选择本次设备。")
                elif kind == "start":
                    self.starting = False
                    if error:
                        self.busy = False
                        self._lock(False)
                        self.status.set(f"连接失败：{error}")
                    else:
                        self.recording = result
                        if self.cancel_start or self.closing:
                            self.finish("aborted", "连接期间取消")
                        else:
                            self.index = 0
                            self.current = None
                            self.health_warnings = ()
                            self._acquisition_display(True)
                            self.data_started = time.monotonic()
                            self.status.set("已开始真实 EEG/XDF 录制；不适或需要结束时按 Esc。")
                            self._next()
                elif kind == "stop":
                    self._handle_stop(result, error)
                elif kind == "train":
                    self.busy = False
                    self._lock(False)
                    if error:
                        self.status.set(f"训练未完成：{error}")
                    else:
                        report, output = result
                        self.status.set(f"{report['status']} · 测试正确率 {report['test']['command_accuracy']:.1%} · 模型：{output}")
                        messagebox.showinfo("个体模型已生成", f"推荐闪烁轮数：{report['recommended_sequences'] or '暂无可靠建议'}\n"
                                            f"独立测试正确率：{report['test']['command_accuracy']:.1%}\n"
                                            f"拒绝率：{1-report['test']['coverage']:.1%}\n保存到：{output}\n尚未验证真机完赛。")
        except queue.Empty:
            pass
        if self.closing and not self.busy:
            self.root.destroy()
            return
        if self.recording and not self.finishing and not self.save_failed:
            elapsed = time.monotonic()-self.started
            health = self.recording.health()
            remaining = max(0, self.current.seconds-(time.monotonic()-self.step_started)) if self.current else 0
            self.clock_text.set(f"已用 {mmss(elapsed)} / 20:00 · 当前阶段剩余 {mmss(remaining)}")
            if elapsed >= HARD_LIMIT:
                self.finish("time_limit", "已达到 20 分钟上限")
            elif not health["ok"]:
                self.finish("quality_failed", health["reason"])
            else:
                warnings = tuple(health["warnings"])
                if warnings != self.health_warnings:
                    try:
                        self.recording.mark("acquisition_quality", {"warnings": list(warnings), "samples": health["samples"]})
                    except Exception as error:
                        self.finish("recording_error", str(error))
                    self.health_warnings = warnings
                if not self.finishing:
                    detail = "；".join(warnings) if warnings else "EEG 正在接收"
                    self.status.set(f"{detail} · 已录制 {health['samples']} 个样本。质量提示不阻断采集；持续 10 秒断流才中止。")
        self.root.after(100, self._poll)

    def _handle_stop(self, result, error):
        self.finishing = False
        if error and self.recording and self.recording.worker.is_alive():
            self.save_failed = self.busy = True
            self.closing = False
            self._lock(True)
            self.task_text.set("录制线程尚未结束")
            self.status.set(f"{error}；保留本次记录，稍后点击“结束并保存”重试。")
            return
        self.recording = None
        self.save_failed = self.busy = False
        self._lock(False)
        self.task_text.set("已结束" if not error else "保存发生错误")
        self.status.set(f"保存失败：{error}" if error else f"状态：{result[1]} · 已保存到 {result[0]}")

    def close(self):
        self.closing = True
        if self.busy:
            self.abort()
        else:
            self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home()/"BSenseDatasets"/"p300")
    parser.add_argument("--plan", action="store_true", help="打印默认时长，不连接设备")
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps(timing_summary(), ensure_ascii=False, indent=2))
        return
    root = create_root()
    App(root, args.data_root)
    root.mainloop()


if __name__ == "__main__":
    main()
