# Windows 设备试测

本项目是 Python 源码版，有 Windows 安装和启动脚本，不是免安装 EXE。先安装 Python 3.12 x64，保留 Python Launcher 和 Tcl/Tk 安装组件。

## 安装和启动

1. 在 GitHub 项目页选择 Code → Download ZIP，完整解压到本机目录，例如 `D:\BCI\bsense-p300-pilot`。不要在压缩包里直接双击脚本。
2. 双击 `windows\setup.bat`，等待出现 `[OK] Setup complete`。首次安装需要联网下载 Python 依赖。
3. 双击 `windows\run.bat`，打开采集界面。

如果已安装的是 Python 3.13，在项目目录的 CMD 窗口运行 `windows\setup.bat 3.13`。不要复制别的电脑或 macOS 的 `.venv`。

## 第一次连设备

1. 连接 BSense 设备，启动设备厂商软件并启用 EEG 的 LSL 发布。仅连接 USB/蓝牙不等于已发布 LSL。
2. 在本项目点击“扫描 EEG”，明确选择本次设备。程序要求双通道 FP1/FP2；缺少标签时，请先核实实际顺序，再手动指定。
3. 使用匿名编号如 `P001`，Session 每次换号。首次先录 30–60 秒，然后点击“结束并保存”。
4. 确认数据目录中有 XDF、`events.jsonl` 和 `context.json`。主动中止的 `aborted` 状态正常，这份短记录不用于训练。
5. 更换 Session，选择“现场短测”，完整完成约 6 分 24 秒流程。结束应显示 `complete`，再尝试“训练当前受试者模型”。

默认数据目录是 `%USERPROFILE%\BSenseDatasets\p300`，也可在界面选择保存目录。这个采集程序不连接机器狗、不发送运动指令。

## 请反馈什么

- Windows 和 Python 版本；设备软件名称/版本。
- 能否安装、打开界面、扫描到 EEG、通过设备检查。
- 结束状态，XDF 是否生成、文件大小，训练是否完成。
- 如果失败，提供终端错误文字/截图和去掉身份信息的 `context.json`。不要把原始受试者脑电上传到公开 GitHub Issue。

`needs_more_validation` 表示模型还没有达到内部验证条件，不等于程序安装失败。GitHub 软件测试通过也不能代替这次真实设备试测。
