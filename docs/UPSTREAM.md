# 来源与提取边界

- bsense-lsl: `a2b221c3836962d306d0d34170c4679a92e228bc`，M7 六指令、100 ms 高亮、175 ms SOA、事件字段；`xdf_writer.py` 原样提取。
- bsense-realtime-classifier: `0e908fa58f543a78964ed314a84b16c90b181840`，`model_runtime.py` 的 `preprocess_window`、`signal_quality`、`erp_features` 原样提取。
- 新项目独立安装；不依赖两个旧仓库的路径。没有复制模型、受试者数据或其他任务模块。
- 采集界面使用当前机器狗控制端的 2×3 布局（前进/后退/左转；右转/急停/待机），覆盖采集端新版十字布局，避免训练与控制布局不同。
- 上游软件测试不能替代本项目真实设备、显示时序和受试者验证。
