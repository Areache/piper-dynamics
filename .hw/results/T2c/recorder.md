# T2c 采集程序核对（只用文字描述，不贴代码 / 不改代码）

来源均为 Piper 主机上私有采集仓库的工作树（有未提交改动），是本次现场实际文件内容。采集程序有两个并行写力矩的链路：30 Hz 主 HDF5（ROS 链路）和独立高频文件（raw-CAN 链路）。

## 4.1 力矩系数写死在哪

- **高频录制器** `collect_data/motor_feedback_hf_recorder.py`：顶部 `_EFFORT_COEFF` 字典按 CAN ID 写死旧系数 —— 0x251/252/253（j1–j3）= `1.18125`，0x254/255/256（j4–j6）= `0.95844`。`parse_high_spd_frame()` 用 `raw_current × 系数 / 1000` 得到 `effort`（N·m）。
- **主 HDF5 的 `motor_torque`**：不经过高频录制器。链路是 SDK `GetArmHighSpdInfoMsgs().motor_*.effort`（piper_sdk 内部已用旧系数 `current × 1.18125 / 0.95844` 算出）→ ROS `piper_start_leader_follower_node.py` 发布 `MotorFeedback.effort`（`/1000`）→ `ros_operator.py` 把左右 6 维拼成 12 维写入 `observation["motor_torque"]` → `collect_data.py` 落 HDF5。也就是说主文件的 `motor_torque` 用的也是**旧系数**，但乘法发生在 piper_sdk 内，不是采集程序里。
- 两条链路都只存「推算力矩」：一个用高频录制器自己的系数，一个用 SDK 的系数；两者对 Piper X 都错（正确应为 k×b×c，见 `scale.md`）。`motor_current` 字段存的是原始电流（A），不受系数影响，旧数据可直接用。

## 4.2 高速反馈帧 0x251–0x256 里除了电流还有什么

以 PyAgxArm `parser.py` 的 `decode_high_spd`（`arm_feedback_high_spd.py` 字节描述一致）为准，每帧 8 字节：

| 字节 | 字段 | 类型/单位 |
|---|---|---|
| 0–1 | `velocity`（转速） | 有符号 int16，×0.001 → rad/s |
| 2–3 | `current`（电流） | 有符号 int16，×0.001 → A |
| 4–7 | `position`（位置） | 有符号 int32，×0.001 → rad（PyAgxArm）；piper_sdk 注释写 `rad` 未列缩放，两者对 `pos` 的缩放不一致，需注意 |

高频录制器 `parse_high_spd_frame()` **只取了 byte 2–3 的 current**，`velocity` 和 `position` 解码后直接丢弃（docstring 写明 "decoded but dropped because only current/torque are recorded"）。所以现有高频文件只有 `current` 和派生 `effort`，没有速度/位置。

## 4.3 时间戳

两点均属实：

1. **高频文件 `timestamp` 是固定采样网格的写入时刻，不是 CAN 帧接收时刻。** `motor_feedback_hf_recorder.py` 的 `_sample_loop()` 用固定 `sample_period` 网格（`next_tick += sample_period`），每个网格点 `append(time.monotonic())`，写文件前减首样本时间。CAN 读取线程 `_read_interface()` 用 `bus.recv()` 收帧，但**没有保存** python-can 的 `msg.timestamp`（帧到达时间）。
2. **主 HDF5 没有逐帧时间戳。** `collect_data.py` 按 `--frame_rate`（默认 30 Hz）写 `motor_torque` / `motor_current`，数据集不带每样本时间戳。

另外，T2 已观测高速反馈实际约 100 Hz，而高频录制器 docstring 写「原生约 50 Hz」、默认 `--motor_feedback_hz` 也按 50 处理 —— 这个不一致仍需远程侧确认默认采样率是否该改。

## 4.4 低速反馈有没有温度、采集程序有没有记录

- **低速反馈有温度。** piper_sdk `GetArmLowSpdInfoMsgs().motor_*.vol / foc_temp / motor_temp / bus_current`（驱动电压 0.1 V、驱动温度 1 ℃、电机温度 1 ℃、总线电流 0.001 A）；PyAgxArm `get_driver_states()` 同样给出 `vol / foc_temp / motor_temp / bus_current`。本次实测：电压 23.0–24.0 V、驱动温度 41–53 ℃、电机温度 34–51 ℃，`bus_current` 恒 0。
- **采集程序没有记录。** 在 `collect_data/`、`ros_operator.py`、`piper_ws/src/piper/scripts/` 里 grep `motor_temp / foc_temp / temperature / GetArmLowSpd / vol / bus_current` 均无命中；ROS 驱动只发布高速反馈的 `effort`/`current`，不发布低速反馈。

## 5. 修改建议（本仓库不改，等用户批准后由维护采集程序的人改）

1. **同帧存电流/速度/位置，共用一个时间戳**：高频录制器把同一 CAN 帧的 `current`、`velocity`、`position` 一起保存，且用该帧的 `msg.timestamp`（python-can 接收时间）作为时间戳，这样电流和关节角按定义无时间差。原网格时间另存一列保留。
2. **每个样本额外存 CAN 帧接收时间**：`_read_interface()` 把 `message.timestamp` 记录进 `_latest`，采样时一并落盘；保留原网格 `timestamp` 列以便兼容旧格式。
3. **不再存推算力矩，只存原始电流**：删掉 `_EFFORT_COEFF` 和 `effort` 列（或保留列名但按 Piper X 的 k×b×c 重算）。旧系数对 Piper X 错误，正确系数见 `scale.md`（j1–j6 = 4.725 / 4.134375 / 4.725 / 1.232201376 / 0.7787701152 / 0.7787701152）。
4. **电机温度按 1–10 Hz 单独存一路**：低速反馈的 `motor_temp`/`foc_temp`（约几 Hz）单独一个文件或数据集。
5. **主 HDF5 每帧存一个时间戳**：`collect_data.py` 在写 `motor_torque`/`motor_current` 时加一个逐样本时间戳列（如 `time.monotonic()` 相对首帧）。
6. 顺带确认：`--motor_feedback_hz` 默认值（50）与实际约 100 Hz 不符，建议改成按实测值或自适应。
