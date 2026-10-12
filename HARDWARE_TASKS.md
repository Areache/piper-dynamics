# 真机任务板（远程 agent ⇄ 真机 agent）

这份文件是远程 agent 和真机 agent 之间唯一的任务通道。远程 agent 在这里派任务，真机 agent 执行后把结果写回这里，双方都通过 git 同步。

| 角色 | 在哪里 | 做什么 |
|---|---|---|
| 远程 agent | 用户 Windows 本机，能访问 mrbd / mrbd-lan（离线数据、URDF、v3 observer），**连不到 Piper 主机** | 写任务、审结果、改库代码 |
| 真机 agent | Piper 主机 `langyi@192.168.222.29`，接着 CAN 和机械臂 | 按任务执行，写结果，不自行扩大范围 |
| 操作员（用户） | 机械臂旁边 | 所有会让机械臂动的步骤都要由操作员在场确认；手握急停 |

分支：`hw-bringup`。每次开始前先 `git pull --rebase`。

---

## 1. 安全规则（每个任务都适用）

1. **没有操作员当场确认，不发任何运动指令。** 包括 `EnableArm`、`MotionCtrl_*`、`JointCtrl`、`GripperCtrl`、示教模式切换、`MasterSlaveConfig`。只读的 `Get*` 调用可以直接执行。
2. 任务标了 🟢 只读的，可以直接执行。标了 🔴 运动的，在本文件写出要执行的完整命令和位姿列表，等操作员回复"确认执行"后再执行。
3. 运动速度上限是 `MotionCtrl_2` 的速度百分比 ≤ 20。每个位姿之间都要停稳。
4. **不要直接运行 `piper-calibrate`（不带 `--samples` 的那种）。** 它会在 URDF 关节范围的 80% 内随机采位姿，在桌面环境下可能撞桌子或者自碰撞。位姿列表必须由远程 agent 和操作员审过，见 T4。
5. 不要改 CAN 接口的配置（波特率、名字），不要停掉别人正在跑的进程。如果发现 CAN 被 ROS 节点或录制程序占着，记下来并停止，不要 kill。
6. 不要把凭据、大文件（>5 MB）或原始数据集提交进 git。原始数据放在 `~/piper_hw_runs/<任务>_<YYYYMMDD>/`，在结果里写路径和 `sha256sum`。

## 2. 通信协议

- 任务状态写在标题后面：`[TODO]` → `[DOING]` → `[DONE]` / `[BLOCKED: 原因]`。
- 真机 agent **只改**两处：任务的状态标记，以及该任务下面的「结果」小节。要提问就写在「结果」里，以 `Q:` 开头。远程 agent 改任务描述，并写「审阅」小节。
- 小的结果文件（txt/md/json/png，单个 <5 MB）放进 `hw_results/<任务号>/`，和本文件一起提交。
- commit 前缀：真机侧用 `hw: T0 ...`，远程侧用 `remote: ...`。
- 如果 Piper 主机推不了 GitHub，就把 `hw_results/` 和本文件打包，由用户转交。在 T0 里报告能不能推。
- 遇到任务描述和实际情况不符（字段名不同、SDK 版本不同等），**按实际情况记录并停下**，不要自己猜一个替代方案继续往下做。

## 3. 远程侧已知的背景（供真机 agent 核对，不要当作事实）

这些结论来自 dish_organize 录制数据（mrbd-lan `/mnt/data2/yujie/dish_organize`，双臂 Piper），**不确定是不是这台机械臂录的**：

- 主文件 `episode_N.hdf5`：`observations/qpos`、`qvel`、`effort` 都是 14 维（左 j0–j5 + 左夹爪，右 j0–j5 + 右夹爪）。**`effort` 的机械臂 6 个关节全是 0**，只有夹爪那一列有值。原因是 ROS 驱动 `piper_start_leader_follower_node.py` 只填了 `effort[6] = grippers_effort / 1000`。
- `observations/motor_current`、`observations/motor_torque` 都是 12 维（左 6 + 右 6）。`motor_torque ≈ motor_current × 1.18`（j0–j2）、`× 0.96`（j3–j5），是线性换算出来的，不是独立测量。current 的单位**没有核实过**，目前按 A 处理。
- 另有 `episode_N_motor_hf.hdf5`：`current`、`effort` 各 12 维，再加 `timestamp`；文件属性是 `can_interfaces=can_left,can_right`、`sample_hz=50`、`layout=left_j0..j5, right_j0..j5`。**不知道是哪个程序写的**，sxl 的 piper-aio `collect_data.py` 不写这些字段。
- ROS 驱动里关节角的换算是 `joint_state.joint_i / 1000 * 0.017444`（0.001° → rad），速度是 `motor_speed / 1000`。
- 远程侧有两版 URDF：`piper_description.urdf` 和 `piper_description_v100.urdf`。需要确认这台臂对应哪一版。
- 本库 `src/piper_dynamics/io/can.py` 里的 `UNIT_SCALE` 和 `PiperSdkArm` 字段名**从没在真机上测过**；`move_j` 故意没实现。

---

## 4. 任务

### T0 环境盘点 🟢 只读 [TODO]

目的：搞清楚这台主机上有什么，再决定后面怎么做。

请执行下面这些检查，结果写到 `hw_results/T0/env.md`：

```bash
uname -a; lsb_release -a; whoami; hostname -I
python3 --version; which python3; conda env list 2>/dev/null
pip show piper_sdk 2>/dev/null || python3 -c "import piper_sdk, os; print(piper_sdk.__file__)"
ip -details link show type can          # 接口名、状态、bitrate
ls /opt/ros 2>/dev/null; echo $ROS_DISTRO
ps aux | grep -Ei "ros|piper|can|collect|record" | grep -v grep   # 有没有进程占着 CAN
find ~ /opt -iname "*piper*.urdf" -o -iname "*piper*.xacro" 2>/dev/null | head
git -C ~ ls-remote https://github.com/Areache/piper-dynamics.git HEAD   # 能不能访问 GitHub
```

另外报告：
- 有几条机械臂，各自接在哪个 CAN 接口上（`can0` / `can_left` / `can_right` ……）。用 `find_all_can_port.sh` 一类脚本或 `ethtool -i` 查 USB 映射。
- 每条臂的固件版本。piper_sdk 里如果有 `GetPiperFirmwareVersion()` 一类只读接口就调用，没有就写"无"。
- 找到的每个 URDF 的路径和 `sha256sum`。
- 这台机器能否 `git push` 到 GitHub。

#### 结果

（真机 agent 填写）

---

### T1 录制程序和 SDK 字段映射 🟢 只读 [TODO]

目的：确定 `motor_current`、`motor_torque`、`motor_hf` 这些字段到底来自哪个 SDK 调用、做了什么换算，从而把 `UNIT_SCALE` 定下来。

1. 在主机上找写这些字段的程序：
   ```bash
   grep -rIl --include=*.py -E "motor_hf|motor_current|motor_torque|can_left" ~ /opt 2>/dev/null
   ```
   对每个命中的文件，写出：数据集名 → SDK 调用 → 换算公式 → 采样频率 → 时间戳来源（`time.time()`、ROS time，还是 SDK 自带时间戳）。把关键代码片段（不超过 80 行）复制到 `hw_results/T1/recorder_snippets.md`。
2. 在 piper_sdk 源码里找 `GetArmHighSpdInfoMsgs` 返回的消息类，以及它的 CAN 解析代码。写出 `motor_speed`、`current`、`pos`、`effort` 每个字段的原始单位，以 SDK 注释或协议文档为准，并注明文件和行号。`GetArmJointMsgs`、`GetArmGripperMsgs` 也同样处理。
3. 对照 `src/piper_dynamics/io/can.py`，列出 `PiperSdkArm.read_state()` 里每个字段名在这个 SDK 版本里存在还是不存在，`UNIT_SCALE` 的每一项对还是错。**只报告，先不改代码。**

#### 结果

（真机 agent 填写）

---

### T2 被动遥测读取 🟢 只读（需要机械臂上电） [TODO]

目的：拿到原始数值，核对单位；顺便测出实际的消息更新频率。

前提：操作员确认机械臂已上电、CAN 已激活、机械臂静止，周围没有人在操作。**不调用任何 Enable/Motion/Ctrl 接口。** 如果 T0 发现 CAN 被别的进程占着，先停下来问。

对每条臂：
1. 写一个只读脚本 `hw_results/T2/read_raw.py`：连上 CAN 端口，以 200 Hz 轮询 10 s，记录**不做任何换算的原始整数**：
   - `GetArmJointMsgs()` 的 6 个关节角
   - `GetArmHighSpdInfoMsgs()` 每个电机的 `motor_speed`、`current`、`pos`、`effort`（字段名以 T1 查到的为准）
   - `GetArmGripperMsgs()` 的角度和 effort
   - `GetArmStatus()`：使能状态、控制模式、错误码
   - 每次读取的 `time.monotonic()`。SDK 消息如果自带时间戳或频率字段，也一并记下。
2. 原始数据存到 `~/piper_hw_runs/T2_<日期>/<can口>.npz`。
3. 在 `hw_results/T2/summary.md` 里写出：
   - 每个关节的原始 `current` 和 `effort` 的均值、标准差，以及 `effort / current` 的比值。和第 3 节的 1.18 / 0.96 对比。
   - 每种消息的实际更新频率：统计相邻两次读取之间数值真正发生变化的间隔，不要直接用轮询频率。
   - 臂当时的状态：使能/未使能、处于什么姿态（静止靠在限位上，还是悬空保持）。
   - 按 T1 推出的单位换算后，current 的量级是否合理（静止悬空时大致在 0.1–3 A 之间）。

#### 结果

（真机 agent 填写）

---

### ⛔ 检查点：T0–T2 完成后停下

把结果 push 上来（或交给用户转交）后**停止**，等远程 agent 审阅。远程 agent 会根据结果：
- 修正 `io/can.py` 的字段名和 `UNIT_SCALE`
- 给出 `move_j` 的实现和速度设置
- 写出 T3–T5 的具体命令和位姿列表

下面几项只是预告，**现在不要执行**。

### T3 两姿态单位/符号检查 + 空载噪声底 🔴 运动 [等待审阅]

对应 `docs/validation.md` 第 0 节。空夹爪，两个差别很大的姿态（水平伸展、竖直收起），每个姿态静止采 2 s，比较 `tau_meas` 和 `tau_model` 的量级和符号；记录空载时的 `|tau_ext|`，作为噪声底。

### T4 静态标定（审过的位姿列表） 🔴 运动 [等待审阅]

每条臂 30–50 个位姿，位姿列表由远程 agent 生成：优先取录制数据里实际到达过的姿态，再由操作员按桌面环境删掉危险的。用 `collect_static` 采集，再用 `piper-calibrate --samples` 离线辨识，输出带 provenance 的 `params/<arm_id>.yaml`。

### T5 已知重量验证 🔴 运动 [等待审阅]

对应 `docs/validation.md` 第 1 节。0 / 100 / 250 / 500 g（先称重记下实际质量），每个重量走约 10 个位姿，每个位姿静止约 2 s。
