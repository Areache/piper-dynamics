# 真机任务板（远程 agent ⇄ 真机 agent）

远程 agent 在这里派任务，真机 agent 执行后把结果写回这里。双方都在 `piper-dynamics` 仓库的 `hw-bringup` 分支上同步，任务板和结果都放在 `.hw/` 下。

| 角色 | 做什么 |
|---|---|
| 远程 agent | 写任务、审结果、改库代码 |
| 真机 agent（Piper 主机上，接着 CAN 和机械臂） | 按任务执行，写结果，不自行扩大范围 |
| 操作员（用户） | 所有会让机械臂动的步骤都要在场确认；手握急停 |

---

## 1. 安全规则

1. **没有操作员当场确认，不发任何运动指令。** 包括 `EnableArm`、`MotionCtrl_*`、`JointCtrl`、`GripperCtrl`、示教模式切换、`MasterSlaveConfig`。只读的 `Get*` 调用可以直接执行。
2. 标 🟢 只读的任务可以直接执行。标 🔴 运动的任务，要等远程 agent 把状态改成 [TODO]，执行前还要操作员回复"确认执行"。
3. 运动速度上限：`MotionCtrl_2` 的速度百分比 ≤ 20，每个位姿之间都要停稳。
4. **不要直接运行 `piper-calibrate`（不带 `--samples` 的那种）。** 它在 URDF 关节范围的 80% 内随机采位姿，桌面环境下可能撞桌子或自碰撞。
5. 不改 CAN 接口配置，不停别人的进程。发现 CAN 被占用就记下来并停下。
6. 运动只用 `hw_scripts/` 里远程 agent 写好的脚本。脚本不合用就停下来报告，不要临时写会让机械臂运动的代码。

## 2. 协议

- **本仓库是公开的。** 绝不提交：IP、主机名、用户名、私有仓库的代码片段、AgileX 的 URDF/xacro、照片、凭据、大文件（>5 MB）、原始数据。这些东西要么用文字概括，要么直接交给用户。
- 真机侧**只改 `.hw/` 目录**：本文件里任务的状态标记和「结果」小节，以及 `.hw/results/<任务号>/` 下的小文件（md/json/txt/png，单个 <5 MB）。对库代码的修改建议写在「结果」里，由远程 agent 来改。
- 原始数据放在主机的 `~/piper_hw_runs/<任务>_<YYYYMMDD>/`，在结果里写相对路径和 `sha256sum`。
- 状态：`[TODO]` → `[DOING]` → `[DONE]` / `[BLOCKED: 原因]`。要提问就在「结果」里写一行，以 `Q:` 开头。
- 开始前先 `git pull --rebase origin hw-bringup`。每完成一个任务提交并 push 一次，commit 前缀 `hw: T2b ...`；远程侧前缀是 `remote: ...`。
- 任务描述和实际情况不符时，按实际情况记录下来并停下，不要自己猜一个替代方案。

## 3. 已确认的事实（T0–T2，2026-10-12）

- 双臂接在 `can_left` / `can_right` 上，1 Mbps；piper_sdk 0.6.2；ROS Noetic。
- 型号是 **Piper X**（用户 2026-10-13 确认）。
- 高速反馈的 `current` 是有符号 int16，单位 0.001 A，实测约 100 Hz。SDK 的 `effort` = current × 1.18125（j1–j3）/ 0.95844（j4–j6），是推算值，不是测量值。
- 关节角单位 0.001°，`motor_speed` 单位 0.001 rad/s。`io/can.py` 已按这些修正，`move_j` 已实现（需要 `allow_motion=True` 且臂已使能，速度 ≤ 20%，到不了位就超时报错）。
- T2 时两臂已使能，`motion_status=1`（未到达目标），j3 约 1.5 A，但操作员说臂靠在支撑上。控制目标可能就在支撑里面，所以 T2 的电流不能当重力参考。

---

## 4. 任务

### T2b 型号细节、末端、T2 状态 🟢 只读（只有一个查询帧需要操作员同意） [TODO]

1. **操作员**写下每条臂末端装的是什么（标准夹爪 / Revo2 手 / 其他），有没有额外挂相机或线缆。序列号和照片直接交给用户，不写进这里。
2. **操作员**描述 T2 时两臂靠在哪里：关节转到了机械限位，还是小臂或末端搁在台面、支架上。
3. 被动读取（`piper_init=False`）每条臂的 `GetArmJointCtrl()`、`GetArmModeCtrl()`（或 SDK 0.6.2 里对应的控制指令回读接口，以源码为准）：看总线上最后一次的关节目标是什么，和 T2 的实际关节角差多少。结果写到 `.hw/results/T2b/state.md`。
4. 固件版本：操作员同意后调用 SDK 的固件查询接口。它会发一个查询帧，不是运动指令；每条臂读一次。
5. Piper X 的 URDF，以及和实际末端对应的 xacro（能展开就再导出一份展开后的 URDF），**打包直接交给用户，不提交**。在结果里只写文件名和 `sha256sum`。
6. 在 `piperaio` 环境里执行 `pip install -e ".[hdf5]"`（会装 `pin`），然后跑 `pytest tests/test_io.py -q`，把结果写进 `state.md`。

#### 结果

（真机 agent 填写）

---

### T3a 示教位姿 🟢 只读（操作员手动拖动） [等待 T2b]

一次做一条臂，先做右臂，末端空载。

1. **操作员**按机械臂上的示教按钮进入拖动模式。这是操作员的物理操作，agent 不发任何模式切换指令。
2. 在仓库根目录运行（脚本只读，不发任何帧）：
   ```bash
   python hw_scripts/teach_poses.py can_right --operator-confirmed
   ```
   操作员每拖到一个位姿、松手停稳后，按 Enter 记录。结果默认存到 `.hw/results/T3/poses_can_right.json`。
3. 每条臂记录 **15–20 个位姿**：
   - 大臂（j2）、小臂（j3）、腕（j5）的角度要拉开，有伸远有收近、有高有低；j1 少量变化就够，它不影响重力。
   - 离桌面、支架、另一条臂至少 10 cm，离关节限位至少 10°。
   - 编号 0 选"大臂接近水平、往前伸"，编号 1 选"大臂竖直、收起"，T3b 会先用这两个。
4. 操作员退出拖动模式，确认臂停在原位保持住、没有下坠。把退出后的使能状态、ctrl_mode、有没有下坠写进结果。
5. 位姿文件只有关节角，可以提交。左臂另做一份 `poses_can_left.json`。

#### 结果

（真机 agent 填写）

---

### T3b 两姿态回放：单位/符号检查 + 空载噪声底 🔴 运动 [等待审阅 T3a]

远程 agent 看过 T3a 的位姿文件、弄清 T2 的状态之后，才会把这里改成 [TODO]。预定命令如下，**现在不要执行**：

```bash
python hw_scripts/collect_poses.py can_right .hw/results/T3/poses_can_right.json \
    --indices 0,1 --speed-pct 10 --settle-s 2 --hold-s 3 --confirm-each --operator-confirmed
```

脚本先打印起始姿态和计划，输入 `YES` 才开始；每次移动前还要再按一次 `y`。输出的 `raw.npz` 和 `samples.npz` 都很小，复制到 `.hw/results/T3/` 一起提交。

### T4 静态标定 🔴 运动 [等待审阅 T3b]

回放 T3a 的全部位姿（不加 `--confirm-each`，仍然要输入 `YES`），远程侧再用 `piper-calibrate --samples` 做辨识。

### T5 已知重量验证 🔴 运动 [等待审阅 T4]

0 / 100 / 250 / 500 g（先称重，记下实际质量），回放约 10 个位姿。末端如果是 Revo2 手，怎么夹持另外商量。
