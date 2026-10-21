# T2b 型号细节、末端、T2 状态（2026-10-19）

> 真机 agent 填写。状态：本任务为 🟢 只读。步骤 1/2/3/4/5 已完成（步骤 4 已获操作员同意）；步骤 6 原命令被环境阻断，但**目的已用等价方式验证通过（16/16）**（见下）。步骤 1/2 中的「逐项称重」仍待操作员提供质量数据。
> 原始数据 `state_raw.json`（被动读数）、`firmware_raw.json`（固件原始字节）在本目录。

## 1. 末端（操作员确认，2026-10-19）

- **左右两臂末端均为标准夹爪（AGX gripper）**。对应 xacro 为 `piper_x_with_gripper_description.xacro`（sha256 见步骤 5）。
- 有无额外相机/线缆、以及逐项质量（相机+支架、线缆扎带、放盘子的盘子、拧瓶盖的瓶/盖，空瓶与装水各一）和相对法兰安装位置：**待操作员补充**（`Q:`）。

## 2. T2 靠放位置（操作员确认，2026-10-19）

- T2 采集时（2026-10-12）两臂是**小臂或末端搁在台面/支架上**（外部支撑），不是关节转到机械限位。

## 4. 固件版本（操作员已同意查询，2026-10-19）

对每臂用 `SearchPiperFirmwareVersion()`（发一个 0x4AF 查询帧，非运动指令），被动读回原始字节（176 字节，`firmware_raw.json`）解析如下：

| 字段 | 左 / `can_left` | 右 / `can_right` |
|---|---|---|
| 硬件版本 | `H-V1.2-1` | `H-V1.2-1` |
| 软件版本 | `S-V1.9-0` | `S-V1.9-0` |
| 节点类型 | `Piper_X_MC` | `Piper_X_MC` |
| 生产/批次 | `260713`（2026-07-13 左右） | `260711` |

- 软件版本 `S-V1.9-0` ≥ `S-V1.8-9`，对应 PyAgxArm 的 `PiperFW.V189`（而非 DEFAULT）。T2c 的高频反馈读取用 DEFAULT 也能解出正确的 torque/current（高速反馈帧格式跨固件稳定），但固件查询等按版本区分的接口建议用 `V189`。
- 节点类型 `Piper_X_MC` 再次确认机型为 **Piper X**。

## 3. 被动读取（`piper_init=False`）—— 已完成

用 `piper_sdk 0.6.2`，`ConnectPort(piper_init=False)`（不发 PiperInit 查询帧），只调用 `Get*`。脚本 `tmp/hw_scripts/read_state_t2b.py`，输出 `state_raw.json`。采集时两臂均已上电、六关节均使能、静止。

| 项 | 左 / `can_left` | 右 / `can_right` |
|---|---|---|
| 关节角反馈（0.001°） | `[-5535, 60, 956, 29256, 4431, -1267]` | `[8082, 0, 653, 6552, 17002, 0]` |
| `arm_status`（0=正常） | 0，`err_code=0`，`ctrl_mode=1`(CAN)，`motion_status=1` | 0，`err_code=0`，`ctrl_mode=1`(CAN)，`motion_status=1` |
| 六关节使能 | 全 true | 全 true |
| 夹爪反馈（angle/effort/status，原始） | `-2700 / -1047 / 64` | `-2300 / -249 / 64` |
| `GetPiperFirmwareVersion()`（被动） | `-1199`（未发查询帧） | `-1199`（未发查询帧） |

### 总线上的「最后一次关节目标」与「模式控制」

- `GetArmJointCtrl().joint_ctrl`：两臂均为全 0，且 `time_stamp=0`、`Hz=0` —— 说明自本次 SDK 实例创建以来**没有收到任何 0x155/0x156/0x157 关节目标帧**，读到的是 `ArmMsgJointCtrl` 的默认值，不是总线上的真实目标。
- `GetArmModeCtrl()` / `GetArmCtrlCode151()`：返回 `ctrl_mode=1, move_mode=1(MOVE J), move_spd_rate_ctrl=50, mit_mode=0`，但同样 `time_stamp=0`、`Hz=0`。对照 `arm_motion_ctrl_2.py:99-105`，这些正是 `ArmMsgMotionCtrl_2` 的**构造函数默认值**（`ctrl_mode=0x01, move_mode=0x01, spd=50`），不是收到的 0x151 帧。即：本次窗口内**没有收到任何 0x151 模式控制帧**。

结论：两臂当前处于「上电 + 使能 + 原地保持」，但总线上没有活跃的关节/模式控制指令（未在跟踪某个目标点）。这与 `motion_status=1`（未到达指定点位）并存的解释是：控制器从未下发过目标，SDK 把缺省目标记为 0。

### 与 T2（2026-10-12）关节角对比

| 臂 | T2 关节角（0.001°） | 本次关节角（0.001°） | 变化 |
|---|---|---|---|
| 左 | `[-5118, -428, 794, 33530, -8652, -5768]` | `[-5535, 60, 956, 29256, 4431, -1267]` | 明显（j4/j5/j6 变化最大） |
| 右 | `[-6393, -477, 611, 27190, 5368, 6627]` | `[8082, 0, 653, 6552, 17002, 0]` | 明显（j1/j4/j5/j6 变化最大） |

两臂自 T2 以来已被移动过（或被拖动/复位），当前姿态与 T2 不一致，因此 T2 的电流不能直接当作当前姿态的重力参考。关节目标读回为 0 与 T2 实际角的「差」无意义（目标帧从未出现过）。

## 5. Piper X 的 URDF / Xacro —— 已完成（sha256，不提交文件本身）

| 文件 | SHA-256 |
|---|---|
| `piper_x_description.urdf` | `34126caac7d5b37bc2409f337ac246afbe0bb8cd47fc9f16df5038f19bd21e3a` |
| `piper_x_with_gripper_description.xacro` | `0ee3f52f9acf3060a7f3e439c6e5e46b6a53d4fa509dd495045548be6b7f90cd` |
| `piper_x_with_left_revo2_description.xacro` | `2439ca8258ba8203173b780504b5b467291b71000b0e3426aec907dba3de0738` |
| `piper_x_with_right_revo2_description.xacro` | `2a9fdb4060f7bacdd9e45c7e2f691e3f5799e0812e67b7fd30afa2b5a72dc069` |

这 4 个文件来自 `agx_arm` 描述包，按 Piper X 建模。操作员已确认末端为标准夹爪，故**与实体匹配的是 `piper_x_with_gripper_description.xacro`**（`piper_x_description.urdf` 为不含末端的裸臂描述）。展开（xacro → urdf）与打包交由用户/远程侧处理，这里只记录文件与校验和。

## 6. `pip install -e ".[hdf5]"` + `pytest tests/test_io.py` —— 原命令被环境阻断，但目的已用等价方式达成

**原命令仍 BLOCKED**（三个环境原因）：① `pin`（pinocchio）未装、pip 连不上 PyPI（清华镜像与 pypi.org 均超时）；② `piperaio` 环境在只读文件系统，`pip install -e` 写不进 site-packages；③ `tests/test_io.py` 只在远程新提交里，本机仓库没 pull 到。

**但步骤 6 的目的（校验 `io/can.py` 的字段/单位换算与运动护栏、`io/hdf5.py` 的读取逻辑）已用等价方式验证通过**：

- 关键点：`test_io.py` 只 import `piper_dynamics.io.can` / `io.hdf5`，这两个模块是**纯 numpy + h5py，不依赖 pin**。`pin` 只是因为它是 `pyproject.toml` 的基础依赖、`pip install -e .` 会顺带装上，测试本身用不到（`from .model import PiperModel` 在顶层 `__init__.py` 才引 pin）。
- 做法：从远程 `hw-bringup` 用 GitHub API 取回 `tests/test_io.py`、`src/piper_dynamics/io/can.py`、`io/hdf5.py`（sha256 见下），把 `can.py`/`hdf5.py` 用 `importlib` **单独加载**（绕开会引 pin 的顶层 `__init__.py`），用 `piperaio` 自带的 numpy + h5py 复现 `test_io.py` 里全部 16 个断言，**未装任何包、未改环境**。
- 结果：**16/16 全部通过**。包括：角度/速度/电流换算（`velocity=0.001` 修正正确）、`move_j` 的 allow_motion / 使能 / 速度≤20 / 超时护栏、`move_j` 按 0.001° 原始单位下发、HDF5 `read_episode` 左右臂选列与全零 effort 拒绝。

脚本与文件在 `tmp/hw_scripts/verify_io/`：`run_verify.py`（复现脚本）、`can.py`、`hdf5.py`、`test_io.py`（远程原样下载）。

| 文件 | SHA-256 |
|---|---|
| `can.py`（远程） | `2502c3a9d29c46fa96c76248c2994d4efbc007bf8a87d2187eaaf21b7bbeccd3` |
| `hdf5.py`（远程） | `9b7c04bab3205a989e0bf44826bd595da64eaad79bc404975e6f3590f8d8e356` |
| `test_io.py`（远程） | `62e351afdafb482911b1c2b8aed799e55463eaacb2156f1af50481df61f761b4` |

建议：严格意义的 `pip install -e ".[hdf5]"` + `pytest` 仍留给可写环境；但真机侧已把测试覆盖的每一条断言都跑过并全绿，可据此放行。

## 待操作员补充（Q:）

- `Q:` 步骤 1 剩余：有无额外相机或线缆；逐项质量（g）和相对法兰安装位置 —— 每条臂腕部相机+支架、末端线缆/扎带、放盘子的盘子、拧瓶盖的瓶/盖（空瓶与装水各一）。能拆就拆下称，不能拆写「未称」。
