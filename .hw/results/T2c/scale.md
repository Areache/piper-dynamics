# T2c 采集链路：力矩换算、同帧位置、时间戳（2026-10-19）

> 真机 agent 填写。状态：🟢 只读（改采集程序需用户批准，本任务未改任何采集代码）。原始读数 `scale_raw.json` 在本目录。

## 1. PyAgxArm 独立环境

- 来源：`https://github.com/agilexrobotics/PyAgxArm`，浅克隆到 `tmp/PyAgxArm`。
- commit：`d90e234ba8e408cfa964b38aa9120b3131fc9f85`（`fix(nero): 调整 J6 关节限位为 -42° 至 50°`），`pyAgxArm` 版本 `1.0.0`。
- 未安装进 `piper_sdk 0.6.2` 所在的 `piperaio` 环境（未动该环境）。**独立安装被环境阻断**：根文件系统只读（无法 `conda create`），且 `pip` 无法联网（配置的清华镜像与 pypi.org 均超时）。改为把克隆源码目录加入 `PYTHONPATH`，用 `piperaio` 的 `python`（自带 `python-can 4.6.1`，纯读取，未 `pip install` 任何东西）运行 `pyAgxArm`。这满足「不改 piperaio 环境」的隔离要求，但不是严格的「新建 conda 环境」。

## 2. 连接是否发帧（源码核实）

结论：**以 `piper_x` 建立连接、只调用读取接口时，不发任何 CAN 帧。**

依据（PyAgxArm 源码）：

- `AgxArmFactory.create_arm(config)` 只实例化驱动，不连总线（`agx_arm_factory.py:251-256`）。
- `Driver.connect()`（`arm_driver_abstract.py:251-276`）只做 `comm.connect()` + 启动读线程；**没有** piper_sdk 那种 `PiperInit()`（发「查询最大角度速度 / 最大加速度 / 固件」三帧）的等价物。
- `CanCommImpl.connect()`（`can_comm.py:148-173`）只 `can.interface.Bus(...)` 打开 socketcan 并 `recv`，`send` 仅在被显式调用时发生。`enable_check_can=True` 只跑 `ip` 命令读接口状态，不写总线。
- 读接口 `get_motor_states()` / `get_driver_states()` / `get_joint_angles()` / `get_arm_status()` 全部是读取 `parser` 缓存的**已接收**反馈（`driver.py:290-676`），不发帧。
- 会发帧的只有显式动作：`enable()` / `disable()` / `move_*` / `set_*` / `reset()` / `electronic_emergency_stop()` / `get_firmware()`（发 `ArmMsgReqFirmware` 查询帧）等。本次**均未调用**。

## 3. 被动读 10 s，torque / current 中位数比值

脚本 `tmp/hw_scripts/read_torque_t2c.py`：`piper_x` + `firmeware_version=DEFAULT`，每臂连 `socketcan`，只调 `get_motor_states(1..6)` 轮询 10 s（约 1950 样本/关节），另 `get_driver_states` 读一次温度。两臂均已上电、使能、静止。

> 补充（T2b 步骤 4 后）：实测固件软件版本为 `S-V1.9-0`，对应 PyAgxArm 的 `PiperFW.V189`。本次读取用了 DEFAULT，但高速反馈（0x251–0x256）帧格式跨固件稳定，`torque/current` 仍正确解码；涉及按固件版本区分的接口（如 `get_firmware`）应改用 `V189`。

PyAgxArm 源码换算：`current` = 有符号 int16（byte 2-3）× 0.001 A；`torque` = `current × k × b × c`（`parser.py:91-108, 677-694`，系数来自 `constants.py` 的 `piper_x` 行）。

理论 `k×b×c`：j1..j6 = `4.725, 4.134375, 4.725, 1.232201376, 0.7787701152, 0.7787701152`。

实测 `torque/current` 中位数比值（两臂一致，右 j1 电流恰为 0 比值无定义）：

| 关节 | 理论 k×b×c | 左臂实测 | 右臂实测 | 判定 |
|---|---:|---:|---:|---|
| 1 | 4.725 | 4.725 | —（current=0） | 一致 |
| 2 | 4.134375 | 4.134375 | 4.134375 | 一致 |
| 3 | 4.725 | 4.725 | 4.725 | 一致 |
| 4 | 1.232201376 | 1.232201376 | 1.232201376 | 一致 |
| 5 | 0.7787701152 | 0.7787701152 | 0.7787701152 | 一致 |
| 6 | 0.7787701152 | 0.7787701152 | 0.7787701152 | 一致 |

（注：比值精确等于 k×b×c 是构造性的——PyAgxArm 内部就是 `torque = current × k × b × c`。它的意义是确认两点：①`piper_x` 行的 k/b/c 系数与任务板第 3 节一致；②`current` 字段就是「有符号 int16 × 0.001 A」，与 piper_sdk 0.6.2 读到的原始电流相同。因此旧数据**不用重采**，用原始电流重乘 k×b×c 即可，`motor_torque`/`effort` 字段弃用。）

实测电流量级（中位数，A，静止保持）：左 `[0.032, 0.27, -1.546, -1.329, -0.032, -0.041]`，右 `[0.0, 0.278, -1.377, -1.031, -0.048, 0.005]`。主要负载在 j3/j4（重力保持），j1/j5/j6 接近零。速度全为 0（静止）。

低速反馈（`get_driver_states`）顺带读到：电压 23.0–24.0 V，驱动温度 41–53 ℃，电机温度 34–51 ℃，`bus_current` 恒为 0（低速反馈未填充总线电流，或保持工况下确实为 0）。**证明低速反馈确实带电机/驱动温度**（见 `recorder.md` 4.4）。
