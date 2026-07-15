# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""按身体分组 + 逐关节量化 SONIC 闭环抖动（sonic_jitter_verify.py 产物的下钻分析器）。

sonic_jitter_report.py 给的是组级汇总；本脚本回答两个更细的问题：
①抖动在身体上怎么分布（手臂/腰/腿哪组最抖、哪个关节最抖）；
②源头在哪侧（目标侧 hf ≫ 实测侧 = policy 输出在抖，软 PD 在过滤）。

判例（2026-07-16 键盘站立协议）：站立下"上抖下稳"不成立——腿 0.178° > 腰 0.164°
> 手臂 0.082°，实测 Top10 全是腿+腰；三组目标侧均为实测的 ~5 倍，源头在 deploy
输出侧，修 plant 不对症。

用法:
    python3 scripts/tools/sonic_jitter_by_group.py /tmp/sonic_jitter/a.npz [b.npz ...]
"""

import sys

import numpy as np

K = 5  # 100ms 滑动均值窗 @ 50Hz，与 sonic_jitter_report.py 同款高通


def hf_rms(x: np.ndarray) -> np.ndarray:
    """逐列 100ms 滑动均值高通，返回每列残差 RMS（输入 rad，输出 deg）。"""
    kern = np.ones(K) / K
    sm = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), 0, x)
    r = (x - sm)[K:-K]
    return np.degrees(np.sqrt((r**2).mean(axis=0)))


def group_of(name: str) -> str:
    if any(k in name for k in ("shoulder", "elbow", "wrist")):
        return "手臂"
    if "waist" in name:
        return "腰"
    if any(k in name for k in ("hip", "knee", "ankle")):
        return "腿"
    return "其他"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for path in sys.argv[1:]:
        d = np.load(path)
        names = [str(n) for n in d["joint_names"]]
        free = d["phase"] == 1
        if not free.any():
            print(f"\n=== {path}: 无自由根段（--no_unlock 轮？），跳过 ===")
            continue
        mq, mt = hf_rms(d["q"][free]), hf_rms(d["target"][free])
        print(f"\n=== {path} 自由根段 {free.sum() / 50:.0f}s ===")
        print("组别   实测hf(deg)  目标hf(deg)  过滤比")
        for g in ("手臂", "腰", "腿"):
            idx = [i for i, n in enumerate(names) if group_of(n) == g]
            a = float(np.sqrt((mq[idx] ** 2).mean()))
            b = float(np.sqrt((mt[idx] ** 2).mean()))
            print(f"{g:<4}   {a:10.3f}  {b:10.3f}  {b / max(a, 1e-9):6.1f}x")
        print("\n实测侧最抖 Top10 关节:")
        for i in np.argsort(mq)[::-1][:10]:
            print(f"  {names[i]:<28} 实测 {mq[i]:6.3f}  目标 {mt[i]:6.3f}  [{group_of(names[i])}]")
        print("\n目标侧最抖 Top5 关节:")
        for i in np.argsort(mt)[::-1][:5]:
            print(f"  {names[i]:<28} 目标 {mt[i]:6.3f}  实测 {mq[i]:6.3f}  [{group_of(names[i])}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
