#!/usr/bin/env python3
"""模块级验证：block.py 显式循环改写后的 eager 输出与导出图一致性。

覆盖生成器 + 可变列表（`y.extend(m(y[-1]) for ...)`）模式涉及的模块：
  - eager：修改后 forward 与修改前（实例级 monkeypatch）输出 max diff 必须为 0
  - 图：export_for_training 图输出与修改后 eager 输出 max diff 必须为 0

用法：
  env PYTHONPATH="$PWD" python scripts/verify_block_fixes.py
"""

import types

import torch

import ultralytics.nn.modules.block as B


def old_hgblock(self, x):
    y = [x]
    y.extend(m(y[-1]) for m in self.m)
    y = self.ec(self.sc(torch.cat(y, 1)))
    return y + x if self.add else y


def old_c2fattn(self, x, guide):
    y = list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.m)
    y.append(self.attn(y[-1], guide))
    return self.cv2(torch.cat(y, 1))


def old_repncs(self, x):
    y = list(self.cv1(x).chunk(2, 1))
    y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
    return self.cv4(torch.cat(y, 1))


def old_sppelan(self, x):
    y = [self.cv1(x)]
    y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
    return self.cv5(torch.cat(y, 1))


def old_c3f(self, x):
    y = [self.cv2(x), self.cv1(x)]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv3(torch.cat(y, 1))


def old_a2c2f(self, x):
    y = [self.cv1(x)]
    y.extend(m(y[-1]) for m in self.m)
    y = self.cv2(torch.cat(y, 1))
    if self.gamma is not None:
        return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
    return y


def old_torchvision(self, x):
    if self.split:
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
    else:
        y = self.m(x)
    return y


def make_cases():
    torch.manual_seed(0)
    cases = [
        ("HGBlock", B.HGBlock(32, 64, 64, n=3), [torch.rand(1, 32, 16, 16)]),
        ("C2fAttn", B.C2fAttn(32, 32, n=2, ec=16, nh=2, gc=64), [torch.rand(1, 32, 16, 16), torch.rand(1, 4, 64)]),
        ("RepNCSPELAN4", B.RepNCSPELAN4(32, 32, 32, 32, n=2), [torch.rand(1, 32, 16, 16)]),
        ("SPPELAN", B.SPPELAN(32, 32, 16, k=5), [torch.rand(1, 32, 16, 16)]),
        ("C3f", B.C3f(32, 32, n=2), [torch.rand(1, 32, 16, 16)]),
        ("A2C2f", B.A2C2f(32, 64, n=2), [torch.rand(1, 32, 16, 16)]),
    ]
    try:
        cases.append(("TorchVision", B.TorchVision("resnet18", weights=None, truncate=1, split=True), [torch.rand(1, 3, 32, 32)]))
    except Exception as exc:  # torchvision weights unavailable offline
        print(f"TorchVision skipped: {exc}")
    return cases


OLD_FORWARD = {
    "HGBlock": old_hgblock,
    "C2fAttn": old_c2fattn,
    "RepNCSPELAN4": old_repncs,
    "SPPELAN": old_sppelan,
    "C3f": old_c3f,
    "A2C2f": old_a2c2f,
    "TorchVision": old_torchvision,
}


def max_diff(a, b):
    if isinstance(a, (list, tuple)):
        return max(max_diff(x, y) for x, y in zip(a, b))
    return (a - b).abs().max().item()


def main():
    print(f"{'module':<14s} | eager new-vs-old | graph new-vs-eager")
    all_pass = True
    for name, module, inputs in make_cases():
        module.eval()
        with torch.no_grad():
            out_new = module(*inputs)

        # eager: instance-level monkeypatch back to the pre-fix generator forward
        module.forward = types.MethodType(OLD_FORWARD[name], module)
        with torch.no_grad():
            out_old = module(*inputs)
        eager_diff = max_diff(out_new, out_old)
        del module.forward  # restore the fixed class-level forward

        # graph: export_for_training of the fixed module must match fixed eager output
        program = torch.export.export_for_training(module, tuple(inputs))
        with torch.no_grad():
            out_graph = program.module()(*inputs)
        graph_diff = max_diff(out_new, out_graph)

        ok = eager_diff == 0.0 and graph_diff == 0.0
        all_pass = all_pass and ok
        print(f"{name:<14s} | {eager_diff:<16.8f} | {graph_diff:<16.8f} | {'PASS' if ok else 'FAIL'}")
    print("ALL:", "PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()
