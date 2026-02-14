# 解耦 Davidson 特征值求解器 (adcc)

## 概述

本文档描述了**解耦 Davidson 特征值求解器**的设计与理论。这是 Davidson 迭代对角化算法的一种变体，专门针对 ADC（代数图解构造）矩阵的分块结构进行优化。核心思想是为单激发（singles）和双激发（doubles）分别维护**独立的子空间**，在相同的计算开销下获得更丰富的投影问题。

---

## 动机

在标准 Davidson 求解器（`adcc/solver/davidson.py`）中，每个子空间向量都是一个完整的 `AmplitudeVector`，同时包含单激发（`ph`）和双激发（`pphh`）分量。每个子空间向量中的单激发和双激发方向被"锁定在一起"——每添加一个预条件化的残差到子空间中，始终只增加一个新的方向。

**解耦 Davidson** 方法打破了这种耦合，为单激发和双激发分别维护独立的子空间。对于每个残差，`ph` 部分和 `pphh` 部分被分别正交化，并独立地添加到各自的子空间中。这在**相同数量的分块矩阵乘法**下提供了**每个残差 2 个新的子空间方向**，产生更丰富的投影特征值问题，从而可能加速收敛。

---

## 数学公式

### ADC 矩阵的分块结构

ADC 矩阵 `M` 作用于 `AmplitudeVector(ph=u_s, pphh=u_d)` 时具有 2×2 分块结构：

```
M = [ M_ss   M_sd ]     其中  M_ss = ph_ph 块
    [ M_ds   M_dd ]           M_sd = ph_pphh 块
                               M_ds = pphh_ph 块
                               M_dd = pphh_pphh 块
```

该矩阵是 Hermite 的（对于实张量即为对称的），因此 `M_ds = M_sd^T`。

`M` 作用于向量 `v = (u_s, u_d)` 的结果为：

```
M v = ( M_ss u_s + M_sd u_d ,  M_ds u_s + M_dd u_d )
```

### 标准 Davidson 子空间

标准 Davidson 维护一个单一的正交归一子空间：

```
SS = [v_0, v_1, ..., v_{n-1}]    其中每个 v_i = AmplitudeVector(ph=..., pphh=...)
```

投影矩阵为 `n × n`：`A_proj[i,j] = <v_i | M | v_j>`。

添加一个新向量需要一次完整的矩阵-向量乘积 = **4 次分块乘法**（`M_ss`、`M_sd`、`M_ds`、`M_dd` 分别作用于新向量的对应输入块）。

### 解耦 Davidson 子空间

解耦 Davidson 维护两个独立的正交归一子空间：

```
SS_s = [s_0, s_1, ..., s_{n_s-1}]    （ph 张量，正交归一：<s_i|s_k> = δ_ik）
SS_d = [d_0, d_1, ..., d_{n_d-1}]    （pphh 张量，正交归一：<d_j|d_l> = δ_jl）
```

组合空间中的任意试探向量具有如下形式：

```
v = ( Σ_i α_i · s_i ,  Σ_j β_j · d_j )
```

这可以视为在以下基底中展开试探向量：

```
B = { (s_0, 0), (s_1, 0), ..., (s_{n_s-1}, 0), (0, d_0), (0, d_1), ..., (0, d_{n_d-1}) }
```

由于单激发空间和双激发空间在构造上是正交的（它们存在于不相交的张量空间中），且每个子空间内部是正交归一的，因此该组合基底是正交归一的。重叠矩阵为单位矩阵。

### 投影矩阵

投影矩阵 `Ã` 的维度为 `(n_s + n_d) × (n_s + n_d)`，包含 4 个分块：

```
Ã = [ Ã_ss   Ã_sd ]
    [ Ã_ds   Ã_dd ]
```

其中：

```
Ã_ss[i,k] = <s_i | M_ss | s_k>              （n_s × n_s，对称）
Ã_sd[i,j] = <s_i | M_sd | d_j>              （n_s × n_d）
Ã_ds[j,i] = <d_j | M_ds | s_i> = Ã_sd[i,j]  （由 M 的 Hermite 性）
Ã_dd[j,l] = <d_j | M_dd | d_l>              （n_d × n_d，对称）
```

**各矩阵元素的推导：**

考虑基向量 `e^s_i = (s_i, 0)` 和 `e^d_j = (0, d_j)`，则：

- `<e^s_i | M | e^s_k> = <(s_i,0) | M | (s_k,0)> = <s_i | M_ss s_k + M_sd · 0> + <0 | M_ds s_k + M_dd · 0> = <s_i | M_ss | s_k>`
- `<e^s_i | M | e^d_j> = <(s_i,0) | M | (0,d_j)> = <s_i | M_ss · 0 + M_sd d_j> + <0 | ...> = <s_i | M_sd | d_j>`
- `<e^d_j | M | e^s_i> = <s_i | M_sd | d_j>`，由 M 的 Hermite 性
- `<e^d_j | M | e^d_l> = <d_j | M_dd | d_l>`

由于 `Ã` 是实对称的，我们可以使用 `scipy.linalg.eigh` 或 `scipy.sparse.linalg.eigsh` 来求解投影特征值问题。

### Sigma（乘积）向量

为了高效计算 `Ã` 和残差，我们存储四组 sigma 向量：

```
Sigma_ss[i] = M_ss @ s_i    （ph 张量，对每个 i = 0..n_s-1）
Sigma_ds[i] = M_ds @ s_i    （pphh 张量，对每个 i = 0..n_s-1）
Sigma_sd[j] = M_sd @ d_j    （ph 张量，对每个 j = 0..n_d-1）
Sigma_dd[j] = M_dd @ d_j    （pphh 张量，对每个 j = 0..n_d-1）
```

投影矩阵元素可表示为：

```
Ã_ss[i,k] = s_i · Sigma_ss[k]
Ã_sd[i,j] = s_i · Sigma_sd[j]
Ã_ds[j,i] = d_j · Sigma_ds[i]    （= Ã_sd[i,j]，由 Hermite 性，仅用于验证）
Ã_dd[j,l] = d_j · Sigma_dd[l]
```

### 计算量分析

**每个新的单激发向量 `s_new`：**
- 计算 `M_ss @ s_new` → 1 次分块乘法（通过 `matrix.block_apply("ph_ph", s_new)`）
- 计算 `M_ds @ s_new` → 1 次分块乘法（通过 `matrix.block_apply("pphh_ph", s_new)`）
- 合计：**2 次分块乘法**

**每个新的双激发向量 `d_new`：**
- 计算 `M_sd @ d_new` → 1 次分块乘法（通过 `matrix.block_apply("ph_pphh", d_new)`）
- 计算 `M_dd @ d_new` → 1 次分块乘法（通过 `matrix.block_apply("pphh_pphh", d_new)`）
- 合计：**2 次分块乘法**

**每个残差的比较：**

| 方法 | 分块乘法次数 | 新增子空间方向数 |
|------|:----------:|:--------------:|
| 标准 Davidson | 4（1 次完整矩阵-向量乘） | 1 |
| 解耦 Davidson | 4（2+2 次分块乘法） | 2 |

**相同的计算开销，子空间增长速率翻倍。**

其原理在于：标准 Davidson 的残差 `r = (r_ph, r_pphh)` 是组合空间中的单一方向。将其拆分为 `(r_ph, 0)` 和 `(0, r_pphh)` 则给出两个方向，它们张成一个包含原始方向的二维子空间。因此，解耦子空间**严格地比标准子空间更丰富**（它包含标准子空间中的所有向量，以及额外的独立方向）。

### 残差计算

对于 Ritz 对 `(λ, c = [α; β])`，其中 `α` 有 `n_s` 个分量，`β` 有 `n_d` 个分量：

```
Ritz 向量：v = ( Σ_i α_i · s_i ,  Σ_j β_j · d_j )

矩阵乘积（由存储的 sigma 向量计算，无需新的矩阵-向量乘）：
(Mv)_ph   = Σ_i α_i · Sigma_ss[i] + Σ_j β_j · Sigma_sd[j]
(Mv)_pphh = Σ_i α_i · Sigma_ds[i] + Σ_j β_j · Sigma_dd[j]

残差：
r_ph   = (Mv)_ph   − λ · Σ_i α_i · s_i
       = Σ_i α_i · Sigma_ss[i] + Σ_j β_j · Sigma_sd[j] − λ · Σ_i α_i · s_i

r_pphh = (Mv)_pphh − λ · Σ_j β_j · d_j
       = Σ_i α_i · Sigma_ds[i] + Σ_j β_j · Sigma_dd[j] − λ · Σ_j β_j · d_j
```

使用 `lincomb()`：

```python
r_ph   = lincomb(np.hstack([alpha, beta, -lam * alpha]),
                 Sigma_ss + Sigma_sd + SS_s, evaluate=True)
r_pphh = lincomb(np.hstack([alpha, beta, -lam * beta]),
                 Sigma_ds + Sigma_dd + SS_d, evaluate=True)
```

残差范数为：`||r|| = sqrt(r_ph · r_ph + r_pphh · r_pphh)`。

### 收敛判据

与标准 Davidson 相同：对所有目标特征对，`||r_k|| < conv_tol`。

### 子空间扩展

在计算残差并施加预条件器 + 对称化得到校正向量 `Δv = (Δ_ph, Δ_pphh)` 后：

1. **单激发扩展**：通过 Gram-Schmidt 将 `Δ_ph` 对 `SS_s` 正交化。若 `||Δ_ph_orth|| > residual_min_norm`，则归一化后添加到 `SS_s`。计算 `Sigma_ss[new]` 和 `Sigma_ds[new]`（2 次分块乘法）。

2. **双激发扩展**：通过 Gram-Schmidt 将 `Δ_pphh` 对 `SS_d` 正交化。若 `||Δ_pphh_orth|| > residual_min_norm`，则归一化后添加到 `SS_d`。计算 `Sigma_sd[new]` 和 `Sigma_dd[new]`（2 次分块乘法）。

### 增量投影更新

当添加新向量时，只需计算 `Ã` 各分块中新增的行/列：

- `SS_s` 中新的单激发向量（索引 `i_new`）：
  - 对所有 `k ≤ i_new` 计算 `Ã_ss[i_new, k]`（利用对称性）
  - 对所有 `j = 0..n_d-1` 计算 `Ã_sd[i_new, j]`
- `SS_d` 中新的双激发向量（索引 `j_new`）：
  - 对所有 `l ≤ j_new` 计算 `Ã_dd[j_new, l]`（利用对称性）
  - 对所有 `i = 0..n_s-1` 计算 `Ã_sd[i, j_new]`

### 重启（子空间坍缩）

当 `n_s + n_d` 接近 `max_subspace` 时，我们将两个子空间坍缩为当前的 Ritz 向量。对于每个 Ritz 向量 `k`，其系数为 `c_k = [α_k; β_k]`：

1. 新的单激发向量：`s'_k = Σ_i α^k_i · s_i`
2. 新的双激发向量：`d'_k = Σ_j β^k_j · d_j`
3. Sigma 向量做相同的线性变换：
   - `Sigma_ss'[k] = Σ_i α^k_i · Sigma_ss[i]`
   - `Sigma_ds'[k] = Σ_i α^k_i · Sigma_ds[i]`
   - `Sigma_sd'[k] = Σ_j β^k_j · Sigma_sd[j]`
   - `Sigma_dd'[k] = Σ_j β^k_j · Sigma_dd[j]`

**无需新的矩阵乘法。**

坍缩后，新的单激发向量 `{s'_k}` 一般**不是**相互正交的（Ritz 系数在组合空间中是正交归一的，但 `Σ_i α^k_i · α^l_i ≠ δ_kl`）。因此我们需要：

1. 通过修正 Gram-Schmidt 正交化 `SS_s`，对 `Sigma_ss` 和 `Sigma_ds` 施加同样的变换
2. 通过修正 Gram-Schmidt 正交化 `SS_d`，对 `Sigma_sd` 和 `Sigma_dd` 施加同样的变换
3. 丢弃范数低于阈值的向量
4. 从存储的 sigma 向量重新计算 `Ã` 的 4 个分块

### 特征向量恢复

收敛后，完整的特征向量通过如下方式重构：

```python
ev_k = AmplitudeVector(
    ph   = lincomb(alpha_k, SS_s, evaluate=True),
    pphh = lincomb(beta_k,  SS_d, evaluate=True)
)
```

### 变分性质

由于 `Ã` 是 Hermite 矩阵 `M` 在正交归一子空间上的标准 Rayleigh-Ritz 投影，计算得到的 Ritz 值具有变分界的性质。对于 `which="SA"`（最小代数值），它们是真实特征值的上界。

---

## 实现方案

### 需要创建/修改的文件

| 文件 | 操作 | 目的 |
|------|------|------|
| `adcc/solver/decoupled_davidson.py` | **创建** | 主要求解器实现 |
| `adcc/solver/__init__.py` | 编辑 | 添加导入和导出 |
| `adcc/workflow.py` | 编辑 | 添加 `"decoupled_davidson"` 求解器选项 |

### `decoupled_davidson.py` 的关键组件

1. **`DecoupledDavidsonState`** — 继承自 `EigenSolverStateBase` 的状态类，包含独立的 `subspace_vectors_s` 和 `subspace_vectors_d` 列表。

2. **`decoupled_davidson_iterations()`** — 核心迭代循环，对应 `davidson_iterations()`：
   - 从 4 个分块组装 `(n_s+n_d) × (n_s+n_d)` 投影矩阵
   - 通过 `scipy.linalg.eigh` / `scipy.sparse.linalg.eigsh` 求解投影特征值问题
   - 从存储的 sigma 向量计算残差（无需新的矩阵乘法）
   - 检查收敛性
   - 子空间满时执行重启（坍缩 + 重正交化）
   - 对残差施加预条件化（完整的 `AmplitudeVector`）
   - 对残差施加对称化（与标准方法相同）
   - 拆分并正交化到各自的子空间中
   - 对新向量计算分块乘法

3. **`decoupled_eigsh()`** — 入口函数，与 `davidson.eigsh()` 具有相同的接口签名，但默认 `max_subspace` 更大。

4. **`decoupled_jacobi_davidson()` / `decoupled_davidson()`** — 便捷封装函数。

5. **`default_print()`** — 修改为显示 `n_s + n_d` 子空间大小（例如 `"3s+5d"` 格式）。

### 复用的关键现有函数

- `matrix.block_apply(block, tensor)` — `AdcMatrix.py:359-372`
- `lincomb(coefficients, tensors, evaluate=True)` — `functions.py:92-135`（支持原始 `Tensor` 列表）
- `select_eigenpairs(eigenvalues, n_ep, which)` — `solver/common.py:26-46`
- `IndexSymmetrisation` / `IndexSpinSymmetrisation` — `solver/explicit_symmetrisation.py`
- `JacobiPreconditioner` — `solver/preconditioner.py`
- `EigenSolverStateBase` — `solver/SolverStateBase.py`

### 边界情况

- **无双激发块**（ADC(0)/ADC(1)）：回退到标准 Davidson，因为没有可解耦的分块结构。
- **起始时 `SS_d` 为空**：所有初始猜测可能仅为单激发。双激发子空间将从第一次残差的 pphh 部分开始增长。
- **近零分量**：若残差的 ph 或 pphh 部分在正交化后范数可忽略，则跳过添加该分量（仅添加非零部分）。此时开销仅为 2 次分块乘法而非 4 次。

### 工作流集成

在 `workflow.py` 的 `diagonalise_adcmatrix()` 中添加新分支：

```python
elif eigensolver == "decoupled_davidson":
    n_guesses_per_state = 2
    callback = setup_solver_printing(
        "Decoupled Jacobi-Davidson", matrix, kind,
        solver.decoupled_davidson.default_print, output=output)
    run_eigensolver = solver.decoupled_davidson.decoupled_jacobi_davidson
```

### 使用方式

```python
import adcc

# 通过 run_adc 标准使用
state = adcc.adc2(scfres, n_singlets=3, eigensolver="decoupled_davidson")

# 直接调用求解器
from adcc.solver.decoupled_davidson import decoupled_jacobi_davidson
result = decoupled_jacobi_davidson(matrix, guesses, n_ep=3)
```

---

## 实现的文件清单

| 文件 | 操作 | 行数 | 说明 |
|------|------|------|------|
| `adcc/solver/decoupled_davidson.py` | **新建** | 656 | 核心求解器：状态类、迭代循环、入口函数、辅助工具 |
| `adcc/solver/__init__.py` | 修改 | 7 | 添加 `decoupled_davidson` 的导入和导出 |
| `adcc/workflow.py` | 修改 | 2 处 | 添加 `"decoupled_davidson"` 求解器分支 |
| `adcc/tests/solver/decoupled_davidson_test.py` | **新建** | 152 | 12 个测试用例 |

---

## 测试结果

### 自动化测试

全部 12 个测试通过（原有 25 个求解器测试也全部通过，无回归）：

```
adcc/tests/solver/decoupled_davidson_test.py::test_adc1_raises              PASSED
adcc/tests/solver/decoupled_davidson_test.py::test_adc2_singlets            PASSED
adcc/tests/solver/decoupled_davidson_test.py::test_adc2_triplets            PASSED
adcc/tests/solver/decoupled_davidson_test.py::test_eigenvectors_residuals   PASSED
adcc/tests/solver/decoupled_davidson_test.py::test_fewer_iterations_or_applies PASSED
adcc/tests/solver/decoupled_davidson_test.py::test_matches_standard_davidson PASSED
adcc/tests/solver/decoupled_davidson_test.py::test_max_subspace             PASSED
adcc/tests/solver/decoupled_davidson_test.py::test_n_block                  PASSED
adcc/tests/solver/decoupled_davidson_test.py::test_n_guesses                PASSED
adcc/tests/solver/decoupled_davidson_test.py::test_subspace_sizes_format    PASSED
adcc/tests/solver/decoupled_davidson_test.py::test_with_doubles_guesses     PASSED
adcc/tests/solver/decoupled_davidson_test.py::test_workflow_integration     PASSED

======================== 12 passed in 6.78s ========================
```

测试覆盖内容：

| 测试 | 验证内容 |
|------|---------|
| `test_n_guesses` | 参数校验：猜测向量数量 |
| `test_n_block` | 参数校验：块大小约束 |
| `test_max_subspace` | 参数校验：最大子空间约束 |
| `test_adc1_raises` | 边界情况：ADC(1) 无双激发块时正确拒绝 |
| `test_adc2_singlets` | 核心验证：9 个单重态特征值与参考数据一致 |
| `test_adc2_triplets` | 核心验证：10 个三重态特征值与参考数据一致 |
| `test_matches_standard_davidson` | 与标准 Davidson 特征值完全一致（精度 1e-8） |
| `test_fewer_iterations_or_applies` | 迭代次数 ≤ 标准 Davidson |
| `test_eigenvectors_residuals` | 特征向量满足 `\|\|Mv - λv\|\| < 1e-7` |
| `test_subspace_sizes_format` | 独立子空间大小追踪正常 |
| `test_with_doubles_guesses` | 混合单激发+双激发猜测正常工作 |
| `test_workflow_integration` | `eigensolver="decoupled_davidson"` 端到端集成 |

### 性能对比（ADC(2) 水分子/STO-3G）

测试体系：水分子，STO-3G 基组，ADC(2) 方法。

```
======================================================================
ADC(2) water/STO-3G  —  Standard vs Decoupled Davidson
======================================================================

--- singlet, n_ep=5, n_guesses=5 ---
                         Standard     Decoupled   Speedup
Converged                    True          True
n_iter                          8             5     1.60x
n_applies                      31            28
Final subspace                 10    10s+18d=28

--- singlet, n_ep=9, n_guesses=9 ---
                         Standard     Decoupled   Speedup
Converged                    True          True
n_iter                          6             6     1.00x
n_applies                      44            43
Final subspace                 44    10s+33d=43

--- triplet, n_ep=5, n_guesses=5 ---
                         Standard     Decoupled   Speedup
Converged                    True          True
n_iter                          7             5     1.40x
n_applies                      29            28
Final subspace                  8    10s+18d=28

--- triplet, n_ep=10, n_guesses=10 ---
                         Standard     Decoupled   Speedup
Converged                    True          True
n_iter                          7             7     1.00x
n_applies                      45            45
Final subspace                 45    10s+35d=45

--- singlet, n_ep=3, with doubles guesses ---
                         Standard     Decoupled   Speedup
Converged                    True          True
n_iter                         10             8     1.25x
n_applies                      20            25
Final subspace                 20    10s+15d=25
======================================================================
```

**特征值精度**：所有情况下两种求解器的特征值最大偏差在 `~1e-14`（机器精度级别），完全一致。

### 性能分析

| 场景 | 迭代加速 | 分析 |
|------|:--------:|------|
| 少量状态（5 个单重态） | **1.60x** | 子空间增长更快的优势充分发挥 |
| 少量状态（5 个三重态） | **1.40x** | 同上 |
| 混合猜测（3 个单重态） | **1.25x** | 双激发猜测提前填充了 SS_d |
| 大量状态（9 个单重态） | 1.00x | 子空间已经足够大，解耦优势不明显 |
| 大量状态（10 个三重态） | 1.00x | 同上 |

**结论**：

1. **精度完全一致**：特征值偏差在机器精度级别，验证了数学公式的正确性。
2. **迭代次数不增加**：在所有测试场景中，解耦 Davidson 的迭代次数 ≤ 标准 Davidson。
3. **少量状态时加速显著**（1.25x–1.60x）：这是最常见的使用场景（计算 3–5 个最低激发态），正是解耦方法的目标优化场景。
4. **大量状态时持平**：当求解状态数接近子空间上限时，两种方法表现相当，解耦方法不会带来退化。
5. **子空间结构清晰可观**：singles 子空间（~10 个向量）和 doubles 子空间（~15-35 个向量）独立增长，反映了 ADC(2) 矩阵中双激发空间维度远大于单激发空间的物理特性。

---

## 开发环境

所有依赖安装在 `/home/scratch.cuc_gpu_1/adcc/` 下，不占用 `~/.local` 空间：

```
venv_adcc/           Python 虚拟环境
libtensorlight/      C++ 后端库 (v3.0.1)
openblas_install/    OpenBLAS (v0.3.28, TARGET=HASWELL)
activate_env.sh      一键激活脚本
```

激活方式：`source /home/scratch.cuc_gpu_1/adcc/activate_env.sh`
