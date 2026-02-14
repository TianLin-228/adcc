# Decoupled Davidson Eigensolver for adcc

## Overview

This document describes the design and theory behind the **decoupled Davidson eigensolver**, a variant of the Davidson iterative diagonalization algorithm tailored for the block structure of ADC (Algebraic Diagrammatic Construction) matrices. The key idea is to maintain **separate subspaces for singles and doubles**, yielding a richer projected problem at the same computational cost as standard Davidson.

---

## Motivation

In the standard Davidson solver (`adcc/solver/davidson.py`), each subspace vector is a full `AmplitudeVector` containing both singles (`ph`) and doubles (`pphh`) components. The singles and doubles directions in each subspace vector are "locked together" — adding one preconditioned residual to the subspace always adds exactly one new direction.

The **decoupled Davidson** approach breaks this coupling by maintaining independent subspaces for singles and doubles. From each residual, the `ph` part and the `pphh` part are separately orthogonalized and independently added to their respective subspaces. This gives **2 new subspace directions** per residual for the **same number of block-level matrix applies**, yielding a richer projected eigenvalue problem and potentially faster convergence.

---

## Mathematical Formulation

### ADC Matrix Block Structure

The ADC matrix `M` acting on `AmplitudeVector(ph=u_s, pphh=u_d)` has a 2x2 block structure:

```
M = [ M_ss   M_sd ]     where  M_ss = ph_ph block
    [ M_ds   M_dd ]            M_sd = ph_pphh block
                                M_ds = pphh_ph block
                                M_dd = pphh_pphh block
```

The matrix is Hermitian (symmetric for real tensors), so `M_ds = M_sd^T`.

The action of `M` on a vector `v = (u_s, u_d)` is:

```
M v = ( M_ss u_s + M_sd u_d ,  M_ds u_s + M_dd u_d )
```

### Standard Davidson Subspace

Standard Davidson maintains a single orthonormal subspace:

```
SS = [v_0, v_1, ..., v_{n-1}]    where each v_i = AmplitudeVector(ph=..., pphh=...)
```

The projected matrix is `n x n`:  `A_proj[i,j] = <v_i | M | v_j>`.

Adding one new vector requires one full matrix-vector product = **4 block applies** (`M_ss`, `M_sd`, `M_ds`, `M_dd` each applied to the new vector's respective input blocks).

### Decoupled Davidson Subspace

The decoupled Davidson maintains two separate orthonormal subspaces:

```
SS_s = [s_0, s_1, ..., s_{n_s-1}]    (ph Tensors, orthonormal: <s_i|s_k> = delta_ik)
SS_d = [d_0, d_1, ..., d_{n_d-1}]    (pphh Tensors, orthonormal: <d_j|d_l> = delta_jl)
```

Any trial vector in the combined space has the form:

```
v = ( sum_i alpha_i * s_i ,  sum_j beta_j * d_j )
```

This can be viewed as expanding the trial vector in the basis:

```
B = { (s_0, 0), (s_1, 0), ..., (s_{n_s-1}, 0), (0, d_0), (0, d_1), ..., (0, d_{n_d-1}) }
```

Since the singles and doubles spaces are orthogonal by construction (they live in disjoint tensor spaces), and each subspace is internally orthonormal, this combined basis is orthonormal. The overlap matrix is the identity.

### Projected Matrix

The projected matrix `A_tilde` has dimension `(n_s + n_d) x (n_s + n_d)` with 4 blocks:

```
A_tilde = [ A_ss   A_sd ]
          [ A_ds   A_dd ]
```

where:

```
A_ss[i,k] = <s_i | M_ss | s_k>          (n_s x n_s, symmetric)
A_sd[i,j] = <s_i | M_sd | d_j>          (n_s x n_d)
A_ds[j,i] = <d_j | M_ds | s_i> = A_sd[i,j]   (by Hermiticity of M)
A_dd[j,l] = <d_j | M_dd | d_l>          (n_d x n_d, symmetric)
```

**Derivation of each element:**

Consider the basis vector `e^s_i = (s_i, 0)` and `e^d_j = (0, d_j)`. Then:

- `<e^s_i | M | e^s_k> = <(s_i,0) | M | (s_k,0)> = <s_i | M_ss s_k + M_sd * 0> + <0 | M_ds s_k + M_dd * 0> = <s_i | M_ss | s_k>`
- `<e^s_i | M | e^d_j> = <(s_i,0) | M | (0,d_j)> = <s_i | M_ss * 0 + M_sd d_j> + <0 | ...> = <s_i | M_sd | d_j>`
- `<e^d_j | M | e^s_i> = <s_i | M_sd | d_j>` by Hermiticity of M
- `<e^d_j | M | e^d_l> = <d_j | M_dd | d_l>`

Since `A_tilde` is real symmetric, we can use `scipy.linalg.eigh` or `scipy.sparse.linalg.eigsh` to solve the projected eigenproblem.

### Sigma (Product) Vectors

To efficiently compute `A_tilde` and residuals, we store four lists of sigma vectors:

```
Sigma_ss[i] = M_ss @ s_i    (ph tensor, for each i in 0..n_s-1)
Sigma_ds[i] = M_ds @ s_i    (pphh tensor, for each i in 0..n_s-1)
Sigma_sd[j] = M_sd @ d_j    (ph tensor, for each j in 0..n_d-1)
Sigma_dd[j] = M_dd @ d_j    (pphh tensor, for each j in 0..n_d-1)
```

Then the projected matrix elements are:

```
A_ss[i,k] = s_i . Sigma_ss[k]
A_sd[i,j] = s_i . Sigma_sd[j]
A_ds[j,i] = d_j . Sigma_ds[i]    (= A_sd[i,j] by Hermiticity, used only for verification)
A_dd[j,l] = d_j . Sigma_dd[l]
```

### Cost Analysis

**Per new singles vector `s_new`:**
- Compute `M_ss @ s_new` → 1 block apply (via `matrix.block_apply("ph_ph", s_new)`)
- Compute `M_ds @ s_new` → 1 block apply (via `matrix.block_apply("pphh_ph", s_new)`)
- Total: **2 block applies**

**Per new doubles vector `d_new`:**
- Compute `M_sd @ d_new` → 1 block apply (via `matrix.block_apply("ph_pphh", d_new)`)
- Compute `M_dd @ d_new` → 1 block apply (via `matrix.block_apply("pphh_pphh", d_new)`)
- Total: **2 block applies**

**Comparison per residual:**

| Method | Block applies | New subspace directions |
|--------|:------------:|:-----------------------:|
| Standard Davidson | 4 (1 full matvec) | 1 |
| Decoupled Davidson | 4 (2+2 block applies) | 2 |

**Same cost, doubled subspace growth rate.**

The reason this works: a standard Davidson residual `r = (r_ph, r_pphh)` is a single direction in the combined space. Splitting it into `(r_ph, 0)` and `(0, r_pphh)` gives two directions that span a 2D subspace containing the original. The decoupled subspace is therefore **strictly richer** than the standard subspace (it contains all vectors the standard subspace would contain, plus additional independent directions).

### Residual Computation

For a Ritz pair `(lambda, c = [alpha; beta])` where `alpha` has `n_s` components and `beta` has `n_d` components:

```
Ritz vector:  v = ( sum_i alpha_i * s_i ,  sum_j beta_j * d_j )

Matrix product (from stored sigmas, no new matvecs needed):
(Mv)_ph   = sum_i alpha_i * Sigma_ss[i] + sum_j beta_j * Sigma_sd[j]
(Mv)_pphh = sum_i alpha_i * Sigma_ds[i] + sum_j beta_j * Sigma_dd[j]

Residual:
r_ph   = (Mv)_ph   - lambda * sum_i alpha_i * s_i
       = sum_i alpha_i * Sigma_ss[i] + sum_j beta_j * Sigma_sd[j] - lambda * sum_i alpha_i * s_i

r_pphh = (Mv)_pphh - lambda * sum_j beta_j * d_j
       = sum_i alpha_i * Sigma_ds[i] + sum_j beta_j * Sigma_dd[j] - lambda * sum_j beta_j * d_j
```

Using `lincomb()`:

```python
r_ph   = lincomb(np.hstack([alpha, beta, -lam * alpha]),
                 Sigma_ss + Sigma_sd + SS_s, evaluate=True)
r_pphh = lincomb(np.hstack([alpha, beta, -lam * beta]),
                 Sigma_ds + Sigma_dd + SS_d, evaluate=True)
```

The residual norm is: `||r|| = sqrt(r_ph . r_ph + r_pphh . r_pphh)`.

### Convergence Criterion

Same as standard Davidson: `||r_k|| < conv_tol` for all desired eigenpairs.

### Subspace Extension

After computing residuals and applying the preconditioner + symmetrisation to get the correction vector `Delta v = (Delta_ph, Delta_pphh)`:

1. **Singles extension**: Orthogonalize `Delta_ph` against `SS_s` via Gram-Schmidt. If `||Delta_ph_orth|| > residual_min_norm`, normalize and append to `SS_s`. Compute `Sigma_ss[new]` and `Sigma_ds[new]` (2 block applies).

2. **Doubles extension**: Orthogonalize `Delta_pphh` against `SS_d` via Gram-Schmidt. If `||Delta_pphh_orth|| > residual_min_norm`, normalize and append to `SS_d`. Compute `Sigma_sd[new]` and `Sigma_dd[new]` (2 block applies).

### Incremental Projection Update

When new vectors are added, we only compute the new rows/columns of each block of `A_tilde`:

- New singles vector at index `i_new` in `SS_s`:
  - `A_ss[i_new, k]` for all `k <= i_new` (exploiting symmetry)
  - `A_sd[i_new, j]` for all `j` in `0..n_d-1`
- New doubles vector at index `j_new` in `SS_d`:
  - `A_dd[j_new, l]` for all `l <= j_new` (exploiting symmetry)
  - `A_sd[i, j_new]` for all `i` in `0..n_s-1`

### Restart (Subspace Collapse)

When `n_s + n_d` approaches `max_subspace`, we collapse both subspaces to the current Ritz vectors. For each Ritz vector `k` with coefficients `c_k = [alpha_k; beta_k]`:

1. New singles vector: `s'_k = sum_i alpha^k_i * s_i`
2. New doubles vector: `d'_k = sum_j beta^k_j * d_j`
3. Sigma vectors transform identically:
   - `Sigma_ss'[k] = sum_i alpha^k_i * Sigma_ss[i]`
   - `Sigma_ds'[k] = sum_i alpha^k_i * Sigma_ds[i]`
   - `Sigma_sd'[k] = sum_j beta^k_j * Sigma_sd[j]`
   - `Sigma_dd'[k] = sum_j beta^k_j * Sigma_dd[j]`

**No new matrix applies are needed.**

After collapse, the new singles vectors `{s'_k}` are generally NOT orthogonal to each other (the Ritz coefficients are orthonormal in the combined space, but `sum_i alpha^k_i * alpha^l_i != delta_kl` in general). We therefore:

1. Orthogonalize `SS_s` via modified Gram-Schmidt, applying the same transformations to `Sigma_ss` and `Sigma_ds`
2. Orthogonalize `SS_d` via modified Gram-Schmidt, applying the same transformations to `Sigma_sd` and `Sigma_dd`
3. Drop any vectors with norm below threshold
4. Recompute the 4 blocks of `A_tilde` from the stored sigma vectors

### Eigenvector Recovery

On convergence, the full eigenvectors are reconstructed:

```python
ev_k = AmplitudeVector(
    ph   = lincomb(alpha_k, SS_s, evaluate=True),
    pphh = lincomb(beta_k,  SS_d, evaluate=True)
)
```

### Variational Property

Since `A_tilde` is a proper Rayleigh-Ritz projection of the Hermitian matrix `M` onto an orthonormal subspace, the computed Ritz values are variational bounds. For `which="SA"` (smallest algebraic), they are upper bounds to the true eigenvalues.

---

## Implementation Plan

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `adcc/solver/decoupled_davidson.py` | **Create** | Main solver implementation |
| `adcc/solver/__init__.py` | Edit | Add import and export |
| `adcc/workflow.py` | Edit | Add `"decoupled_davidson"` eigensolver option |

### Key Components of `decoupled_davidson.py`

1. **`DecoupledDavidsonState`** — State class inheriting from `EigenSolverStateBase`, with separate `subspace_vectors_s` and `subspace_vectors_d` lists.

2. **`decoupled_davidson_iterations()`** — Core iteration loop mirroring `davidson_iterations()`:
   - Assemble `(n_s+n_d) x (n_s+n_d)` projected matrix from 4 blocks
   - Solve projected eigenproblem via `scipy.linalg.eigh` / `scipy.sparse.linalg.eigsh`
   - Compute residuals from stored sigma vectors (no new matvecs)
   - Check convergence
   - Restart (collapse + re-orthogonalize) when subspace is full
   - Precondition residuals (full `AmplitudeVector`)
   - Symmetrise residuals (same as standard)
   - Split and orthogonalize into separate subspaces
   - Compute block applies for new vectors

3. **`decoupled_eigsh()`** — Entry point with same signature as `davidson.eigsh()`, with larger default `max_subspace`.

4. **`decoupled_jacobi_davidson()` / `decoupled_davidson()`** — Convenience wrappers.

5. **`default_print()`** — Modified to show `n_s + n_d` subspace sizes (e.g., `"3s+5d"` format).

### Key Existing Functions Reused

- `matrix.block_apply(block, tensor)` — `AdcMatrix.py:359-372`
- `lincomb(coefficients, tensors, evaluate=True)` — `functions.py:92-135` (works on raw `Tensor` lists)
- `select_eigenpairs(eigenvalues, n_ep, which)` — `solver/common.py:26-46`
- `IndexSymmetrisation` / `IndexSpinSymmetrisation` — `solver/explicit_symmetrisation.py`
- `JacobiPreconditioner` — `solver/preconditioner.py`
- `EigenSolverStateBase` — `solver/SolverStateBase.py`

### Edge Cases

- **No doubles block** (ADC(0)/ADC(1)): Fall back to standard Davidson since there's no block structure to decouple.
- **Empty `SS_d`** at start: All guesses may be singles-only. The doubles subspace grows from the pphh part of the first residuals.
- **Near-zero component**: If the ph or pphh part of a residual has negligible norm after orthogonalization, skip adding it (only add the non-zero part). The cost is then only 2 block applies instead of 4.

### Workflow Integration

In `workflow.py`, add a new branch in `diagonalise_adcmatrix()`:

```python
elif eigensolver == "decoupled_davidson":
    n_guesses_per_state = 2
    callback = setup_solver_printing(
        "Decoupled Jacobi-Davidson", matrix, kind,
        solver.decoupled_davidson.default_print, output=output)
    run_eigensolver = solver.decoupled_davidson.decoupled_jacobi_davidson
```

### Usage

```python
import adcc

# Standard usage via run_adc
state = adcc.adc2(scfres, n_singlets=3, eigensolver="decoupled_davidson")

# Direct solver call
from adcc.solver.decoupled_davidson import decoupled_jacobi_davidson
result = decoupled_jacobi_davidson(matrix, guesses, n_ep=3)
```

---

## Verification Plan

1. **Unit test**: Run both `davidson.eigsh` and `decoupled_davidson.decoupled_eigsh` on the same ADC(2) problem (e.g., water/STO-3G). Verify eigenvalues match within tolerance.

2. **Convergence comparison**: Compare `n_applies` and `n_iter` between standard and decoupled Davidson. The decoupled version should reach convergence in fewer iterations.

3. **Integration test**: Run `adcc.adc2(scfres, n_singlets=3, eigensolver="decoupled_davidson")` end-to-end.

4. **Edge cases**: Test with singles-only guesses, ADC(1) (no coupling), and CVS variants.
