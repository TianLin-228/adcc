#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import sys
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

from adcc import evaluate, lincomb
from adcc.AdcMatrix import AdcMatrixlike
from adcc.AmplitudeVector import AmplitudeVector

from .common import select_eigenpairs
from .preconditioner import JacobiPreconditioner
from .SolverStateBase import EigenSolverStateBase
from .explicit_symmetrisation import IndexSymmetrisation


class DecoupledDavidsonState(EigenSolverStateBase):
    def __init__(self, matrix, guesses):
        super().__init__(matrix)
        self.residuals = None
        self.algorithm = "decoupled_davidson"
        self.subspace_vectors_s = []  # list of ph Tensors
        self.subspace_vectors_d = []  # list of pphh Tensors
        self._split_guesses(guesses)

    def _split_guesses(self, guesses):
        """Split guess AmplitudeVectors into separate singles/doubles subspaces."""
        eps = np.finfo(float).eps
        for guess in guesses:
            if "ph" in guess.blocks:
                t = guess.ph
                norm = np.sqrt(float(t.dot(t)))
                if norm > eps:
                    self.subspace_vectors_s.append(evaluate(t / norm))
            if "pphh" in guess.blocks:
                t = guess.pphh
                norm = np.sqrt(float(t.dot(t)))
                if norm > eps:
                    self.subspace_vectors_d.append(evaluate(t / norm))

    @property
    def subspace_vectors(self):
        """Combined subspace vectors for reporting compatibility."""
        return self.subspace_vectors_s + self.subspace_vectors_d


def default_print(state, identifier, file=sys.stdout):
    """A default print function for the decoupled davidson callback"""
    from adcc.timings import strtime, strtime_short

    if identifier == "start" and state.n_iter == 0:
        print("Niter   n_ss  max_residual  time  Ritz values",
              file=file)
    elif identifier == "next_iter":
        time_iter = state.timer.current("iteration")
        n_s = len(state.subspace_vectors_s)
        n_d = len(state.subspace_vectors_d)
        ss_str = f"{n_s}s+{n_d}d"
        fmt = "{n_iter:3d}  {ss_size:>9s}  {residual:12.5g}  {tstr:5s}"
        print(fmt.format(n_iter=state.n_iter, tstr=strtime_short(time_iter),
                         ss_size=ss_str,
                         residual=np.max(state.residual_norms)),
              "", state.eigenvalues[:7], file=file)
        if hasattr(state, "subspace_orthogonality"):
            print(33 * " " + "nonorth: {:5.3g}"
                  "".format(state.subspace_orthogonality))
    elif identifier == "is_converged":
        soltime = state.timer.total("iteration")
        print("=== Converged ===", file=file)
        print("    Number of matrix applies:   ", state.n_applies, file=file)
        print("    Total solver time:          ", strtime(soltime), file=file)
    elif identifier == "restart":
        print("=== Restart ===", file=file)


def _orthogonalize_with_sigmas(vectors, sigma_lists, min_norm):
    """Orthogonalize vectors via modified Gram-Schmidt, applying the same
    transformations to the corresponding sigma vector lists.

    Parameters
    ----------
    vectors : list of Tensor
        Vectors to orthogonalize
    sigma_lists : list of list of Tensor
        Each inner list has the same length as vectors
    min_norm : float
        Minimum norm to keep a vector

    Returns
    -------
    (new_vectors, new_sigma_lists)
    """
    n = len(vectors)
    if n == 0:
        return [], [[] for _ in sigma_lists]

    result_vecs = []
    result_sigmas = [[] for _ in sigma_lists]

    for i in range(n):
        v = vectors[i]
        sigs = [sl[i] for sl in sigma_lists]

        # Modified Gram-Schmidt: subtract projections onto already-orthogonalized
        for j in range(len(result_vecs)):
            overlap = float(v.dot(result_vecs[j]))
            v = evaluate(v - overlap * result_vecs[j])
            sigs = [evaluate(s - overlap * rs[j])
                    for s, rs in zip(sigs, result_sigmas)]

        norm = np.sqrt(float(v.dot(v)))
        if norm < min_norm:
            continue

        v = evaluate(v / norm)
        sigs = [evaluate(s / norm) for s in sigs]

        result_vecs.append(v)
        for k in range(len(sigma_lists)):
            result_sigmas[k].append(sigs[k])

    return result_vecs, result_sigmas


def _build_eigenvectors(epair_mask, rvecs, n_s, n_d, SS_s, SS_d, matrix):
    """Build full AmplitudeVector eigenvectors from Ritz coefficients."""
    eigenvectors = []
    for idx in epair_mask:
        rvec = rvecs[:, idx]
        alpha = rvec[:n_s]
        beta = rvec[n_s:]
        if n_s > 0:
            ev_ph = lincomb(alpha, SS_s, evaluate=True)
        else:
            ev_ph = matrix.diagonal().ph.zeros_like()
        if n_d > 0:
            ev_pphh = lincomb(beta, SS_d, evaluate=True)
        else:
            ev_pphh = matrix.diagonal().pphh.zeros_like()
        eigenvectors.append(AmplitudeVector(ph=ev_ph, pphh=ev_pphh))
    return eigenvectors


def decoupled_davidson_iterations(matrix, state, max_subspace, max_iter,
                                  n_ep, n_block, is_converged, which,
                                  callback=None, preconditioner=None,
                                  preconditioning_method="Davidson",
                                  debug_checks=False, residual_min_norm=None,
                                  explicit_symmetrisation=None):
    """Drive the decoupled davidson iterations.

    Maintains separate orthonormal subspaces for singles (ph) and doubles
    (pphh), yielding 2 new subspace directions per residual at the same
    cost as standard Davidson (4 block applies per residual).

    Parameters
    ----------
    matrix
        Matrix to diagonalise
    state
        DecoupledDavidsonState containing the split guesses
    max_subspace : int
        Maximal subspace size (n_s + n_d)
    max_iter : int
        Maximal number of iterations
    n_ep : int
        Number of eigenpairs to be computed
    n_block : int
        Davidson block size
    is_converged
        Function to test for convergence
    which : str, optional
        Which eigenvectors to converge to
    callback : callable, optional
        Callback to run after each iteration
    preconditioner
        Preconditioner (type or instance)
    preconditioning_method : str, optional
        Preconditioning method
    debug_checks : bool, optional
        Enable potentially costly debug checks
    residual_min_norm : float or NoneType, optional
        Minimal norm for a new subspace vector
    explicit_symmetrisation
        Explicit symmetrisation to apply to new subspace vectors
    """
    if preconditioning_method not in ["Davidson", "Sleijpen-van-der-Vorst"]:
        raise ValueError("Only 'Davidson' and 'Sleijpen-van-der-Vorst' "
                         "are valid preconditioner methods")
    if preconditioning_method == "Sleijpen-van-der-Vorst":
        raise NotImplementedError("Sleijpen-van-der-Vorst preconditioning "
                                  "not yet implemented.")

    if callback is None:
        def callback(state, identifier):
            pass

    n_problem = matrix.shape[1]
    eps = np.finfo(float).eps
    if residual_min_norm is None:
        residual_min_norm = 2 * n_problem * eps

    SS_s = state.subspace_vectors_s
    SS_d = state.subspace_vectors_d
    n_s = len(SS_s)
    n_d = len(SS_d)

    assert n_s + n_d >= n_block

    # Sigma vectors: block-wise matrix products
    Sigma_ss = []  # M_ss @ s_i  (ph tensors)
    Sigma_ds = []  # M_ds @ s_i  (pphh tensors)
    Sigma_sd = []  # M_sd @ d_j  (ph tensors)
    Sigma_dd = []  # M_dd @ d_j  (pphh tensors)

    # Pre-allocate projected matrix blocks
    Ass_cont = np.empty((max_subspace, max_subspace))
    Asd_cont = np.empty((max_subspace, max_subspace))
    Add_cont = np.empty((max_subspace, max_subspace))

    callback(state, "start")
    state.timer.restart("iteration")

    # Initial block applies
    with state.timer.record("projection"):
        for i in range(n_s):
            Sigma_ss.append(evaluate(matrix.block_apply("ph_ph", SS_s[i])))
            Sigma_ds.append(evaluate(matrix.block_apply("pphh_ph", SS_s[i])))
        for j in range(n_d):
            Sigma_sd.append(evaluate(matrix.block_apply("ph_pphh", SS_d[j])))
            Sigma_dd.append(evaluate(
                matrix.block_apply("pphh_pphh", SS_d[j])))
        state.n_applies += n_s + n_d

    # Initial projection
    with state.timer.record("projection"):
        for i in range(n_s):
            for k in range(i, n_s):
                Ass_cont[i, k] = float(SS_s[i].dot(Sigma_ss[k]))
                if i != k:
                    Ass_cont[k, i] = Ass_cont[i, k]
        for i in range(n_s):
            for j in range(n_d):
                Asd_cont[i, j] = float(SS_s[i].dot(Sigma_sd[j]))
        for j in range(n_d):
            for l in range(j, n_d):
                Add_cont[j, l] = float(SS_d[j].dot(Sigma_dd[l]))
                if j != l:
                    Add_cont[l, j] = Add_cont[j, l]

    while state.n_iter < max_iter:
        state.n_iter += 1

        n_s = len(SS_s)
        n_d = len(SS_d)
        n_total = n_s + n_d

        assert n_total >= n_block
        assert n_total <= max_subspace

        # --- Rayleigh-Ritz ---
        with state.timer.record("rayleigh_ritz"):
            Afull = np.empty((n_total, n_total))
            Afull[:n_s, :n_s] = Ass_cont[:n_s, :n_s]
            if n_d > 0:
                Afull[:n_s, n_s:] = Asd_cont[:n_s, :n_d]
                Afull[n_s:, :n_s] = Asd_cont[:n_s, :n_d].T
                Afull[n_s:, n_s:] = Add_cont[:n_d, :n_d]

            if n_total == n_block:
                rvals, rvecs = la.eigh(Afull)
            else:
                rvals, rvecs = sla.eigsh(Afull, k=n_block, which=which)

        # --- Residuals ---
        with state.timer.record("residuals"):
            residuals = []
            for idx in range(n_block):
                rvec = rvecs[:, idx]
                rval = rvals[idx]
                alpha = rvec[:n_s]
                beta = rvec[n_s:]

                # r_ph = sum_i alpha_i Sigma_ss[i] + sum_j beta_j Sigma_sd[j]
                #        - lambda * sum_i alpha_i s_i
                coeffs_ph = np.hstack([alpha, beta, -rval * alpha])
                tensors_ph = Sigma_ss + Sigma_sd + SS_s
                r_ph = lincomb(coeffs_ph, tensors_ph, evaluate=True)

                # r_pphh = sum_i alpha_i Sigma_ds[i] + sum_j beta_j Sigma_dd[j]
                #          - lambda * sum_j beta_j d_j
                coeffs_pphh = np.hstack([alpha, beta, -rval * beta])
                tensors_pphh = Sigma_ds + Sigma_dd + SS_d
                r_pphh = lincomb(coeffs_pphh, tensors_pphh, evaluate=True)

                residuals.append(AmplitudeVector(ph=r_ph, pphh=r_pphh))

            epair_mask = select_eigenpairs(rvals, n_ep, which)
            state.eigenvalues = rvals[epair_mask]
            state.residuals = [residuals[i] for i in epair_mask]
            state.residual_norms = np.array([np.sqrt(float(r @ r))
                                             for r in state.residuals])

        callback(state, "next_iter")
        state.timer.restart("iteration")
        if is_converged(state):
            state.eigenvectors = _build_eigenvectors(
                epair_mask, rvecs, n_s, n_d, SS_s, SS_d, matrix)
            state.converged = True
            callback(state, "is_converged")
            state.timer.stop("iteration")
            return state

        if state.n_iter == max_iter:
            warnings.warn(la.LinAlgWarning(
                f"Maximum number of iterations (== {max_iter}) "
                "reached in decoupled davidson procedure."))
            state.eigenvectors = _build_eigenvectors(
                epair_mask, rvecs, n_s, n_d, SS_s, SS_d, matrix)
            state.timer.stop("iteration")
            state.converged = False
            return state

        # --- Restart if needed ---
        if n_total + 2 * n_block > max_subspace:
            callback(state, "restart")
            with state.timer.record("projection"):
                # Collapse both subspaces to Ritz vectors
                alpha_coeffs = rvecs[:n_s, :].T  # (n_block, n_s)
                beta_coeffs = rvecs[n_s:, :].T   # (n_block, n_d)

                # Collapse singles subspace via linear combinations
                if n_s > 0:
                    new_SS_s = [lincomb(c, SS_s, evaluate=True)
                                for c in alpha_coeffs]
                    new_Sigma_ss = [lincomb(c, Sigma_ss, evaluate=True)
                                    for c in alpha_coeffs]
                    new_Sigma_ds = [lincomb(c, Sigma_ds, evaluate=True)
                                    for c in alpha_coeffs]
                else:
                    new_SS_s, new_Sigma_ss, new_Sigma_ds = [], [], []

                # Collapse doubles subspace via linear combinations
                if n_d > 0:
                    new_SS_d = [lincomb(c, SS_d, evaluate=True)
                                for c in beta_coeffs]
                    new_Sigma_sd = [lincomb(c, Sigma_sd, evaluate=True)
                                    for c in beta_coeffs]
                    new_Sigma_dd = [lincomb(c, Sigma_dd, evaluate=True)
                                    for c in beta_coeffs]
                else:
                    new_SS_d, new_Sigma_sd, new_Sigma_dd = [], [], []

                # Re-orthogonalize within each subspace
                SS_s, (Sigma_ss, Sigma_ds) = _orthogonalize_with_sigmas(
                    new_SS_s, [new_Sigma_ss, new_Sigma_ds], residual_min_norm)
                SS_d, (Sigma_sd, Sigma_dd) = _orthogonalize_with_sigmas(
                    new_SS_d, [new_Sigma_sd, new_Sigma_dd], residual_min_norm)

                state.subspace_vectors_s = SS_s
                state.subspace_vectors_d = SS_d

                n_s = len(SS_s)
                n_d = len(SS_d)

                # Recompute projected matrix from stored sigmas
                for i in range(n_s):
                    for k in range(i, n_s):
                        Ass_cont[i, k] = float(SS_s[i].dot(Sigma_ss[k]))
                        if i != k:
                            Ass_cont[k, i] = Ass_cont[i, k]
                for i in range(n_s):
                    for j in range(n_d):
                        Asd_cont[i, j] = float(SS_s[i].dot(Sigma_sd[j]))
                for j in range(n_d):
                    for l in range(j, n_d):
                        Add_cont[j, l] = float(SS_d[j].dot(Sigma_dd[l]))
                        if j != l:
                            Add_cont[l, j] = Add_cont[j, l]
            # continue to add residuals to space

        # --- Preconditioner ---
        with state.timer.record("preconditioner"):
            if preconditioner:
                if hasattr(preconditioner, "update_shifts"):
                    rvals_eps = 1e-6
                    preconditioner.update_shifts(rvals - rvals_eps)
                preconds = evaluate(preconditioner @ residuals)
            else:
                preconds = residuals

            if explicit_symmetrisation:
                explicit_symmetrisation.symmetrise(preconds)

        # --- Extend subspaces ---
        with state.timer.record("orthogonalisation"):
            n_s_added = 0
            n_d_added = 0

            for i in range(n_block):
                pvec = preconds[i]

                # --- Orthogonalize singles part against SS_s ---
                pvec_ph = pvec.ph
                if len(SS_s) > 0:
                    overlaps = np.asarray(pvec_ph.dot(SS_s))
                    coefficients = np.hstack(([1.0], -overlaps))
                    pvec_ph = lincomb(coefficients, [pvec_ph] + SS_s,
                                      evaluate=True)
                pnorm_ph = np.sqrt(float(pvec_ph.dot(pvec_ph)))

                if pnorm_ph >= residual_min_norm:
                    # Reorthogonalization check
                    with state.timer.record("reorthogonalisation"):
                        if len(SS_s) > 0:
                            overlaps2 = np.asarray(pvec_ph.dot(SS_s))
                            max_overlap = np.max(np.abs(overlaps2)) / pnorm_ph
                            if max_overlap > n_problem * eps:
                                coefficients = np.hstack(([1.0], -overlaps2))
                                pvec_ph = lincomb(
                                    coefficients, [pvec_ph] + SS_s,
                                    evaluate=True)
                                pnorm_ph = np.sqrt(float(
                                    pvec_ph.dot(pvec_ph)))
                                state.reortho_triggers.append(max_overlap)

                    if pnorm_ph >= residual_min_norm:
                        SS_s.append(evaluate(pvec_ph / pnorm_ph))
                        n_s_added += 1

                # --- Orthogonalize doubles part against SS_d ---
                pvec_pphh = pvec.pphh
                if len(SS_d) > 0:
                    overlaps = np.asarray(pvec_pphh.dot(SS_d))
                    coefficients = np.hstack(([1.0], -overlaps))
                    pvec_pphh = lincomb(coefficients, [pvec_pphh] + SS_d,
                                        evaluate=True)
                pnorm_pphh = np.sqrt(float(pvec_pphh.dot(pvec_pphh)))

                if pnorm_pphh >= residual_min_norm:
                    with state.timer.record("reorthogonalisation"):
                        if len(SS_d) > 0:
                            overlaps2 = np.asarray(pvec_pphh.dot(SS_d))
                            max_overlap = (np.max(np.abs(overlaps2))
                                           / pnorm_pphh)
                            if max_overlap > n_problem * eps:
                                coefficients = np.hstack(([1.0], -overlaps2))
                                pvec_pphh = lincomb(
                                    coefficients, [pvec_pphh] + SS_d,
                                    evaluate=True)
                                pnorm_pphh = np.sqrt(float(
                                    pvec_pphh.dot(pvec_pphh)))
                                state.reortho_triggers.append(max_overlap)

                    if pnorm_pphh >= residual_min_norm:
                        SS_d.append(evaluate(pvec_pphh / pnorm_pphh))
                        n_d_added += 1

        if n_s_added + n_d_added == 0:
            state.timer.stop("iteration")
            state.converged = False
            state.eigenvectors = _build_eigenvectors(
                epair_mask, rvecs, n_s, n_d, SS_s, SS_d, matrix)
            warnings.warn(la.LinAlgWarning(
                "Decoupled Davidson could not generate any further vectors "
                "for the subspace. Iteration cannot be continued and will "
                "be aborted without convergence. Try a different guess."))
            return state

        # --- Block applies for new vectors ---
        with state.timer.record("projection"):
            for idx in range(len(SS_s) - n_s_added, len(SS_s)):
                Sigma_ss.append(evaluate(
                    matrix.block_apply("ph_ph", SS_s[idx])))
                Sigma_ds.append(evaluate(
                    matrix.block_apply("pphh_ph", SS_s[idx])))
            for idx in range(len(SS_d) - n_d_added, len(SS_d)):
                Sigma_sd.append(evaluate(
                    matrix.block_apply("ph_pphh", SS_d[idx])))
                Sigma_dd.append(evaluate(
                    matrix.block_apply("pphh_pphh", SS_d[idx])))
            state.n_applies += n_s_added + n_d_added

        # Update n_s, n_d
        n_s = len(SS_s)
        n_d = len(SS_d)

        # --- Incremental projection update ---
        with state.timer.record("projection"):
            # New rows/cols in A_ss
            for i in range(n_s - n_s_added, n_s):
                for k in range(i + 1):
                    Ass_cont[i, k] = float(SS_s[i].dot(Sigma_ss[k]))
                    if i != k:
                        Ass_cont[k, i] = Ass_cont[i, k]

            # New rows in A_sd (from new singles vectors)
            for i in range(n_s - n_s_added, n_s):
                for j in range(n_d):
                    Asd_cont[i, j] = float(SS_s[i].dot(Sigma_sd[j]))
            # New cols in A_sd (from new doubles vectors, existing singles)
            for j in range(n_d - n_d_added, n_d):
                for i in range(n_s - n_s_added):
                    Asd_cont[i, j] = float(SS_s[i].dot(Sigma_sd[j]))

            # New rows/cols in A_dd
            for j in range(n_d - n_d_added, n_d):
                for l in range(j + 1):
                    Add_cont[j, l] = float(SS_d[j].dot(Sigma_dd[l]))
                    if j != l:
                        Add_cont[l, j] = Add_cont[j, l]


def decoupled_eigsh(matrix, guesses, n_ep=None, n_block=None,
                    max_subspace=None, conv_tol=1e-9, which="SA",
                    max_iter=70, callback=None, preconditioner=None,
                    preconditioning_method="Davidson", debug_checks=False,
                    residual_min_norm=None,
                    explicit_symmetrisation=IndexSymmetrisation):
    """Decoupled Davidson eigensolver for ADC problems.

    Maintains separate subspaces for singles (ph) and doubles (pphh),
    yielding 2 new subspace directions per residual at the same
    computational cost as standard Davidson.

    Parameters
    ----------
    matrix
        ADC matrix instance
    guesses : list
        Guess vectors (AmplitudeVector objects)
    n_ep : int or NoneType, optional
        Number of eigenpairs to be computed
    n_block : int or NoneType, optional
        The solver block size
    max_subspace : int or NoneType, optional
        Maximal subspace size
    conv_tol : float, optional
        Convergence tolerance on the l2 norm of residuals
    which : str, optional
        Which eigenvectors to converge to (e.g. LM, LA, SM, SA)
    max_iter : int, optional
        Maximal number of iterations
    callback : callable, optional
        Callback to run after each iteration
    preconditioner
        Preconditioner (type or instance)
    preconditioning_method : str, optional
        Preconditioning method
    explicit_symmetrisation
        Explicit symmetrisation to apply to new subspace vectors
    debug_checks : bool, optional
        Enable potentially costly debug checks
    residual_min_norm : float or NoneType, optional
        Minimal norm for a new subspace vector
    """
    if not isinstance(matrix, AdcMatrixlike):
        raise TypeError("matrix is not of type AdcMatrixlike")
    for guess in guesses:
        if not isinstance(guess, AmplitudeVector):
            raise TypeError("One of the guesses is not of type AmplitudeVector")

    if "pphh" not in matrix.axis_blocks:
        raise ValueError("Decoupled Davidson requires a matrix with both "
                         "singles (ph) and doubles (pphh) blocks. "
                         "Use standard Davidson for ADC(0)/ADC(1).")

    if preconditioner is not None and isinstance(preconditioner, type):
        preconditioner = preconditioner(matrix)

    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(matrix)

    if n_ep is None:
        n_ep = len(guesses)
    elif n_ep > len(guesses):
        raise ValueError(f"n_ep (= {n_ep}) cannot exceed the number of guess "
                         f"vectors (= {len(guesses)}).")

    if n_block is None:
        n_block = n_ep
    elif n_block < n_ep:
        raise ValueError(f"n_block (= {n_block}) cannot be smaller than the "
                         f"number of states requested (= {n_ep}).")
    elif n_block > len(guesses):
        raise ValueError(f"n_block (= {n_block}) cannot exceed the number of "
                         f"guess vectors (= {len(guesses)}).")

    if not max_subspace:
        # Larger default: subspace grows ~2x faster than standard Davidson
        max_subspace = max(12 * n_ep, 40, 10 * len(guesses))
    elif max_subspace < 2 * n_block:
        raise ValueError(f"max_subspace (= {max_subspace}) needs to be at "
                         f"least twice as large as n_block (= {n_block}).")
    elif max_subspace < len(guesses):
        raise ValueError(f"max_subspace (= {max_subspace}) cannot be smaller "
                         f"than the number of guess vectors "
                         f"(= {len(guesses)}).")

    def convergence_test(state):
        state.residuals_converged = state.residual_norms < conv_tol
        state.converged = np.all(state.residuals_converged)
        return state.converged

    if conv_tol < matrix.shape[1] * np.finfo(float).eps:
        warnings.warn(la.LinAlgWarning(
            "Convergence tolerance (== {:5.2g}) lower than "
            "estimated maximal numerical accuracy (== {:5.2g}). "
            "Convergence might be hard to achieve."
            "".format(conv_tol, matrix.shape[1] * np.finfo(float).eps)
        ))

    state = DecoupledDavidsonState(matrix, guesses)
    decoupled_davidson_iterations(
        matrix, state, max_subspace, max_iter,
        n_ep=n_ep, n_block=n_block, is_converged=convergence_test,
        callback=callback, which=which,
        preconditioner=preconditioner,
        preconditioning_method=preconditioning_method,
        debug_checks=debug_checks,
        residual_min_norm=residual_min_norm,
        explicit_symmetrisation=explicit_symmetrisation)
    return state


def decoupled_jacobi_davidson(*args, **kwargs):
    return decoupled_eigsh(*args, preconditioner=JacobiPreconditioner,
                           preconditioning_method="Davidson", **kwargs)


def decoupled_davidson(*args, **kwargs):
    return decoupled_eigsh(*args, preconditioner=None, **kwargs)
