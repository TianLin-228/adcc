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
import adcc
import unittest
import pytest

from adcc import LazyMp
from adcc.solver.davidson import jacobi_davidson
from adcc.solver.decoupled_davidson import (
    decoupled_jacobi_davidson, decoupled_eigsh
)
from adcc.misc import cached_property

from ..testdata_cache import testdata_cache


class TestSolverDecoupledDavidson(unittest.TestCase):
    @cached_property
    def matrix(self):
        return adcc.AdcMatrix(
            "adc2", LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        )

    def test_n_guesses(self):
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=1, block="ph")
        with pytest.raises(ValueError):
            decoupled_eigsh(self.matrix, guesses, n_ep=2)
        res = decoupled_eigsh(self.matrix, guesses, n_ep=1, max_iter=1)
        assert len(res.eigenvalues) == 1

    def test_n_block(self):
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=3, block="ph")
        with pytest.raises(ValueError):
            decoupled_eigsh(self.matrix, guesses, n_ep=2, n_block=1)
        with pytest.raises(ValueError):
            decoupled_eigsh(self.matrix, guesses, n_ep=2, n_block=4)

    def test_max_subspace(self):
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=3, block="ph")
        with pytest.raises(ValueError):
            decoupled_eigsh(self.matrix, guesses, n_ep=1, n_block=2,
                            max_subspace=3)
        with pytest.raises(ValueError):
            decoupled_eigsh(self.matrix, guesses, n_ep=1, n_block=1,
                            max_subspace=2)

    def test_adc1_raises(self):
        """ADC(1) has no doubles block, decoupled Davidson should refuse."""
        matrix_adc1 = adcc.AdcMatrix(
            "adc1", LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        )
        guesses = adcc.guesses_singlet(matrix_adc1, n_guesses=2, block="ph")
        with pytest.raises(ValueError, match="singles.*doubles"):
            decoupled_eigsh(matrix_adc1, guesses, n_ep=2)

    def test_adc2_singlets(self):
        """Core test: decoupled Davidson matches reference ADC(2) singlets."""
        refdata = testdata_cache.adcman_data(
            system="h2o_sto3g", method="adc2", case="gen"
        )["singlet"]

        guesses = adcc.guesses_singlet(self.matrix, n_guesses=9, block="ph")
        res = decoupled_jacobi_davidson(self.matrix, guesses, n_ep=9)

        ref_singlets = refdata["eigenvalues"]
        n_states = min(len(ref_singlets), len(res.eigenvalues))
        assert n_states > 1
        assert res.converged
        assert res.eigenvalues[:n_states] == pytest.approx(
            ref_singlets[:n_states])

    def test_adc2_triplets(self):
        """Decoupled Davidson matches reference ADC(2) triplets."""
        refdata = testdata_cache.adcman_data(
            system="h2o_sto3g", method="adc2", case="gen"
        )["triplet"]

        guesses = adcc.guesses_triplet(self.matrix, n_guesses=10, block="ph")
        res = decoupled_jacobi_davidson(self.matrix, guesses, n_ep=10)

        ref_triplets = refdata["eigenvalues"]
        n_states = min(len(ref_triplets), len(res.eigenvalues))
        assert n_states > 1
        assert res.converged
        assert res.eigenvalues[:n_states] == pytest.approx(
            ref_triplets[:n_states])

    def test_matches_standard_davidson(self):
        """Decoupled Davidson gives identical eigenvalues to standard."""
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=5, block="ph")

        res_std = jacobi_davidson(self.matrix, guesses, n_ep=5)
        res_dec = decoupled_jacobi_davidson(self.matrix, guesses, n_ep=5)

        assert res_std.converged
        assert res_dec.converged
        assert res_dec.eigenvalues == pytest.approx(
            res_std.eigenvalues, abs=1e-8)

    def test_fewer_iterations_or_applies(self):
        """Decoupled Davidson should converge in no more iterations than
        standard Davidson (and typically fewer)."""
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=5, block="ph")

        res_std = jacobi_davidson(self.matrix, guesses, n_ep=5)
        res_dec = decoupled_jacobi_davidson(self.matrix, guesses, n_ep=5)

        assert res_std.converged
        assert res_dec.converged
        # The decoupled solver should need no more iterations
        assert res_dec.n_iter <= res_std.n_iter

    def test_eigenvectors_residuals(self):
        """Verify eigenvectors satisfy the eigenvalue equation."""
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=3, block="ph")
        res = decoupled_jacobi_davidson(self.matrix, guesses, n_ep=3)
        assert res.converged

        for i, evec in enumerate(res.eigenvectors):
            Mv = self.matrix @ evec
            lam_v = res.eigenvalues[i] * evec
            residual = Mv - lam_v
            rnorm = float(residual @ residual) ** 0.5
            assert rnorm < 1e-7

    def test_subspace_sizes_format(self):
        """Verify the state tracks separate subspace sizes."""
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=3, block="ph")
        res = decoupled_jacobi_davidson(self.matrix, guesses, n_ep=3,
                                        max_iter=2)
        assert hasattr(res, "subspace_vectors_s")
        assert hasattr(res, "subspace_vectors_d")
        assert len(res.subspace_vectors_s) >= 0
        assert len(res.subspace_vectors_d) >= 0
        # Combined property should work
        assert (len(res.subspace_vectors)
                == len(res.subspace_vectors_s) + len(res.subspace_vectors_d))

    def test_with_doubles_guesses(self):
        """Test with mixed singles + doubles guesses."""
        guesses_s = adcc.guesses_singlet(self.matrix, n_guesses=3, block="ph")
        guesses_d = adcc.guesses_singlet(self.matrix, n_guesses=2,
                                          block="pphh")
        guesses = guesses_s + guesses_d

        res = decoupled_jacobi_davidson(self.matrix, guesses, n_ep=3)
        assert res.converged

        # Also check against standard Davidson with same guesses
        res_std = jacobi_davidson(self.matrix, guesses, n_ep=3)
        assert res_std.converged
        assert res.eigenvalues == pytest.approx(
            res_std.eigenvalues, abs=1e-8)

    def test_workflow_integration(self):
        """Test using eigensolver='decoupled_davidson' via diagonalise_adcmatrix."""
        from adcc.workflow import diagonalise_adcmatrix
        res = diagonalise_adcmatrix(
            self.matrix, n_states=3, kind="singlet",
            eigensolver="decoupled_davidson"
        )
        assert res.converged
        assert len(res.eigenvalues) == 3
