#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dhigley

Code for keeping track of the dynamics of an X-ray pulse resonantly propagating
through a sample. The dynamics are calculated with the Maxwell-Bloch
equations, as described in Bruce Shore's book.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from LB51.xbloch import phot_fft_utils

# Physical constants
HBAR = 6.582e-1  # eV*fs
HBAR_EVOLVE = 1.055e-19  # J*fs
EPSILON_0 = 8.854188e-12  # C/(Vm)
C = 2.99792e8 * 1e-15  # (m/fs) (speed of light)
DENSITY = (
    90.9 * 1e27
)  # (atoms/m^3)  (calculated from CRC handbook of chemistry and physics),
# also in stohr/scherz prl

# Model material parameters:
DIPOLE = (
    4.04e-12 + 0j
) * 1.602e-19  # Dipole matrix element to use   (C*m)
Co_L3_BROAD = 0.43  # Natural lifetime of Co 2p_{3/2} core hole
# Thickness of sample:
# THICKNESS = 1E-9    # thickness of sample in m
THICKNESS = 5e-11
# We simulate propagation through a sample which is thin with respect to an absorption length
# so that the first born approximation approximately holds.


class LambdaBloch:
    """Calculate evolution of density matrix elements for Lambda coupling

    Parameters
    ----------
    E1: float
        Ground state energy (eV)
    E2: float
        Core-excited state energy (eV)
    E3: float
        Valence-excited state energy (eV)
    d12: float
        Dipole coupling between ground and core-excited states (C*m)
    d23: float
        Dipole coupling between core-excited and valence-excited states (C*m)
    omega: float
        Angular frequency of carrier wave of X-ray electric field (rad/s)
    gamma_a: float
        Spectral broadening due to Auger lifetime (eV)

    Attributes
    ----------
    omega: float
        Angular frequency of carrier wave of X-ray electric field (rad/s)
    d12: float
        Dipole coupling between ground and core-excited states (C*m)
    d23: float
        Dipole coupling between core-excited and valence-excited states (C*m)
    s: 2d np.ndarray (complex)
        Rotating reference frame density matrix
    gamma: 2d np.ndarray
        Phenomenological relaxation tensor as described in Bruce Shroe's book
    t: float
        Current time of simulation (fs)
    polarization: float
        Complex envelope of current polarization density (C/m^2)
    history: dictionary
        Contains previous values of t, field_envelope, s, polarization_envelope
    W: 2d np.ndarray (complex)
        Matrix that gives coherent evolution of density matrix, as described by Shore
    """

    def __init__(
        self,
        E1: float,
        E2: float,
        E3: float,
        d12: float,
        d23: float,
        omega: float,
        gamma_a: float = 0.43,
    ):
        self.omega = omega
        self._detunings = [
            0,
            (E2 - E1 - HBAR * omega) / HBAR,
            (E2 - E3 - HBAR * omega) / HBAR,
        ]
        self.d12 = d12
        self.d23 = d23
        self.s = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
        self.gamma = np.zeros((3, 3, 3, 3))
        self.gamma[0, 0, 1, 1] = -1 * gamma_a
        self.gamma[1, 1, 1, 1] = gamma_a
        self.gamma[0, 1, 0, 1] = gamma_a / 2
        self.gamma[1, 0, 1, 0] = gamma_a / 2
        self.gamma[1, 2, 1, 2] = gamma_a / 2
        self.gamma[2, 1, 2, 1] = gamma_a / 2
        self.t = 0
        self.polarization = 0
        self.history = {
            "t": [],
            "field_envelope": [],
            "s": [],
            "polarization_envelope": [],
        }

    def run_simulation(self, times: np.ndarray, field_envelope_list: np.ndarray) -> "LambdaBloch":
        """Run simulation with input incident electric field
        
        Parameters
        ----------
        times: 1d np.ndarray
            Time points of the simulation (fs)
        field_envelope_list: 1d np.ndarray (complex)
            Electric field strengths (V/m)

        Returns:
        --------
        self: LambdaBloch
            self
        """
        delta_ts = np.diff(times)
        for ind_minus_1, delta_t in enumerate(delta_ts):
            self._evolve_step(delta_t, field_envelope_list[ind_minus_1])
        self.density_panda_ = self._density_to_panda()
        self.phot_results_ = self._get_phot_result()
        return self

    def _evolve_step(self, delta_t: float, field_envelope: complex):
        """Evolve system by delta_t

        Parameters:
        -----------
        delta_t: float
            time step to evolve by (fs)
        field_envelope: complex
            complex envelope of field at current time step (V/m)
        """
        self._update_history(field_envelope)
        self._update_W(field_envelope)
        num_states = self.s.shape[0]
        ds = np.zeros_like(self.s, dtype=complex)
        for row in range(num_states):
            for col in range(num_states):
                ds_coh = 0
                for n in range(num_states):
                    ds_coh += (
                        delta_t
                        * -1j
                        * (
                            self.W[row, n] * self.s[n, col]
                            - self.s[row, n] * self.W[n, col]
                        )
                    )
                ds_inc = 0
                for n in range(num_states):
                    for n_prime in range(num_states):
                        ds_inc += (
                            -1
                            * delta_t
                            * self.gamma[row, col, n, n_prime]
                            * self.s[n, n_prime]
                        )
                ds[row, col] = ds_coh + ds_inc
        self.s = self.s + ds
        self.t += delta_t

    def _update_W(self, field_envelope: float):
        """Update W matrix that determines coherent density matrix evolution

        Parameters
        ----------
        field_envelope: complex
           Envelope of electric field at current time (V/m)
        """
        self._rabi12 = -1 * self.d12 * field_envelope / HBAR_EVOLVE
        self._rabi23 = -1 * self.d23 * field_envelope / HBAR_EVOLVE
        self.W = np.array(
            [
                [self._detunings[0], 0.5 * np.conj(self._rabi12), 0],
                [0.5 * self._rabi12, self._detunings[1], 0.5 * self._rabi23],
                [0, 0.5 * np.conj(self._rabi23), self._detunings[2]],
            ]
        )

    def _update_history(self, field_envelope: complex):
        """Update history with values at current time

        Parameters:
        -----------
        field_envelope: complex
            Envelope of electric field at current time (V/m)
        """
        self.history["t"].append(self.t)
        self.history["field_envelope"].append(field_envelope)
        self.history["s"].append(self.s)
        self.history["polarization_envelope"].append(
            self._calculate_polarization_envelope()
        )

    def _calculate_polarization_envelope(self) -> complex:
        """Calculate polarization envelope for current time

        Returns:
        --------
        polarization: complex
            Complex envelope of polarization
        """
        d32 = np.conj(self.d23)
        polarization = 2 * DENSITY * (self.d12 * self.s[1, 0] + d32 * self.s[1, 2])
        return polarization

    def _density_to_panda(self) -> pd.DataFrame:
        """Return history of density matrix elements as pandas data frame

        Returns:
        --------
        rho_panda: pd.DataFrame
            Density matrix elements vs time
        """
        rho_dict = {}
        num_states = self.s.shape[0]
        for row in range(num_states):
            for col in range(num_states):
                # the below condition is to not have redundancy in the
                # stored dataframe as it is hermitian
                if row >= col:
                    rho_str = f"rho({row}, {col})"
                    rho_dict[rho_str] = [np.abs(i[row, col]) for i in self.history["s"]]
        rho_panda = pd.DataFrame(rho_dict, index=self.history["t"])
        return rho_panda

    def _get_phot_result(self) -> Dict[str, np.ndarray]:
        """Return incident and output spectra

        Returns
        -------
        phot_results: dictionary
            phots: 1d np.ndarray
                photon energies (eV)
            E_in: 1d np.ndarray
                incident electric field spectral amplitude
            polarization: 1d np.ndarray
                material polarization spectrum
            E_delta: 1d np.ndarray
                change of electric field spectral amplitude after propagation through sample
            E_out: 1d np.ndarray
                output electric field
        """
        E_in = {
            "phot": phot_fft_utils.convert_times_to_phots(
                self.history["t"], HBAR * self.omega
            ),
            "phot_y": phot_fft_utils.convert_time_signal_to_phot_signal(
                self.history["field_envelope"]
            ),
        }
        phot_polarization = {
            "phot": phot_fft_utils.convert_times_to_phots(
                self.history["t"], HBAR * self.omega
            ),
            "phot_y": phot_fft_utils.convert_time_signal_to_phot_signal(
                self.history["polarization_envelope"]
            ),
        }
        E_delta = (
            1j
            * THICKNESS
            * phot_polarization["phot_y"]
            * self.omega
            / (2 * C * EPSILON_0)
        )
        E_out = E_in["phot_y"] + E_delta
        phot_results = {
            "phots": phot_polarization["phot"],
            "E_in": E_in["phot_y"],
            "polarization": phot_polarization,
            "E_delta": E_delta,
            "E_out": E_out,
        }
        return phot_results


def make_model_system() -> LambdaBloch:
    """Make model system for 3-level simulations at Co L3 resonance of Co metal

    Returns:
    --------
        model_system: LambdaBloch
            The model three-level system
    """
    model_system = LambdaBloch(0, 778, 2, DIPOLE, 1 * np.sqrt(3) * DIPOLE, 778 / HBAR)
    return model_system
