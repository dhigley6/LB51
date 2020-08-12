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
import copy

from xbloch import phot_fft_utils

# Physical constants
HBAR = 6.582E-1    # eV*fs
HBAR_EVOLVE = 1.055E-19   # J*fs
EPSILON_0 = 8.854188E-12    # C/(Vm)
C = 2.99792E8*1E-15    #(m/fs) (speed of light)
DENSITY = 90.9*1E27      # (atoms/m^3)  (calculated from CRC handbook of chemistry and physics),
# also in stohr/scherz prl

# Model material parameters:
DIPOLE = (4.07E-12+0j)*1.602E-19    # Dipole matrix element to use   (I think this is in C*m, but need to check)
DIPOLE = (8.03E-12+0j)*1.602E-19
Co_L3_BROAD = 0.43    # Natural lifetime of Co 2p_{3/2} core hole
# Thickness of sample:
#THICKNESS = 1E-9    # thickness of sample in m
THICKNESS = 5E-11
# We simulate propagation through a sample which is thin with respect to an absorption length
# so that the first born approximation approximately holds.

def make_model_system():
    """Make model system for 3-level simulations at Co L3 resonance of Co metal
    """
    model_system = LambdaBloch(0, 778, 2, DIPOLE, 1*np.sqrt(3)*DIPOLE, 778/HBAR)
    return model_system

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
    s: array of floats
        Rotating reference frame density matrix
    gamma: floats
        Phenomenological relaxation tensor as described in Bruce Shroe's book
    t: float
        Current time of simulation (fs)
    polarization: float
        Complex envelope of current polarization density (C/m^2)
    history: dictionary
        Contains previous values of t, field_envelope, s, polarization_envelope
    W: array of floats
        Matrix that gives coherent evolution of density matrix, as described by Shore
    """

    def __init__(self, E1, E2, E3, d12, d23, omega, gamma_a=0.43):
        self.omega = omega
        self._detunings = [
            0,
            (E2-E1-HBAR*omega)/HBAR,
            (E2-E3-HBAR*omega)/HBAR
        ]
        self.d12 = d12
        self.d23 = d23
        self.s = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=complex)
        self.gamma = np.zeros((3, 3, 3, 3))
        self.gamma[0, 0, 1, 1] = -1*gamma_a
        self.gamma[1, 1, 1, 1] = gamma_a
        self.gamma[0, 1, 0, 1] = gamma_a/2
        self.gamma[1, 0, 1, 0] = gamma_a/2
        self.gamma[1, 2, 1, 2] = gamma_a/2
        self.gamma[2, 1, 2, 1] = gamma_a/2
        self.t = 0
        self.polarization = 0
        self.history = {
            't': [],
            'field_envelope': [],
            's': [],
            'polarization_envelope': []
        }

    def run_simulation(self, times, field_envelope_list):
        """Run simulation with input incident electric field
        
        Parameters
        ----------
        times: list of floats
            Time points of the simulation (fs)
        field_envelope_list: list of complex floats
            Electric field strengths (V/m) 
        """
        delta_ts = np.diff(times)
        for ind_minus_1, delta_t in enumerate(delta_ts):
            self._evolve_step(delta_t, field_envelope_list[ind_minus_1])
        self.density_panda_ = self._density_to_panda()
        self.phot_results_ = self._get_phot_result()
        return self

    def _evolve_step(self, delta_t, field_envelope):
        self._update_history(field_envelope)
        self._update_W(field_envelope)
        num_states = self.s.shape[0]
        ds = np.zeros_like(self.s, dtype=complex)
        for row in range(num_states):
            for col in range(num_states):
                ds_coh = 0
                for n in range(num_states):
                    ds_coh += delta_t*-1j*(self.W[row, n]*self.s[n, col]-self.s[row, n]*self.W[n, col])
                ds_inc = 0
                for n in range(num_states):
                    for n_prime in range(num_states):
                        ds_inc += -1*delta_t*self.gamma[row, col, n, n_prime]*self.s[n, n_prime]
                ds[row, col] = ds_coh+ds_inc
        self.s = self.s+ds
        self.t += delta_t


    def _update_W(self, field_envelope):
        """Update W matrix that determines coherent density matrix evolution

        Parameters
        ----------
        field_envelope: complex float
           Envelope of electric field at current time (V/m)
        """
        self._rabi12 = -1*self.d12*field_envelope/HBAR_EVOLVE
        self._rabi23 = -1*self.d23*field_envelope/HBAR_EVOLVE
        self.W = np.array([
            [self._detunings[0], 0.5*np.conj(self._rabi12), 0],
            [0.5*self._rabi12, self._detunings[1], 0.5*self._rabi23],
            [0, 0.5*np.conj(self._rabi23), self._detunings[2]]
        ])

    def _update_history(self, field_envelope):
        self.history['t'].append(self.t)
        self.history['field_envelope'].append(field_envelope)
        self.history['s'].append(self.s)
        self.history['polarization_envelope'].append(self._calculate_polarization_envelope())

    def _calculate_polarization_envelope(self):
        d32 = np.conj(self.d23)
        polarization = 2*DENSITY*(self.d12*self.s[1, 0]+d32*self.s[1, 2])
        return polarization

    def _density_to_panda(self):
        """Return history of density matrix elements as pandas data frame
        """
        rho_dict = {}
        num_states = self.s.shape[0]
        for row in range(num_states):
            for col in range(num_states):
                # the below condition is to not have redundancy in the 
                # stored dataframe as it is hermitian
                if row >= col:
                    rho_str = f'rho({row}, {col})'
                    rho_dict[rho_str] = [np.abs(i[row, col]) for i in self.history['s']]
        rho_panda = pd.DataFrame(rho_dict, index=self.history['t'])
        return rho_panda

    def _get_phot_result(self):
        """Return incident and output spectra

        Returns
        -------
        phot_results: dictionary
            phots: array
                photon energies (eV)
            E_in: array
                incident electric field spectral amplitude
            polarization: array
                material polarization spectrum
            E_delta: array
                change of electric field spectral amplitude after propagation through sample
            E_out: array
                
        """
        E_in = {
            'phot': phot_fft_utils.convert_times_to_phots(self.history['t'], HBAR*self.omega),
            'phot_y': phot_fft_utils.convert_time_signal_to_phot_signal(self.history['field_envelope'])
        }
        phot_polarization = {
            'phot': phot_fft_utils.convert_times_to_phots(self.history['t'], HBAR*self.omega),
            'phot_y': phot_fft_utils.convert_time_signal_to_phot_signal(self.history['polarization_envelope'])
        }
        E_delta = 1j*THICKNESS*phot_polarization['phot_y']*self.omega/(2*C*EPSILON_0)
        E_out = E_in['phot_y']+E_delta
        phot_results = {
            'phots': phot_polarization['phot'],
            'E_in': E_in['phot_y'],
            'polarization': phot_polarization,
            'E_delta': E_delta,
            'E_out': E_out
        }
        return phot_results        