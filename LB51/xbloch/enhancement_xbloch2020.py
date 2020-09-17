"""Ad hoc incorporation of s&s enhancement factor into xbloch simulations
"""

import numpy as np 

from LB51.xbloch import xbloch2020

class EnhancementLambdaBloch(xbloch2020.LambdaBloch):
    """LambdaBloch with s&s enhancement factor of absorption reduction
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
        xbloch2020.LambdaBloch.__init__(
            self, E1, E2, E3, d12, d23, omega, gamma_a
        )
        self.linear_reduction = 1E-5    # factor by which to reduce strength
        # of electric field applied to estimate linear response
        self.linear_instance = xbloch2020.LambdaBloch(
            E1, E2, E3, d12, d23, omega, gamma_a
        )

    def _calculate_polarization_envelope(self) -> complex:
        """Calculate enhancement-modifed polarization envelope for current time
        """
        d32 = np.conj(self.d23)
        s_10 = self.s[1, 0]
        s_10_linear = self.linear_instance.s[1, 0]/self.linear_reduction
        enhancement_factor = self._get_enhancement_factor(50)
        enhanced_s_10 = (s_10-s_10_linear)*enhancement_factor+s_10_linear
        polarization = 2 * xbloch2020.DENSITY * (self.d12 * enhanced_s_10 + d32 * self.s[1, 2])
        return polarization

    def _get_enhancement_factor(self, intensity: float) -> float:
        return 5.0

    def run_simulation(self, times: np.ndarray, field_envelope_list: np.ndarray) -> xbloch2020.LambdaBloch:
        """Run simulation with input incident electric field
        """
        delta_ts = np.diff(times)
        for ind_minus_1, delta_t in enumerate(delta_ts):
            self._evolve_step(delta_t, field_envelope_list[ind_minus_1])
            self.linear_instance._evolve_step(delta_t, self.linear_reduction*field_envelope_list[ind_minus_1])
        self.density_panda_ = self._density_to_panda()
        self.phot_results_ = self._get_phot_result()
        return self

def make_model_system() -> xbloch2020.LambdaBloch:
    """Make model system for 3-level simulations at Co L3 resonance of Co metal

    Returns:
    --------
        model_system: LambdaBloch
            The model three-level system
    """
    model_system = EnhancementLambdaBloch(0, 778, 2, xbloch2020.DIPOLE, 1 * np.sqrt(3) * xbloch2020.DIPOLE, 778 / xbloch2020.HBAR)
    return model_system
