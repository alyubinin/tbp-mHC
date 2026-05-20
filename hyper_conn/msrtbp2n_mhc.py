"""
Canonical spelling wrapper for MSRTBP2N-mHC.

This file exposes the corrected `msrtbp2n_mhc` naming while reusing the existing
implementation that was initially added under the typo `msrtdp2n_mhc`.
"""

from .msrtdp2n_mhc import (
    MSRTDP2N_BETA as MSRTBP2N_BETA,
    MSRTDP2N_EPSILON as MSRTBP2N_EPSILON,
    MSRTDP2N_RHO as MSRTBP2N_RHO,
    MSRTDP2N_MHC as MSRTBP2N_MHC,
    MSRTDP2NHC as MSRTBP2NHC,
    MarginedScaledPowerOfTwoRecursiveTransportBirkhoff as MarginedScaledPowerOfTwoRecursiveTransportBirkhoff,
    get_init_and_expand_reduce_stream_functions,
    margined_scaled_recursive_transport_birkhoff_power2,
)
