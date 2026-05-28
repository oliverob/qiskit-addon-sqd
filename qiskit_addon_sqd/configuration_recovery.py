# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Functions for performing self-consistent configuration recovery."""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from qiskit.utils.deprecation import deprecate_func


@deprecate_func(
    since="0.12.0",
    package_name="qiskit-addon-sqd",
    removal_timeline="no earlier than v0.13.0",
    additional_msg=("Instead, use the ``postselect_by_hamming_right_and_left`` function."),
)
def post_select_by_hamming_weight(
    bitstring_matrix: np.ndarray, *, hamming_right: int, hamming_left: int
) -> np.ndarray:
    """Post-select bitstrings based on the hamming weight of each half.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring
        hamming_right: The target hamming weight of the right half of bitstrings
        hamming_left: The target hamming weight of the left half of bitstrings

    Returns:
        A mask signifying which samples (rows) were selected from the input matrix.

    """
    if hamming_left < 0 or hamming_right < 0:
        raise ValueError("Hamming weights must be non-negative integers.")
    num_bits = bitstring_matrix.shape[1]

    # Find the bitstrings with correct hamming weight on both halves
    up_keepers = np.sum(bitstring_matrix[:, num_bits // 2 :], axis=1) == hamming_right
    down_keepers = np.sum(bitstring_matrix[:, : num_bits // 2], axis=1) == hamming_left
    correct_bs_mask = np.array(np.logical_and(up_keepers, down_keepers))

    return correct_bs_mask


def recover_configurations(
    bitstring_matrix: np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    avg_occupancies: tuple[np.ndarray, np.ndarray],
    num_elec_a: int,
    num_elec_b: int,
    rand_seed: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine bitstrings based on average orbital occupancy and a target hamming weight.

    This function refines each bit in isolation in an attempt to transform the Hilbert space
    represented by the input ``bitstring_matrix`` into a space closer to that which supports
    the ground state.

    .. note::

        This function makes the assumption that bit ``i`` represents the spin-down orbital
        corresponding to the spin-up orbital in bit ``i + N`` where ``N`` is the number of
        spatial orbitals and ``i < N``.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring
        probabilities: A 1D array specifying a probability distribution over the bitstrings
        avg_occupancies: A length-2 tuple of arrays holding the mean occupancy of the spin-up
            and spin-down orbitals, respectively. The occupancies should be formatted as:
            ``(array([occ_a_0, ..., occ_a_N]), array([occ_b_0, ..., occ_b_N]))``
        num_elec_a: The number of spin-up electrons in the system.
        num_elec_b: The number of spin-down electrons in the system.
        rand_seed: A seed for controlling randomness

    Returns:
        A refined bitstring matrix and an updated probability array.

    References:
        [1]: J. Robledo-Moreno, et al., `Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer <https://arxiv.org/abs/2405.05068>`_,
             arXiv:2405.05068 [quant-ph].
    """
    rng = np.random.default_rng(rand_seed)

    occ_dims = len(np.array(avg_occupancies).shape)
    if occ_dims == 1:
        warnings.warn(
            "Passing avg_occupancies as a 1D array is deprecated. Pass a length-2 tuple containing the spin-up and spin-down occupancies respectively.",
            DeprecationWarning,
            stacklevel=2,
        )
        norb = bitstring_matrix.shape[1] // 2
        avg_occupancies = (np.flip(avg_occupancies[norb:]), np.flip(avg_occupancies[:norb]))

    if num_elec_a < 0 or num_elec_b < 0:
        raise ValueError("The numbers of electrons must be specified as non-negative integers.")

    corrected_dict: defaultdict[str, float] = defaultdict(float)
    occs_array = np.flip(avg_occupancies).flatten()
    for bitstring, freq in zip(bitstring_matrix, probabilities):
        bs_corrected = _bipartite_bitstring_correcting(
            bitstring,
            occs_array,
            num_elec_a,
            num_elec_b,
            rng=rng,
        )
        bs_str = "".join("1" if bit else "0" for bit in bs_corrected)
        corrected_dict[bs_str] += freq
    bs_mat_out = np.array([[bit == "1" for bit in bs] for bs in corrected_dict])
    freqs_out = np.array([f for f in corrected_dict.values()])
    freqs_out = np.abs(freqs_out) / np.sum(np.abs(freqs_out))

    return bs_mat_out, freqs_out


def _p_flip_0_to_1(ratio_exp: float, occ: float, eps: float = 0.01) -> float:  # pragma: no cover
    """Calculate the probability of flipping a bit from 0 to 1.

    This function will more aggressively flip bits which are in disagreement
    with the occupation information.

    Args:
        ratio_exp: The ratio of 1's expected in the set of bits
        occ: The occupancy of a particular bit, based estimated ground
            state found at the end of each configuration recovery iteration.
        eps: A value for controlling how aggressively to flip bits

    Returns:
        The probability with which to flip the bit

    """
    # Occupancy is < than naive expectation.
    # Flip 0s to 1 with small (<eps) probability in this case
    if occ < ratio_exp:
        return occ * eps / ratio_exp

    # Occupancy is >= naive expectation.
    # The probability weight to flip the bit increases linearly from ``eps`` to
    # ``1.0`` as the occupation deviates further from the expected ratio
    if ratio_exp == 1.0:
        return eps
    slope = (1 - eps) / (1 - ratio_exp)
    intercept = 1 - slope
    return occ * slope + intercept


def _p_flip_1_to_0(ratio_exp: float, occ: float, eps: float = 0.01) -> float:  # pragma: no cover
    """Calculate the probability of flipping a bit from 1 to 0.

    This function will more aggressively flip bits which are in disagreement
    with the occupation information.

    Args:
        ratio_exp: The ratio of 1's expected in the set of bits
        occ: The occupancy of a particular bit, based estimated ground
            state found at the end of each configuration recovery iteration.
        eps: A value for controlling how aggressively to flip bits

    Returns:
        The probability with which to flip the bit

    """
    return _p_flip_0_to_1(1 - ratio_exp, 1 - occ, eps)


def _bipartite_bitstring_correcting(
    bit_array: np.ndarray,
    avg_occupancies: np.ndarray,
    hamming_right: int,
    hamming_left: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Use occupancy information and target hamming weight to correct a bitstring.

    Args:
        bit_array: A 1D array of ``bool`` representations of bit values
        avg_occupancies: A 1D array containing the mean occupancy of each orbital.
        hamming_right: The target hamming weight used for the right half of the bitstring
        hamming_left: The target hamming weight used for the left half of the bitstring
        rng: A random number generator

    Returns:
        A corrected bitstring

    """
    # This function must not mutate the input arrays.
    bit_array = bit_array.copy()

    # The number of bits should be even
    num_bits = bit_array.shape[0]
    partition_size = num_bits // 2

    # Get the probability of flipping each bit, separated into LEFT and RIGHT subsystems,
    # based on the avg occupancy of each bit and the target hamming weight
    probs_left = np.zeros(partition_size)
    probs_right = np.zeros(partition_size)
    for i in range(partition_size):
        if bit_array[i]:
            probs_left[i] = _p_flip_1_to_0(hamming_left / partition_size, avg_occupancies[i], 0.01)
        else:
            probs_left[i] = _p_flip_0_to_1(hamming_left / partition_size, avg_occupancies[i], 0.01)

        if bit_array[i + partition_size]:
            probs_right[i] = _p_flip_1_to_0(
                hamming_right / partition_size, avg_occupancies[i + partition_size], 0.01
            )
        else:
            probs_right[i] = _p_flip_0_to_1(
                hamming_right / partition_size, avg_occupancies[i + partition_size], 0.01
            )
    # Force probabilities to be beween 0 and 1, which can fail due to floating point precision
    probs_left = np.minimum(1, np.maximum(0, probs_left))
    probs_right = np.minimum(1, np.maximum(0, probs_right))

    ######################## Handle LEFT bits ########################
    if np.any(probs_left):
        # Normalize
        probs_left /= np.sum(probs_left)

        # Get difference between # of 1s and expected # of 1s in LEFT bits
        n_left = np.sum(bit_array[:partition_size])
        n_diff = n_left - hamming_left

        # Too many electrons in LEFT bits
        if n_diff > 0:
            indices_occupied = np.where(bit_array[:partition_size])[0]
            # Get the probabilities that each 1 should be flipped to 0
            p_choice = probs_left[bit_array[:partition_size]] / np.sum(
                probs_left[bit_array[:partition_size]]
            )
            # Correct the hamming by probabilistically flipping some bits to flip to 0
            indices_to_flip = rng.choice(
                indices_occupied, size=round(n_diff), replace=False, p=p_choice
            )
            bit_array[:partition_size][indices_to_flip] = False

        # too few electrons in LEFT bits
        if n_diff < 0:
            indices_empty = np.where(np.logical_not(bit_array[:partition_size]))[0]
            # Get the probabilities that each 0 should be flipped to 1
            p_choice = probs_left[np.logical_not(bit_array[:partition_size])] / np.sum(
                probs_left[np.logical_not(bit_array[:partition_size])]
            )
            # Correct the hamming by probabilistically flipping some bits to flip to 1
            indices_to_flip = rng.choice(
                indices_empty, size=round(np.abs(n_diff)), replace=False, p=p_choice
            )
            bit_array[:partition_size][indices_to_flip] = np.logical_not(
                bit_array[:partition_size][indices_to_flip]
            )

    ######################## Handle RIGHT bits ########################
    if np.any(probs_right):
        # Normalize
        probs_right /= np.sum(probs_right)

        # Get difference between # of 1s and expected # of 1s in RIGHT bits
        n_right = np.sum(bit_array[partition_size:])
        n_diff = n_right - hamming_right

        # too many electrons in RIGHT bits
        if n_diff > 0:
            indices_occupied = np.where(bit_array[partition_size:])[0]
            # Get the probabilities that each 1 should be flipped to 0
            p_choice = probs_right[bit_array[partition_size:]] / np.sum(
                probs_right[bit_array[partition_size:]]
            )
            # Correct the hamming by probabilistically flipping some bits to flip to 0
            indices_to_flip = rng.choice(
                indices_occupied, size=round(n_diff), replace=False, p=p_choice
            )
            bit_array[partition_size:][indices_to_flip] = np.logical_not(
                bit_array[partition_size:][indices_to_flip]
            )

        # too few electrons in RIGHT bits
        if n_diff < 0:
            indices_empty = np.where(np.logical_not(bit_array[partition_size:]))[0]
            # Get the probabilities that each 1 should be flipped to 0
            p_choice = probs_right[np.logical_not(bit_array[partition_size:])] / np.sum(
                probs_right[np.logical_not(bit_array[partition_size:])]
            )
            # Correct the hamming by probabilistically flipping some bits to flip to 1
            indices_to_flip = rng.choice(
                indices_empty, size=round(np.abs(n_diff)), replace=False, p=p_choice
            )
            bit_array[partition_size:][indices_to_flip] = np.logical_not(
                bit_array[partition_size:][indices_to_flip]
            )

    return bit_array


def bitstring_to_IJ(bits, Na, Nb, Q):
    """Convert one bitstring into I and J lists of integers. bits is array-like of 0/1."""
    I = []
    J = []
    # alpha electrons
    for a in range(Na):
        block = bits[a*Q : (a+1)*Q]
        I.append(int("".join(str(int(x)) for x in block), 2))
    # beta electrons
    for b in range(Nb):
        block = bits[(Na+b)*Q : (Na+b+1)*Q]
        J.append(int("".join(str(int(x)) for x in block), 2))
    return I, J

def IJ_to_bitstring(I, J, Q):
    """Convert integer lists I and J into a flat bitstring (numpy array of 0/1) with Q qubits per integer."""
    bits = []
    for val in I + J:
        bs = format(int(val), 'b').zfill(Q)
        bits.extend([int(ch) for ch in bs])
    return np.array(bits, dtype=int)

def is_physical_from_IJ(I, J, M):
    """Return (phys_occ_a, phys_occ_b, pauli_a, pauli_b)."""
    # alpha range check: each occupation index must be in [0, M-1]
    phys_occ_a = all(0 <= x < M for x in I) if len(I) > 0 else True
    # beta range check
    phys_occ_b = all(0 <= x < M for x in J) if len(J) > 0 else True
    # alpha Pauli (no duplicates)
    pauli_a = (len(I) == len(set(I)))
    # beta Pauli (no duplicates)
    pauli_b = (len(J) == len(set(J)))
    return phys_occ_a, phys_occ_b, pauli_a, pauli_b


# Recovery helpers
def recover_list(current_list, M, weights, rng=None):
    """
    Recover only INVALID entries in `current_list`:
      - Out-of-range elements (>= M or < 0)
      - Duplicates (Pauli violations)

    Valid elements are kept fixed.
    Invalid elements are replaced by sampling WITHOUT replacement
    from the remaining allowed orbitals, according to `weights`.

    Returns a new repaired list.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = len(current_list)
    if k == 0:
        return []

    # Normalize weights
    w = np.asarray(weights, dtype=float)
    if w.size != M:
        raise ValueError("weights must have length M")
    if np.all(w <= 0):
        w = np.ones_like(w)
    p = w / w.sum()

    # Step 1: Identify valid entries
    valid = []
    invalid = []

    # Count duplicates
    seen = set()
    for val in current_list:
        if 0 <= val < M and val not in seen:
            valid.append(val)
            seen.add(val)
        else:
            invalid.append(val)

    # If no invalid values, return same list
    if len(invalid) == 0:
        return list(current_list)

    # Step 2: Determine allowed replacement orbitals
    valid_set = set(valid)
    allowed = np.array([u for u in range(M) if u not in valid_set])

    # Probability restricted to allowed domain
    p_allowed = p[allowed]
    p_allowed /= p_allowed.sum()

    # Step 3: Resample only invalid positions 
    new_vals = list(
        rng.choice(allowed, size=len(invalid), replace=False, p=p_allowed)
    )

    # Step 4: Construct final repaired list
    # Keep original ordering: replace invalid items in-place
    repaired = []
    invalid_idx = 0
    used = set(valid)

    for val in current_list:
        if val in used:
            repaired.append(val)
            used.remove(val)  # consume once
        else:
            repaired.append(new_vals[invalid_idx])
            invalid_idx += 1

    return repaired


# Occupation estimation
def estimate_occupations_from_samples(Xb, probs, Na, Nb, Q, M):
    """
    Estimate single-particle occupation probabilities (n_a, n_b) from sampled bitstrings Xb with associated probs.
    - Xb: (n_out, (Na+Nb)*Q) array of bitstrings (0/1)
    - probs: length n_out array with probabilities / frequencies that sum (or not). We sum weighted counts.
    """
    n_out = Xb.shape[0]
    n_a = np.zeros(M, dtype=float)
    n_b = np.zeros(M, dtype=float)
    for idx in range(n_out):
        bits = Xb[idx].astype(int)
        p = float(probs[idx])
        I, J = bitstring_to_IJ(bits, Na, Nb, Q)
        for u in I:
            if 0 <= u < M:
                n_a[u] += p
        for u in J:
            if 0 <= u < M:
                n_b[u] += p
    eps = 1e-12
    if n_a.sum() <= 0: n_a += eps
    if n_b.sum() <= 0: n_b += eps
    return n_a, n_b


# Apply recovery to distribution
def apply_recovery_to_distribution(Xb, probs, Na, Nb, Q, M, n_a, n_b, rng=None):
    """
    Try to repair each sampled bitstring to a physical state using recover_bitstring.
    Returns: (Xb_rec, probs_copy, success_flags)
    - Xb_rec has same shape as Xb
    - success_flags is boolean array whether recovered state is physical
    """
    if rng is None:
        rng = np.random.default_rng()
    n_out = Xb.shape[0]
    Xb_rec = np.empty_like(Xb)
    success_flags = []
    for idx in range(n_out):
        bits = Xb[idx].astype(int)
        I, J = bitstring_to_IJ(bits, Na, Nb, Q)
        phys_occ_a, phys_occ_b, pauli_a, pauli_b = is_physical_from_IJ(I, J, M)

        if phys_occ_a and phys_occ_b and pauli_a and pauli_b:
            # already physical
            Xb_rec[idx] = bits
            success_flags.append(True)
            continue

        new_bits, I_new, J_new, ok = recover_bitstring(bits, Na, Nb, Q, M, n_a, n_b, rng=rng)
        Xb_rec[idx] = new_bits
        success_flags.append(ok)

    return Xb_rec, probs.copy(), np.array(success_flags, dtype=bool)


def recover_bitstring(bts, Na, Nb, Q, M, n_a, n_b, rng=None):
    """
    Wrapper that:
      - parses bts -> I,J
      - recovers new I,J using weighted sampling from n_a/n_b
      - returns new bitstring, I_new, J_new, ok_flag
    """
    if rng is None:
        rng = np.random.default_rng()
    I, J = bitstring_to_IJ(bts, Na, Nb, Q)
    I_new = recover_list(I, M, n_a, rng=rng)
    J_new = recover_list(J, M, n_b, rng=rng)
    new_bits = IJ_to_bitstring(I_new, J_new, Q)
    phys_occ_a, phys_occ_b, pauli_a, pauli_b = is_physical_from_IJ(I_new, J_new, M)
    ok = (phys_occ_a and phys_occ_b and pauli_a and pauli_b)
    return new_bits, I_new, J_new, ok