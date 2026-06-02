import argparse
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from qiskit.primitives import BitArray

from .fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch


def _energy_history(
    result_history: list[list[SCIResult]],
    *,
    nuclear_repulsion_energy: float,
) -> np.ndarray:
    min_e = [
        min(iteration_results, key=lambda res: res.energy).energy
        + nuclear_repulsion_energy
        for iteration_results in result_history
    ]
    return np.asarray([e for e in min_e], dtype=float)


def _pad_ragged(histories: list[np.ndarray], pad_value: float = np.nan) -> np.ndarray:
    if not histories:
        return np.empty((0, 0), dtype=float)
    max_len = max((h.shape[0] for h in histories), default=0)
    out = np.full((len(histories), max_len), pad_value, dtype=float)
    for i, h in enumerate(histories):
        out[i, : h.shape[0]] = h
    return out


def _run_single_sqd(
    *,
    hcore: np.ndarray,
    eri: np.ndarray,
    samples: np.ndarray | BitArray,
    nuclear_repulsion_energy: float,
    reference_energy: float,
    norb: int,
    nelec: tuple[int, int],
    energy_tol: float,
    occupancies_tol: float,
    max_iterations: int,
    num_batches: int,
    samples_per_batch: int,
    symmetrize_spin: bool,
    carryover_threshold: int,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    sci_solver,
    first_quantisation: bool,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    result_history: list[list[SCIResult]] = []

    def _callback(results: list[SCIResult]):
        result_history.append(results)
        if not verbose:
            return
        iteration = len(result_history)
        print(f"Iteration {iteration}")
        for i, result in enumerate(results):
            print(f"\tSubsample {i}")
            print(f"\t\tEnergy: {result.energy + nuclear_repulsion_energy}")
            print(
                f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}"
            )

    result = diagonalize_fermionic_hamiltonian(
        hcore,
        eri,
        samples,
        samples_per_batch=samples_per_batch,
        norb=norb,
        nelec=nelec,
        num_batches=num_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        sci_solver=sci_solver,
        symmetrize_spin=symmetrize_spin,
        carryover_threshold=carryover_threshold,
        initial_occupancies=initial_occupancies,
        callback=_callback,
        first_quantisation=first_quantisation,
    )

    history = _energy_history(
        result_history,
        nuclear_repulsion_energy=nuclear_repulsion_energy,
    )
    occupancy = np.sum(result.orbital_occupancies, axis=0)
    return history, occupancy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run first- and second-quantized SQD multiple times and plot mean±std "
            "energy-error histories (shaded bands)."
        )
    )
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=20_000)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    # Build molecule
    mol = gto.Mole()
    # mol.build(
    #     atom=[["H", (0, 0, 0)], ["H", (1.0, 0, 0)], ["H", (2.0, 0, 0)], ["H", (3.0, 0, 0)]],
    #     basis="6-31g",
    # )
    mol.build(
        atom=[["Li", (0, 0, 0)], ["H", (1.0, 0, 0)]],
        basis="6-31g",
    )
    scf_result = scf.RHF(mol).run()

    # Define active space
    n_frozen = 0
    active_space = range(n_frozen, mol.nao)

    # Get molecular integrals
    norb = len(active_space)
    n_electrons = int(sum(scf_result.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)
    cas = mcscf.CASCI(scf_result, norb, nelec)
    mo = cas.sort_mo(active_space, base=0)
    hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    eri = ao2mo.restore(1, cas.get_h2cas(mo), norb)

    # Compute exact energy using FCI
    reference_energy = cas.run().e_tot

    print(f"norb = {norb}")
    print(f"nelec = {nelec}")

    # SQD options
    energy_tol = 1e-3
    occupancies_tol = 1e-3

    # Eigenstate solver options
    num_batches = 3
    samples_per_batch = 10
    symmetrize_spin = True
    carryover_threshold = 1
    max_cycle = 200

    sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)

    first_histories: list[np.ndarray] = []
    second_histories: list[np.ndarray] = []
    first_occ: list[np.ndarray] = []
    second_occ: list[np.ndarray] = []

    n_alpha, n_beta = nelec

    for run_idx in range(args.n_runs):
        first_quantised_samples = np.random.randint(0, 2, size=(args.num_samples, n_electrons, int(np.ceil(np.log2(norb)))), dtype=np.uint8)
        second_quantised_bitstrings = BitArray.from_bool_array(np.random.randint(0, 2, size=(args.num_samples, 2 * norb), dtype=np.bool))

        print(f"Run {run_idx + 1}/{args.n_runs}")

        first_hist, first_o = _run_single_sqd(
            hcore=hcore,
            eri=eri,
            samples=first_quantised_samples,
            nuclear_repulsion_energy=nuclear_repulsion_energy,
            reference_energy=reference_energy,
            norb=norb,
            nelec=nelec,
            energy_tol=energy_tol,
            occupancies_tol=occupancies_tol,
            max_iterations=args.max_iterations,
            num_batches=num_batches,
            samples_per_batch=samples_per_batch,
            symmetrize_spin=symmetrize_spin,
            carryover_threshold=carryover_threshold,
            sci_solver=sci_solver,
            first_quantisation=True,
            verbose=args.verbose,
        )
        initial_occupancies = (
            np.array([1] * n_alpha + [0] * (norb - n_alpha)),
            np.array([1] * n_beta + [0] * (norb - n_beta)),
        )
        second_hist, second_o = _run_single_sqd(
            hcore=hcore,
            eri=eri,
            samples=second_quantised_bitstrings,
            nuclear_repulsion_energy=nuclear_repulsion_energy,
            reference_energy=reference_energy,
            norb=norb,
            nelec=nelec,
            energy_tol=energy_tol,
            occupancies_tol=occupancies_tol,
            max_iterations=args.max_iterations,
            num_batches=num_batches,
            samples_per_batch=samples_per_batch,
            symmetrize_spin=symmetrize_spin,
            carryover_threshold=carryover_threshold,
            # initial_occupancies=initial_occupancies,
            sci_solver=sci_solver,
            first_quantisation=False,
            verbose=args.verbose,
        )

        first_histories.append(first_hist)
        second_histories.append(second_hist)
        first_occ.append(first_o)
        second_occ.append(second_o)

    # Aggregate histories (ragged → padded with NaNs)
    first_mat = _pad_ragged(first_histories)
    second_mat = _pad_ragged(second_histories)

    first_mean = np.abs(np.nanmean(first_mat, axis=0) - reference_energy)
    first_std = np.nanstd(first_mat, axis=0)
    second_mean = np.abs(np.nanmean(second_mat, axis=0) - reference_energy)
    second_std = np.nanstd(second_mat, axis=0)

    # Aggregate occupancies
    first_occ_mean = np.mean(np.stack(first_occ, axis=0), axis=0)
    second_occ_mean = np.mean(np.stack(second_occ, axis=0), axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    # yt1 = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]

    # Chemical accuracy (+/- 1 milli-Hartree)
    chem_accuracy = 0.001

    for i, (label, mean, std, occ_mean) in enumerate(
        [
            ("First", first_mean, first_std, first_occ_mean),
            ("Second", second_mean, second_std, second_occ_mean),
        ]
    ):
        x = np.arange(1, mean.shape[0] + 1)
        lower = np.maximum(mean - std, 0)
        upper = mean + std

        axs[i, 0].plot(x, mean, label="mean energy error")
        axs[i, 0].fill_between(x, lower, upper, alpha=0.25, label="±1σ")
        # axs[i, 0].set_yticks(yt1)
        # axs[i, 0].set_yticklabels(yt1)
        # axs[i, 0].set_yscale("log")
        # axs[i, 0].set_ylim(1e-4)
        axs[i, 0].axhline(
            y=chem_accuracy,
            color="black",
            linestyle="--",
            label="chemical accuracy",
        )
        axs[i, 0].set_title(
            f"{label} Quantisation Approximated Ground State Energy Error vs SQD Iterations"
        )
        axs[i, 0].set_xlabel("SQD iteration", fontdict={"fontsize": 12})
        axs[i, 0].set_ylabel("Energy Error (Ha)", fontdict={"fontsize": 12})
        axs[i, 0].legend()

        x2 = np.arange(occ_mean.shape[0])
        axs[i, 1].bar(x2, occ_mean, width=0.8)
        axs[i, 1].set_xticks(x2)
        axs[i, 1].set_xticklabels(x2)
        axs[i, 1].set_title(f"{label} Quantisation Avg Occupancy per Spatial Orbital")
        axs[i, 1].set_xlabel("Orbital Index", fontdict={"fontsize": 12})
        axs[i, 1].set_ylabel("Avg Occupancy", fontdict={"fontsize": 12})

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()