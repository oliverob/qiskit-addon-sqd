import argparse
import contextlib
import io
import pickle
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyscf import ao2mo, gto, mcscf, scf, tools
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
    seed: int | None = None,
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

    if verbose:
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
            seed=seed,
            first_quantisation=first_quantisation,
        )
    else:
        # The SQD implementation currently prints quite a lot; keep the sweep plots readable.
        with contextlib.redirect_stdout(io.StringIO()):
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
                seed=seed,
                first_quantisation=first_quantisation,
            )

    history = _energy_history(
        result_history,
        nuclear_repulsion_energy=nuclear_repulsion_energy,
    )
    occupancy = np.sum(result.orbital_occupancies, axis=0)
    return history, occupancy


def _load_first_quantised_samples(
    npy_path: str | Path,
    *,
    num_electrons: int,
    norb: int,
) -> np.ndarray:
    arr = np.load(npy_path)
    num_bits = int(np.ceil(np.log2(norb)))
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1:] == (num_electrons, num_bits):
        out = arr
    else:
        out = arr.reshape(-1, num_electrons, num_bits)
    return out.astype(np.uint8, copy=False)


def _load_second_quantised_samples_bool(
    npy_path: str | Path,
    *,
    norb: int,
) -> np.ndarray:
    arr = np.asarray(np.load(npy_path))
    out = arr.reshape(-1, 2 * norb).astype(bool, copy=False)
    return out


def _generate_uniform_first_quantised_samples(
    rng: np.random.Generator,
    *,
    num_samples: int,
    norb: int,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    num_bits = int(np.ceil(np.log2(norb)))
    n_electrons = n_alpha + n_beta
    if num_samples == 0:
        return np.empty((0, n_electrons, num_bits), dtype=np.uint8)

    first_quantised_samples = np.random.randint(0, 2, size=(num_samples, n_electrons, num_bits), dtype=np.uint8)

    return first_quantised_samples


def _generate_uniform_second_quantised_samples_bool(
    rng: np.random.Generator,
    *,
    num_samples: int,
    norb: int,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    """Generate 2nd-quantized samples with correct (n_beta | n_alpha) Hamming weights.

    Output uses the same little-endian convention as the existing loaded `.npy` samples.
    """
    if num_samples == 0:
        return np.empty((0, 2 * norb), dtype=bool)

    second_quantised_bitstrings = np.random.randint(0, 2, size=(num_samples, 2 * norb), dtype=np.bool)

    return second_quantised_bitstrings


def _mix_samples(
    *,
    rng: np.random.Generator,
    loaded: np.ndarray,
    uniform: np.ndarray,
    alpha: float,
    total_samples: int,
) -> np.ndarray:
    """Mix two sample pools by concatenation.

    `alpha` is the fraction of samples drawn from `loaded`.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")

    n_loaded = int(np.clip(np.round(alpha * total_samples), 0, total_samples))
    n_uniform = total_samples - n_loaded

    if n_loaded:
        loaded_idx = rng.choice(
            loaded.shape[0],
            size=n_loaded,
            replace=n_loaded > loaded.shape[0],
        )
        loaded_sel = loaded[loaded_idx]
    else:
        loaded_sel = loaded[:0]

    if n_uniform:
        uniform_idx = rng.choice(
            uniform.shape[0],
            size=n_uniform,
            replace=n_uniform > uniform.shape[0],
        )
        uniform_sel = uniform[uniform_idx]
    else:
        uniform_sel = uniform[:0]

    mixed = np.concatenate([loaded_sel, uniform_sel], axis=0)
    rng.shuffle(mixed, axis=0)
    return mixed


def _parse_alpha_list(alpha_csv: str | None, *, alpha_min: float, alpha_max: float, alpha_steps: int) -> np.ndarray:
    if alpha_csv is not None:
        parts = [p.strip() for p in alpha_csv.split(",") if p.strip()]
        if not parts:
            raise ValueError("--alphas was provided but no values were parsed")
        alphas = np.asarray([float(p) for p in parts], dtype=float)
    else:
        if alpha_steps < 2:
            raise ValueError("--alpha-steps must be >= 2")
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), alpha_steps, dtype=float)
    if np.any(alphas < 0.0) or np.any(alphas > 1.0):
        raise ValueError("All alpha values must be in [0, 1]")
    return alphas


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run first- and second-quantized SQD multiple times and plot mean±std "
            "energy-error histories (shaded bands)."
        )
    )
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=20_000)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="Fraction of samples drawn from the loaded `.npy` pool (0→all uniform, 1→all loaded).",
    )
    parser.add_argument(
        "--alpha-sweep",
        action="store_true",
        help="Sweep alpha and plot mean energy at iteration 1 and final iteration vs alpha.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated alpha values for --alpha-sweep (overrides --alpha-min/--alpha-max/--alpha-steps).",
    )
    parser.add_argument("--alpha-min", type=float, default=0.000001)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-steps", type=int, default=10)
    parser.add_argument(
        "--first-samples-npy",
        type=str,
        default="./lih_1stresults.npy",
        help="Path to loaded 1st-quantized samples `.npy`.",
    )
    parser.add_argument(
        "--second-samples-npy",
        type=str,
        default="./uccsd_bitstring_samples.npy",
        help="Path to loaded 2nd-quantized samples `.npy`.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    # Build molecule
    # mol = gto.Mole()
    # mol.build(
    #     atom=[["H", (0, 0, 0)], ["H", (1.0, 0, 0)], ["H", (2.0, 0, 0)], ["H", (3.0, 0, 0)]],
    #     basis="6-31g",
    # )
    # scf_result = scf.RHF(mol).run()
    dist=3.0
    basis="6-31g"
    scf_result = tools.fcidump.to_scf('./fci_dump files/' + f'LiH_{dist:.3f}A_{basis}.txt').run()
    
    # dist=1.5
    # basis="6-31g"
    # scf_result = tools.fcidump.to_scf('./fci_dump files/' + f'fci_dump-H4_{dist:.2f}A_{basis}.txt').run()
    mol = scf_result.mol
    # mol.build(
    #     atom=[["Li", (0, 0, 0)], ["H", (1.0, 0, 0)]],
    #     basis="6-31g",
    # )
    # scf_result = scf.RHF(mol).run()

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

    # Load sample pools once; we draw (with replacement if needed) per run.
    loaded_first_pool = _load_first_quantised_samples(
        args.first_samples_npy,
        num_electrons=n_electrons,
        norb=norb,
    )
    loaded_second_pool_bool = _load_second_quantised_samples_bool(args.second_samples_npy, norb=norb)

    if args.alpha_sweep:
        alphas = _parse_alpha_list(
            args.alphas,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_steps=args.alpha_steps,
        )

        first_iter1_mean = np.empty_like(alphas)
        first_iter1_std = np.empty_like(alphas)
        first_final_mean = np.empty_like(alphas)
        first_final_std = np.empty_like(alphas)
        second_iter1_mean = np.empty_like(alphas)
        second_iter1_std = np.empty_like(alphas)
        second_final_mean = np.empty_like(alphas)
        second_final_std = np.empty_like(alphas)

        for a_idx, alpha in enumerate(alphas):
            first_iter1_runs: list[float] = []
            first_final_runs: list[float] = []
            second_iter1_runs: list[float] = []
            second_final_runs: list[float] = []

            for run_idx in range(args.n_runs):
                print(f"Starting run {run_idx + 1}/{args.n_runs} for alpha={alpha:.3f}...")
                run_seed = int(rng.integers(0, 2**32 - 1))

                uniform_first = _generate_uniform_first_quantised_samples(
                    rng,
                    num_samples=args.num_samples,
                    norb=norb,
                    n_alpha=n_alpha,
                    n_beta=n_beta,
                )
                mixed_first = _mix_samples(
                    rng=rng,
                    loaded=loaded_first_pool,
                    uniform=uniform_first,
                    alpha=alpha,
                    total_samples=args.num_samples,
                )

                uniform_second_bool = _generate_uniform_second_quantised_samples_bool(
                    rng,
                    num_samples=args.num_samples,
                    norb=norb,
                    n_alpha=n_alpha,
                    n_beta=n_beta,
                )
                mixed_second_bool = _mix_samples(
                    rng=rng,
                    loaded=loaded_second_pool_bool,
                    uniform=uniform_second_bool,
                    alpha=alpha,
                    total_samples=args.num_samples,
                )
                mixed_second = BitArray.from_bool_array(mixed_second_bool)

                if args.verbose:
                    print(f"alpha={alpha:.3f} run {run_idx + 1}/{args.n_runs}")

                first_hist, _ = _run_single_sqd(
                    hcore=hcore,
                    eri=eri,
                    samples=mixed_first,
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
                    seed=run_seed,
                )
                second_hist, _ = _run_single_sqd(
                    hcore=hcore,
                    eri=eri,
                    samples=mixed_second,
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
                    first_quantisation=False,
                    verbose=args.verbose,
                    seed=run_seed + 1,
                )

                first_iter1_runs.append(float(first_hist[0]))
                first_final_runs.append(float(first_hist[-1]))
                second_iter1_runs.append(float(second_hist[0]))
                second_final_runs.append(float(second_hist[-1]))

            first_iter1_mean[a_idx] = float(np.mean(first_iter1_runs))
            first_iter1_std[a_idx] = float(np.std(first_iter1_runs)/np.sqrt(args.n_runs))
            first_final_mean[a_idx] = float(np.mean(first_final_runs))
            first_final_std[a_idx] = float(np.std(first_final_runs)/np.sqrt(args.n_runs))
            second_iter1_mean[a_idx] = float(np.mean(second_iter1_runs))
            second_iter1_std[a_idx] = float(np.std(second_iter1_runs)/np.sqrt(args.n_runs))
            second_final_mean[a_idx] = float(np.mean(second_final_runs))
            second_final_std[a_idx] = float(np.std(second_final_runs)/np.sqrt(args.n_runs))

        with Path(f"alpha_{alpha:.6f}_sqd_results.pkl").open("wb") as f:
            pickle.dump(
                {
                    "first_iter1_mean": first_iter1_mean,
                    "first_iter1_std": first_iter1_std,
                    "first_final_mean": first_final_mean,
                    "first_final_std": first_final_std,
                    "second_iter1_mean": second_iter1_mean,
                    "second_iter1_std": second_iter1_std,
                    "second_final_mean": second_final_mean,
                    "second_final_std": second_final_std,
                },
                f,
            )

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(alphas, first_iter1_mean, marker="o", label="iteration 1")
        axs[0].fill_between(
            alphas,
            first_iter1_mean - first_iter1_std,
            first_iter1_mean + first_iter1_std,
            alpha=0.25,
            color=axs[0].lines[-1].get_color(),
        )
        axs[0].plot(alphas, first_final_mean, marker="s", label="final iteration")
        axs[0].fill_between(
            alphas,
            first_final_mean - first_final_std,
            first_final_mean + first_final_std,
            alpha=0.25,
            color=axs[0].lines[-1].get_color()
        )
        # axs[0].axhline(reference_energy, color="black", linestyle="--", label="FCI reference")
        axs[0].set_title("First-quantized SQD: mean energy vs alpha")
        axs[0].set_ylabel("Energy (Ha)")
        axs[0].set_xscale("log")
        axs[0].legend()

        axs[1].plot(alphas, second_iter1_mean, marker="o", label="iteration 1")
        axs[1].fill_between(
            alphas,
            second_iter1_mean - second_iter1_std,
            second_iter1_mean + second_iter1_std,
            alpha=0.25,
            color=axs[1].lines[-1].get_color(),
        )
        axs[1].plot(alphas, second_final_mean, marker="s", label="final iteration")
        axs[1].fill_between(
            alphas,
            second_final_mean - second_final_std,
            second_final_mean + second_final_std,
            alpha=0.25,
            color=axs[1].lines[-1].get_color()
        )
        # axs[1].axhline(reference_energy, color="black", linestyle="--", label="FCI reference")
        axs[1].set_title("Second-quantized SQD: mean energy vs alpha")
        axs[1].set_xlabel("alpha (fraction loaded samples)")
        axs[1].set_ylabel("Energy (Ha)")
        axs[1].set_xscale("log")
        axs[1].legend()

        plt.tight_layout()
        plt.show()
        
        return

    for run_idx in range(args.n_runs):
        run_seed = int(rng.integers(0, 2**32 - 1))

        uniform_first = _generate_uniform_first_quantised_samples(
            rng,
            num_samples=args.num_samples,
            norb=norb,
            n_alpha=n_alpha,
            n_beta=n_beta,
        )
        mixed_first = _mix_samples(
            rng=rng,
            loaded=loaded_first_pool,
            uniform=uniform_first,
            alpha=args.alpha,
            total_samples=args.num_samples,
        )

        uniform_second_bool = _generate_uniform_second_quantised_samples_bool(
            rng,
            num_samples=args.num_samples,
            norb=norb,
            n_alpha=n_alpha,
            n_beta=n_beta,
        )
        mixed_second_bool = _mix_samples(
            rng=rng,
            loaded=loaded_second_pool_bool,
            uniform=uniform_second_bool,
            alpha=args.alpha,
            total_samples=args.num_samples,
        )
        mixed_second = BitArray.from_bool_array(mixed_second_bool)

        print(f"Run {run_idx + 1}/{args.n_runs} (alpha={args.alpha:.3f})")

        first_hist, first_o = _run_single_sqd(
            hcore=hcore,
            eri=eri,
            samples=mixed_first,
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
            seed=run_seed,
        )
        second_hist, second_o = _run_single_sqd(
            hcore=hcore,
            eri=eri,
            samples=mixed_second,
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
            seed=run_seed + 1,
        )
        print(first_hist- reference_energy)
        print(second_hist- reference_energy)
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

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
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

        axs[i].plot(x, mean, label="mean energy error")
        axs[i].fill_between(x, lower, upper, alpha=0.25, label="±1σ")
        # axs[i, 0].set_yticks(yt1)
        # axs[i, 0].set_yticklabels(yt1)
        # axs[i, 0].set_yscale("log")
        # axs[i, 0].set_ylim(1e-4)
        axs[i].axhline(
            y=chem_accuracy,
            color="black",
            linestyle="--",
            label="chemical accuracy",
        )
        axs[i].set_title(
            f"LiH 6-31g {label} Quantisation Approximated Ground State Energy Error vs SQD Iterations"
        )
        axs[i].set_xlabel("SQD iteration", fontdict={"fontsize": 12})
        axs[i].set_ylabel("Energy Error (Ha)", fontdict={"fontsize": 12})
        axs[i].legend()

        # x2 = np.arange(occ_mean.shape[0])
        # axs[i, 1].bar(x2, occ_mean, width=0.8)
        # axs[i, 1].set_xticks(x2)
        # axs[i, 1].set_xticklabels(x2)
        # axs[i, 1].set_title(f"{label} Quantisation Avg Occupancy per Spatial Orbital")
        # axs[i, 1].set_xlabel("Orbital Index", fontdict={"fontsize": 12})
        # axs[i, 1].set_ylabel("Avg Occupancy", fontdict={"fontsize": 12})

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()