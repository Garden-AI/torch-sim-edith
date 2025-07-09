def run_relaxation(atoms_dict):
    import ase
    import numpy as np
    import torch
    import torch_sim as ts
    from mace.calculators.foundations_models import mace_mp
    from torch_sim.models.mace import MaceModel, MaceUrls

    # Reconstruct atoms from dict
    atoms = ase.Atoms.fromdict(atoms_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    loaded_model = mace_mp(
        model=MaceUrls.mace_mpa_medium,
        return_raw_model=True,
        default_dtype=dtype,
        device = device,
    )

    model = MaceModel(
        model=loaded_model,
        device=device,
        compute_forces=True,
        compute_stress=True,
        dtype=dtype,
    )

    final_state = ts.optimize(
        system=atoms,
        model=model,
        optimizer=ts.unit_cell_fire,
    )

    result_atoms = final_state.to_atoms()
    # Convert result back to dict for serialization
    result_dicts = [atoms.todict() for atoms in result_atoms]
    # Convert numpy arrays to python lists
    for d in result_dicts:
        for k, v in d.items():
            if type(v) == np.ndarray:
                d[k] = v.tolist()
    return result_dicts


def main():
    import ase
    from ase.build import bulk
    from edith_utils import EdithExecutor

    atoms = bulk("Fe", "fcc", a=5.26, cubic=True)

    # Convert atoms to dict for serialization
    atoms_dict = atoms.todict()

    # Convert numpy arrays to lists for serialization
    import numpy as np
    for key, value in atoms_dict.items():
        if isinstance(value, np.ndarray):
            atoms_dict[key] = value.tolist()

    ex = EdithExecutor()
    result = ex.run_function_on_endpoint(run_relaxation, atoms_dict)
    result_atoms = [ase.Atoms.fromdict(atoms) for atoms in result['result']]
    print(result_atoms)


if __name__ == "__main__":
    main()
