"""
Simple utility for running Python functions on Globus Compute endpoints
using subprocess with conda environments.
"""

import subprocess
import pickle
import base64
import inspect
from typing import Any, Callable

import globus_sdk
from globus_compute_sdk import Executor
from globus_compute_sdk.serialize import ComputeSerializer, CombinedCode

EDITH_EP_ID = "a01b9350-e57d-4c8e-ad95-b4cb3c4cd1bb"

class EdithExecutor:
    def __init__(self, endpoint_id=EDITH_EP_ID):
        self.endpoint_id = endpoint_id
        self.endpoint_config = {
            "worker_init": """
                # need to load openmpi to avoid 'no non PBS mpiexec available' error
                module load openmpi
                # path where globus-compute-endpoint lives
                export PATH=$PATH:/usr/sbin
            """,
            "engine": {
                "provider": {
                    "nodes_per_block": 2,
                },
            }
        }

    def run_function_on_endpoint(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function on a Globus Compute endpoint using subprocess with conda environment.

        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function execution
        """

        # Get function source and serialize arguments
        func_source = inspect.getsource(func)
        func_name = func.__name__

        # Create the remote executor function
        def remote_executor(func_source, func_name, *args, **kwargs):
            import subprocess
            import pickle
            import base64
            import tempfile
            import os

            # Function data to execute
            func_data = {
                'source': func_source,
                'name': func_name,
                'args': args,
                'kwargs': kwargs
            }

            # Encode function data
            encoded_data = base64.b64encode(pickle.dumps(func_data)).decode()

            # Python script to run in conda environment
            script = f"""import pickle
import base64

# Decode function data
func_data = pickle.loads(base64.b64decode("{encoded_data}"))

# Execute function source to define it
exec(func_data["source"])

# Get function object and execute it
func_obj = locals()[func_data["name"]]
result = func_obj(*func_data["args"], **func_data["kwargs"])

# Serialize and print result for capture
result_data = base64.b64encode(pickle.dumps(result)).decode()
print("RESULT_DATA:", result_data)
"""

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                script_path = f.name
            
            try:
                # Run in conda environment
                cmd = ['conda', 'run', '-n', 'torch-sim-edith', 'python', script_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            finally:
                # Clean up temporary file
                os.unlink(script_path)

            if result.returncode != 0:
                return {'error': result.stderr, 'stdout': result.stdout}

            # Extract result from subprocess output
            result_data = None
            
            for line in result.stdout.split('\n'):
                if line.startswith('RESULT_DATA: '):
                    result_data = line[13:].strip()
                    break
            
            if result_data is not None:
                return {
                    'raw_data': result_data, 
                    'stdout': result.stdout, 
                    'stderr': result.stderr
                }

            return {'error': 'No result found', 'stdout': result.stdout, 'stderr': result.stderr}

        # Execute on endpoint
        with Executor(endpoint_id=self.endpoint_id, user_endpoint_config=self.endpoint_config) as gce:
            gce.serializer = ComputeSerializer(strategy_code=CombinedCode())
            future = gce.submit(remote_executor, func_source, func_name, *args, **kwargs)
            result = future.result()
            
            # If we got raw data back, try to unpickle it locally
            if 'raw_data' in result:
                try:
                    actual_result = self.decode_result_data(result['raw_data'])
                    return {'result': actual_result}
                except Exception as e:
                    return {'error': f'Failed to decode result locally: {e}', 'raw_data': result['raw_data'], 'stdout': result.get('stdout', ''), 'stderr': result.get('stderr', '')}
            
            return result

    def decode_result_data(self, raw_data: str):
        """
        Decode base64 pickled result data when ASE is available.
        
        Args:
            raw_data: Base64 encoded pickled data
            
        Returns:
            Unpickled result object
        """
        try:
            import ase
            import ase.atoms
            import ase.cell
            return pickle.loads(base64.b64decode(raw_data))
        except ImportError:
            raise ImportError("ASE is required to decode result data containing ASE objects")


# Test function
def hello_world():
    return "Hello, world!"

def test_conda_env():
    import torch
    return f"Torch Version: {torch.__version__}"


def test_basic_md():
    import torch
    import torch_sim as ts
    from ase.build import bulk
    from torch_sim.models.lennard_jones import LennardJonesModel

    # Create a Lennard-Jones model with parameters suitable for Si
    lj_model = LennardJonesModel(
        sigma=2.0,  # Å, typical for Si-Si interaction
        epsilon=0.1,  # eV, typical for Si-Si interaction
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float64,
    )
    print(f"Model running on device: {lj_model.device}")
    # Create a silicon FCC structure using ASE
    cu_atoms = bulk("Cu", "fcc", a=5.43, cubic=True)

    n_steps = 50
    final_state = ts.integrate(
        system=cu_atoms,  # Input atomic system
        model=lj_model,  # Energy/force model
        integrator=ts.nvt_langevin,  # Integrator to use
        n_steps=n_steps,  # Number of MD steps
        temperature=2000,  # Target temperature (K)
        timestep=0.002,  # Integration timestep (ps)
    )

    # Convert the final state back to ASE atoms
    return final_state.to_atoms()
