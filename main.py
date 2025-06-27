from globus_compute_sdk import Executor


def hello_edith() -> str:
     return "Hello from Edith!"

def hello_container():
    hholb_test_ep_id = "161b58af-f871-4975-bae5-79ad984244d6"
    config = {
        "container_type": "podman",
        "container_uri": "compute-worker:3.8.0",
        "container_cmd_options": "-v /tmp:/tmp",
    }
    with Executor(
        endpoint_id=hholb_test_ep_id,
        user_endpoint_config=config,
    ) as gce:
        f = gce.submit(hello_edith)
        print(f.result())

def main():
    edith_endpoint_id = "a01b9350-e57d-4c8e-ad95-b4cb3c4cd1bb" 
    
    with Executor(endpoint_id=edith_endpoint_id) as gce:
        future = gce.submit(hello_edith)
        print(future.result())
    
