import ray
import os

print(f"{'CUDA_VISIBLE_DEVICES' in os.environ=}")

ray.init()

@ray.remote
def f():
    print(f"{'CUDA_VISIBLE_DEVICES' in os.environ=}")
    print(f"{os.environ['CUDA_VISIBLE_DEVICES'] == ''=}")

ray.get(f.remote())