import numpy as np
import torch
import time
import json

from torch_radon import Radon, Projection
# from torch_radon import Radon, RadonFanbeam


def benchmark(f, x, warmup, repeats, min_time):
    for _ in range(warmup):
        y = f(x)

    count = 0
    torch.cuda.synchronize()
    s = time.time()
    while True:
        y = f(x)
        torch.cuda.synchronize()
        count += 1
        e = time.time()
        if count >= repeats and (e-s) >= min_time:
            break

    mean_execution_time = (time.time() - s)/count

    return mean_execution_time


def bench_parallel_forward(task, dtype, device, *bench_args):
    num_angles = task["num_angles"]
    det_count = task["det_count"]

    x = torch.randn(task["batch_size"], task["size"], task["size"], dtype=dtype, device=device)
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    projection = Projection.parallel_beam(det_count)
    radon = Radon(angles, task["size"], projection)

    def f(x): return radon.forward(x)

    return benchmark(f, x, *bench_args)


def bench_parallel_backward(task, dtype, device, *bench_args):
    num_angles = task["num_angles"]
    det_count = task["det_count"]

    x = torch.randn(task["batch_size"], task["size"], task["size"], dtype=dtype, device=device)
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    projection = Projection.parallel_beam(det_count)
    radon = Radon(angles, task["size"], projection)
    # radon = Radon(phantom.size(1), np.linspace(0, np.pi, num_angles, endpoint=False), det_count)

    sino = radon.forward(x)
    def f(x): return radon.backward(x)

    return benchmark(f, sino, *bench_args)


def bench_fanbeam_forward(task, dtype, device, *bench_args):
    num_angles = task["num_angles"]
    det_count = task["det_count"]
    source_dist = task["source_distance"]
    det_dist = task["detector_distance"]

    x = torch.randn(task["batch_size"], task["size"], task["size"], dtype=dtype, device=device)
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)

    projection = Projection.fanbeam(source_dist, det_dist, det_count)
    radon = Radon(angles, task["size"], projection)
    # radon = RadonFanbeam(phantom.size(1), angles, source_dist, det_dist, det_count)

    def f(x): return radon.forward(x)

    return benchmark(f, x, *bench_args)


def bench_fanbeam_backward(task, dtype, device, *bench_args):
    num_angles = task["num_angles"]
    det_count = task["det_count"]
    source_dist = task["source_distance"]
    det_dist = task["detector_distance"]

    x = torch.randn(task["batch_size"], task["size"], task["size"], dtype=dtype, device=device)
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)

    projection = Projection.fanbeam(source_dist, det_dist, det_count)
    radon = Radon(angles, task["size"], projection)
    # radon = RadonFanbeam(phantom.size(1), angles, source_dist, det_dist, det_count)

    sino = radon.forward(x)
    def f(x): return radon.backward(x)

    return benchmark(f, x, *bench_args)


def do_benchmarks(config, dtype, device):
    output_prefix = "hp_" if dtype == torch.float16 else ""

    warmup = config["warmup"]
    min_repeats = config["min_repeats"]
    min_time = config["min_time"]
    bench_args = (warmup, min_repeats, min_time)

    results = []
    for task in config["tasks"]:
        print(f"Benchmarking task '{task['task']}', batch size: { task['batch_size']}, size: {task['size']}")

        if task["task"] == "parallel forward":
            exec_time = bench_parallel_forward(task, dtype, device, *bench_args)
        elif task["task"] == "parallel backward":
            exec_time = bench_parallel_backward(task, dtype, device, *bench_args)
        elif task["task"] == "fanbeam forward":
            exec_time = bench_fanbeam_forward(task, dtype, device, *bench_args)
        elif task["task"] == "fanbeam backward":
            exec_time = bench_fanbeam_backward(task, dtype, device, *bench_args)
        else:
            print(f"Unknown task '{task['task']}'")
            continue

        print("Execution time:", exec_time)

        res = dict()
        for k in task:
            res[k] = task[k]

        res["time"] = exec_time
        results.append(res)
        print("")

    return results


def main():
    with open("../config.json") as f:
        config = json.load(f)

    with torch.no_grad():
        device = torch.device("cuda")

        gpu_name = torch.cuda.get_device_name(device)
        gpu_encoded = gpu_name.lower().replace(" ", "_")
        print(f"Running benchmarks on {gpu_name} ({gpu_encoded})")

        print("\nBenchmarking Single Precision")
        results = do_benchmarks(config, torch.float32, device)
        print("\n\nBenchmarking Half Precision")
        results_hp = do_benchmarks(config, torch.float16, device)

    with open(f"../results/torch_radon_{gpu_encoded}.json", "w") as f:
        json.dump({
            "library": "TorchRadon",
            "warmup": config["warmup"],
            "min_repeats": config["min_repeats"],
            "min_time": config["min_time"],
            "gpu": gpu_name,

            "results": results
        }, f, indent=4)

    with open(f"../results/torch_radon_hp_{gpu_encoded}.json", "w") as f:
        json.dump({
            "library": "TorchRadon half",
            "warmup": config["warmup"],
            "min_repeats": config["min_repeats"],
            "min_time": config["min_time"],
            "gpu": gpu_name,

            "results": results_hp
        }, f, indent=4)


main()
