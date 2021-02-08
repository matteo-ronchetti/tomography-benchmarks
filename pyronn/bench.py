import ctypes
import sys
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.geometry.geometry_fan_2d import GeometryFan2D
from pyronn.ct_reconstruction.layers.backprojection_2d import fan_backprojection2d
from pyronn.ct_reconstruction.layers.projection_2d import fan_projection2d
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d
from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d
import json
import time
import tensorflow as tf
import numpy as np
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_gpu_name():
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + ' '.join(libnames))

    name = b' ' * 100
    device = ctypes.c_int()

    cuda.cuDeviceGet(ctypes.byref(device), 0)
    cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device)
    return name.split(b'\0', 1)[0].decode()


def benchmark(f, x, warmup, repeats, min_time):
    for _ in range(warmup):
        y = f(x)

    count = 0
    s = time.time()
    while True:
        y = f(x)
        count += 1
        e = time.time()
        if count >= repeats and (e-s) >= min_time:
            break

    mean_execution_time = (time.time() - s)/count

    return mean_execution_time


def create_parallel_geometry(size, det_count, num_angles):
    # Detector Parameters:
    detector_shape = det_count
    detector_spacing = 1

    # Trajectory Parameters:
    number_of_projections = num_angles
    angular_range = np.pi

    # create Geometry class
    geometry = GeometryParallel2D([size, size], [1, 1], detector_shape, detector_spacing, number_of_projections,
                                  angular_range)
    geometry.set_trajectory(circular_trajectory.circular_trajectory_2d(geometry))

    return geometry


def create_fanbeam_geometry(size, det_count, num_angles, source_dist, det_dist, det_spacing):
    # Detector Parameters:
    detector_shape = det_count
    detector_spacing = det_spacing

    # Trajectory Parameters:
    number_of_projections = num_angles
    angular_range = np.pi

    source_detector_distance = source_dist + det_dist

    # create Geometry class
    geometry = GeometryFan2D([size, size], [1, 1], detector_shape, detector_spacing, number_of_projections,
                             angular_range, source_detector_distance, source_dist)
    geometry.set_trajectory(circular_trajectory.circular_trajectory_2d(geometry))

    return geometry


def bench_parallel_forward(batch, size, det_count, num_angles, *bench_args):
    with tf.device('/GPU:0'):
        phantom = tf.random.normal((batch, size, size))
    geometry = create_parallel_geometry(size, det_count, num_angles)

    def f(x): return parallel_projection2d(x, geometry)

    return benchmark(f, phantom, *bench_args)


def bench_parallel_backward(batch, size, det_count, num_angles, *bench_args):
    with tf.device('/GPU:0'):
        phantom = tf.random.normal((batch, size, size))
    geometry = create_parallel_geometry(size, det_count, num_angles)

    sino = parallel_projection2d(phantom, geometry)
    def f(x): return parallel_backprojection2d(x, geometry)

    return benchmark(f, sino, *bench_args)


def bench_fanbeam_forward(batch, size, det_count, num_angles, source_dist, det_dist, det_spacing, *bench_args):
    with tf.device('/GPU:0'):
        phantom = tf.random.normal((batch, size, size))
    geometry = create_fanbeam_geometry(size, det_count, num_angles, source_dist, det_dist, det_spacing)

    def f(x): return fan_projection2d(x, geometry)

    return benchmark(f, phantom, *bench_args)


def bench_fanbeam_backward(batch, size, det_count, num_angles, source_dist, det_dist, det_spacing, *bench_args):
    with tf.device('/GPU:0'):
        phantom = tf.random.normal((batch, size, size))
    geometry = create_fanbeam_geometry(size, det_count, num_angles, source_dist, det_dist, det_spacing)

    sino = fan_projection2d(phantom, geometry)
    def f(x): return fan_backprojection2d(x, geometry)

    return benchmark(f, sino, *bench_args)


with open("../config.json") as f:
    config = json.load(f)

warmup = config["warmup"]
min_repeats = config["min_repeats"]
min_time = config["min_time"]
bench_args = (warmup, min_repeats, min_time)

# Place phantom on the GPU
# with tf.device('/GPU:0'):
#     phantom = tf.convert_to_tensor(phantom, dtype=tf.float32)

gpu_name = get_gpu_name()
gpu_encoded = gpu_name.lower().replace(" ", "_")
print(f"Running benchmarks on {gpu_name} ({gpu_encoded})")

print("\n")
results = []
for task in config["tasks"]:
    bs = [task["batch_size"], task["size"]]
    print(f"Benchmarking task '{task['task']}', batch size: { task['batch_size']}, size: {task['size']}")

    if task["task"] == "parallel forward":
        exec_time = bench_parallel_forward(*bs, task["num_angles"], task["det_count"], *bench_args)
    elif task["task"] == "parallel backward":
        exec_time = bench_parallel_backward(*bs, task["num_angles"], task["det_count"], *bench_args)
    elif task["task"] == "fanbeam forward":
        exec_time = bench_fanbeam_forward(*bs,
                                          task["num_angles"], task["det_count"],
                                          task["source_distance"], task["detector_distance"], task["det_spacing"],
                                          *bench_args)
    elif task["task"] == "fanbeam backward":
        exec_time = bench_fanbeam_backward(*bs,
                                           task["num_angles"], task["det_count"],
                                           task["source_distance"], task["detector_distance"], task["det_spacing"],
                                           *bench_args)
    else:
        print(f"ERROR Unknown task '{task['task']}'")
        continue

    print("Execution time:", exec_time)

    res = dict()
    for k in task:
        res[k] = task[k]

    res["time"] = exec_time
    results.append(res)
    print("")

with open(f"../results/pyronn_{gpu_encoded}.json", "w") as f:
    config = json.dump({
        "library": "Pyronn",
        "warmup": config["warmup"],
        "min_repeats": config["min_repeats"],
        "min_time": config["min_time"],
        "gpu": gpu_name,

        "results": results
    }, f, indent=4)
