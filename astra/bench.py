import subprocess
import json
import time
import numpy as np
import astra

def get_gpu_name():
    command = "nvidia-smi --query-gpu=name --format=csv,noheader,nounits".split(" ")
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    return stdout.decode('UTF-8').strip()
    # libnames = ('libcuda.so', 'libcudart.so', 'libcuda.dylib', 'cuda.dll')
    # for libname in libnames:
    #     try:
    #         cuda = ctypes.CDLL(libname)
    #     except OSError:
    #         continue
    #     else:
    #         break
    # else:
    #     raise OSError("could not load any of: " + ' '.join(libnames))

    # name = b' ' * 100
    # device = ctypes.c_int()

    # cuda.cuDeviceGet(ctypes.byref(device), 0)
    # cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device)
    # return name.split(b'\0', 1)[0].decode()


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


class AstraParallelWrapper:
    def __init__(self, angles, img_size, det_count):
        self.angles = angles
        self.vol_geom = astra.create_vol_geom(img_size, img_size)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, det_count, self.angles)
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)

        self.data2d = []
        
    def forward(self, x):
        Y = np.empty((x.shape[0], len(self.angles), x.shape[1]), dtype=np.float32)

        for i in range(x.shape[0]):
            pid, Y[i] = astra.create_sino(x[i], self.proj_id)
            self.data2d.append(pid)

        return Y

    def backward(self, x):
        Y = np.empty((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float32)

        for i in range(x.shape[0]):
            pid, Y[i] = astra.create_backprojection(x[i], self.proj_id)
            self.data2d.append(pid)

        return Y

    def clean(self):
        for pid in self.data2d:
            astra.data2d.delete(pid)



class AstraFanbeamWrapper:
    def __init__(self, angles, img_size, det_count, sdist, ddist):
        self.angles = angles
        self.vol_geom = astra.create_vol_geom(img_size, img_size)
        self.proj_geom = astra.create_proj_geom('fanflat', 1.0, det_count, self.angles, sdist, ddist)
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)

        self.data2d = []

    def forward(self, x):
        Y = np.empty((x.shape[0], len(self.angles), x.shape[1]), dtype=np.float32)

        for i in range(x.shape[0]):
            pid, Y[i] = astra.create_sino(x[i], self.proj_id)
            self.data2d.append(pid)

        return Y

    def backward(self, x):
        Y = np.empty((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float32)

        for i in range(x.shape[0]):
            pid, Y[i] = astra.create_backprojection(x[i], self.proj_id)
            self.data2d.append(pid)

        return Y

    def clean(self):
        for pid in self.data2d:
            astra.data2d.delete(pid)


def bench_parallel_forward(batch, size, det_count, num_angles, *bench_args):
    phantom = np.random.randn(batch, size, size)
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)

    radon = AstraParallelWrapper(angles, size, det_count)
    def f(x): return radon.forward(x)

    res = benchmark(f, phantom, *bench_args)
    radon.clean()
    return res

def bench_parallel_backward(batch, size, det_count, num_angles, *bench_args):
    phantom = np.random.randn(batch, size, size)
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)

    radon = AstraParallelWrapper(angles, size, det_count)
    def f(x): return radon.backward(x)

    res = benchmark(f, phantom, *bench_args)
    radon.clean()
    return res



def bench_fanbeam_forward(batch, size, det_count, num_angles, source_dist, det_dist, *bench_args):
    phantom = np.random.randn(batch, size, size)
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)

    radon = AstraFanbeamWrapper(angles, size, det_count, source_dist, det_dist)
    def f(x): return radon.forward(x)
    
    res = benchmark(f, phantom, *bench_args)
    radon.clean()
    return res

def bench_fanbeam_backward(batch, size, det_count, num_angles, source_dist, det_dist, *bench_args):
    phantom = np.random.randn(batch, size, size)
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)

    radon = AstraFanbeamWrapper(angles, size, det_count, source_dist, det_dist)
    def f(x): return radon.backward(x)
    
    res = benchmark(f, phantom, *bench_args)
    radon.clean()
    return res

with open("../config.json") as f:
    config = json.load(f)

warmup = config["warmup"]
min_repeats = config["min_repeats"]
min_time = config["min_time"]
bench_args = (warmup, min_repeats, min_time)

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
                                          task["source_distance"], task["detector_distance"],
                                          *bench_args)
    elif task["task"] == "fanbeam backward":
        exec_time = bench_fanbeam_backward(*bs,
                                           task["num_angles"], task["det_count"],
                                           task["source_distance"], task["detector_distance"],
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

with open(f"../results/astra_{gpu_encoded}.json", "w") as f:
    config = json.dump({
        "library": "Astra",
        "warmup": config["warmup"],
        "min_repeats": config["min_repeats"],
        "min_time": config["min_time"],
        "gpu": gpu_name,

        "results": results
    }, f, indent=4)
