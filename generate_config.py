from pprint import pprint
import json


def parallel(size=256, batch_size=256, num_angles=-1, det_count=-1, op="forward"):
    num_angles = num_angles if num_angles > 0 else size
    det_count = det_count if det_count > 0 else size

    return {
        "size": size,
        "batch_size": batch_size,
        "task": f"parallel {op}",
        "num_angles": num_angles,
        "det_count": det_count,
    }

def fanbeam(size=256, batch_size=256, num_angles=-1, det_count=-1, s_dist=-1, d_dist=-1, op="forward"):
    num_angles = num_angles if num_angles > 0 else size
    det_count = det_count if det_count > 0 else size
    s_dist = s_dist if s_dist > 0 else size
    d_dist = d_dist if d_dist > 0 else size

    return {
        "size": size,
        "batch_size": batch_size,
        "task": f"fanbeam {op}",
        "num_angles": num_angles,
        "det_count": det_count,
        "source_distance": s_dist,
        "detector_distance": d_dist,
        "det_spacing": 2.0
    }


tasks = []

for size in [32, 64, 96, 128, 128+64, 256, 256+128, 512]:
    tasks.append(parallel(size=size, op="forward"))
    tasks.append(parallel(size=size, op="backward"))
    tasks.append(fanbeam(size=size, op="forward"))
    tasks.append(fanbeam(size=size, op="backward"))


for batch_size in [4, 16, 32, 64, 96, 128, 128+64, 256, 256+128, 512]:
    tasks.append(parallel(batch_size=batch_size, size=64, op="forward"))
    tasks.append(parallel(batch_size=batch_size, size=64, op="backward"))
    tasks.append(fanbeam(batch_size=batch_size, size=64, op="forward"))
    tasks.append(fanbeam(batch_size=batch_size, size=64, op="backward"))


# make tasks unique
tasks = [dict(s) for s in set(frozenset(d.items()) for d in tasks)]
print(len(tasks))

with open("config.json", "w") as f:
    json.dump({
        "warmup": 10,
        "min_repeats": 10,
        "min_time": 2.0,
        "tasks": tasks
    }, f, indent=2)
