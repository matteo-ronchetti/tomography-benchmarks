from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import json
import glob

def match(pattern, target):
    for k in pattern:
        if not k in target or target[k] != pattern[k]:
            return False
    
    return True

class ObjectList:
    def __init__(self, objs: list):
        self.objs = objs

    def __len__(self):
        return len(self.objs)

    def __getitem__(self, idx):
        return self.objs[idx]
        
    def add(self, obj):
        if isinstance(obj, list):
            self.objs += obj
        else:
            self.objs.append(obj)
        
    def select(self, **pattern):
        return ObjectList([x for x in self.objs if match(pattern, x)])
    
    def sort(self, key):
        return ObjectList(sorted(self.objs, key=lambda x: x[key]))
    
    def project(self, *keys):
        res = []
        for k in keys:
            res.append([o[k] for o in self.objs])
        return res

    def unique(self, key):
        return list(set([o[key] for o in self.objs if key in o]))


def results_matrix(results, libraries, tasks):
    A = np.zeros((len(libraries), len(tasks)))

    for res in results:
        if res["library"] in libraries:
            y = libraries.index(res["library"])
            for line in res["results"]:
                if line["task"] in tasks:
                    x = tasks.index(line["task"])
                    A[y, x] = line["fps"]

    return A


def barplot(A, columns, groups, title="", spacing=0.1):
    x = np.arange(len(groups))  # the label locations
    width = (1.0 - spacing) / A.shape[0]  # the width of the bars

    params = {
        'font.size': 12,
        "figure.figsize": [10, 8]
    }
    plt.rcParams.update(params)

    fig, ax = plt.subplots()
    rects = []
    for i, lib in enumerate(columns):
        px = x + (i - len(columns) // 2) * width
        rects.append(ax.bar(px, A[i], width, label=lib))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Images/second')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(bbox_to_anchor=(0.5, 1), loc='upper center', ncol=len(columns))

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{int(np.round(height))}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    for rect in rects:
        autolabel(rect)

    plt.margins(0.05, 0.15)


def main():
    # gpu = 'GeForce GTX 1650'
    gpu = "Tesla T4"
    gpu_encoded = gpu.lower().replace(" ", "_")

    results = [json.load(open(p)) for p in glob.glob("results/*.json")]
    df = ObjectList([])

    for res in results:
        lib = res["library"]
        ggpu = res["gpu"] 
        for x in res["results"]:
            x["library"] = lib
            x["fps"] = x["batch_size"] / x["time"]
            x["gpu"] = ggpu
            df.add(x)

    df = df.select(gpu=gpu)
    tasks = ['parallel forward', 'parallel backward', 'fanbeam forward', 'fanbeam backward']
    libraries = sorted(df.unique("library"))

    # summary barplot
    fps = np.zeros((len(libraries), len(tasks)))
    for i, task in enumerate(tasks):
        fps[:, i] = df.select(task=task, batch_size=256, size=256).sort("library").project("fps")[0]
    barplot(fps, libraries, tasks, title=gpu)
    plt.savefig(f"figures/{gpu_encoded}_barplot.png")

    
    # input size plots
    batch_size = 256
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes.ravel()

    for i, task in enumerate(tasks):
        ax[i].set_title(task)
        for lib in libraries:
            x, y = df.select(library=lib, task=task, batch_size=batch_size).sort("size").project("size", "fps")
            ax[i].plot(x, y, label=lib)

        ax[i].set_ylabel('Images/second')
        ax[i].set_xlabel('Input size')
        ax[i].grid()
        ax[i].set_xticks(x)
        ax[i].set_yscale("log")
        ax[i].legend()

    fig.tight_layout(pad=5.0, h_pad=2.0, w_pad=3.0)
    fig.suptitle(f"Benchmarks executed on a {gpu} with batch size {batch_size}", fontsize=16)
    plt.savefig(f"figures/{gpu_encoded}_input_size.png")

    # batch size plots
    size = 64
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax = axes.ravel()

    for i, task in enumerate(tasks):
        ax[i].set_title(task)
        for lib in libraries:
            x, y = df.select(library=lib, task=task, size=64).sort("batch_size").project("batch_size", "fps")
            y = np.asarray(y)
            y /= y[0]
            ax[i].plot(x, y, label=lib)

        ax[i].set_ylabel('Relative speedup')
        ax[i].set_xlabel('Batch size')
        # ax[i].legend()
        ax[i].grid()
        ax[i].set_xticks([xx for xx in x if xx != 16])
        ax[i].set_yticks([2*i+1 for i in range(9)])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(libraries))
    fig.tight_layout(pad=5.0, h_pad=2.0, w_pad=3.0)
    fig.suptitle(f"Benchmarks executed on a {gpu} with volume size {size}$\\times${size}", fontsize=16)
    plt.savefig(f"figures/{gpu_encoded}_batch_size.png")

    plt.show()


main()