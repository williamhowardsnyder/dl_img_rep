import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# ImageNet1k
db_size = "1281167"
# db_size = "127149"

# Places365
# db_size = "178050"
# db_size = "1803460"

PLOT_PATH = "/home/howarwil/llc_ann/performance_plots/db_size_"+ db_size +"/"
RES_PATH = "/home/howarwil/llc_ann/results_" + "imagenet1k/"#"imagenet1k/fine_features/super_fine_"

# num_features  - Number of dimensions used from ResNet Model
# delay         - Time in seconds it takes to retrieve a query (average over validation set)
# prep_time     - Time to build the index
metric_list = ["MAP_100", "MAP_50", "delay", "prep_time", "index_size"]

name_mapping = {
    "MAP_100": "MAP@100",
    "MAP_50": "MAP@50",
    "delay": "Query Time (s)",
    "prep_time": "Construction Time (s)",
    "index_size": "Size of Index (bytes)"
    }

param_mapping = {
    "mneighbors": "Number of Neighbors per Node",
    "nbits": "Number of Bits per Point",
    "nleaves": "Number of Objects per Leaf"
    }

index_mapping = {
    "hnsw": "HNSW Graph",
    "lsh": "LSH Table",
    "kd": "KD Tree",
    "exact_sklearn": "Exact Search (sklearn)",
    "exact": "Exact Search"
    }


def main():
    print("Plotting results...")

    # Generates a table of values that can be easily plugged into latex
    # generate_tables()


    for metric in metric_list:
        title = "Comparison of " + name_mapping[metric] # + " on Places"
        generate_data_structure_comparison_plots(metric, title=title)
    
    # for index in ["hnsw", "lsh", "kd"]:
    #     results = torch.load("/home/howarwil/llc_ann/results/"+ index + "_results_gpu0_" + db_size + ".pt")
    #     if index == "hnsw":
    #         param = "mneighbors"
    #     elif index == "lsh":
    #         param = "nbits"
    #     elif index == "kd":
    #         param = "nleaves"

    #     for metric in metric_list:
    #         # title1 = name_mapping[metric] + " vs Dimensionality of Representation"
    #         # generate_metric_vs_num_features_plots(results, metric, title=title1)
    #         title = name_mapping[metric] + " vs " + param_mapping[param] + " in " + index_mapping[index]
    #         generate_metric_vs_param_plots(results, metric, title=title, param=param, index=index)


def generate_tables():
    index_info = [("hnsw", "mneighbors", 32), ("lsh", "nbits", 256), ("kd", "nleaves", 40), ("exact_sklearn", "no_param", -1)]
    # index_info = [("kd", "nleaves", 40), ("exact_sklearn", "no_param", -1)]
    results = {} # dict: index -> (dict: dim -> (dict: metric -> value))
    for index, param, val in index_info:
        if index == "exact":
            ind  = "hnsw"
            param = "mneighbors"
            val = -1
            res = torch.load("/home/howarwil/llc_ann/results_imagenet1k/"+ ind + "_results_gpu0_" + db_size + ".pt")
        else:
            res = torch.load("/home/howarwil/llc_ann/results_imagenet1k/"+ index + "_results_gpu0_" + db_size + ".pt")
        
        print(res)
        results[index] = {}
        for i in range(len(res)):
            info = res[i]
            if info[param] != val:
                continue

            dim = info["num_features"]
            values = {}
            for metric in metric_list:
                values[metric] = info[metric]
            
            # if results[index][dim] is None:
            #     results[index][dim] = {}
            
            results[index][dim] = values

    # torch.save(results)
    # Write to file in latex format
    f = open(f'performance_table_{db_size}.txt','w')
    for dim in [16, 32, 64, 128, 256, 2048]:
    # for dim in [8, 12, 16, 20, 24, 28, 32]:
        f.write(f"{dim}\n")
        for metric in metric_list:
            f.write(f"{name_mapping[metric]}")
            for index, param, val in index_info:
                val = results[index][dim][metric]
                if metric == "index_size":
                    val /= 1000000   # MB
                    f.write(f" & {val:.0f}")
                elif metric == "delay":
                    val *= 1000000   # micro seconds
                    f.write(f" & {val:.0f}")
                else:
                    f.write(f" & {val:.3f}")

                if index == "exact_sklearn":
                    f.write(" \\\\ \n \hline \n")
        f.write("\n")


def generate_data_structure_comparison_plots(metric, title="Index MAP@100 Comparison"):
    indexes = {     # x   y
              "kd": ([], []),
#    "exact_sklearn": ([], [])
             "lsh": ([], []),
            "hnsw": ([], [])
    }
    for index in indexes.keys():
        if index == "hnsw":
            param = "mneighbors"
        elif index == "lsh":
            param = "nbits"
        elif index == "kd":
            param = "nleaves"
        
        # TODO: Generalize this file path or find a way to let the user change it easily
        results = torch.load(RES_PATH + index + "_results_gpu0_" + db_size + ".pt")

        # print(index)
        # for i in range(len(results)):
        #     print(results[i])
        # print()

        for i in range(len(results)):
            result = results[i]
            # Good param values for hnsw, lsh, and kd, respectively
            if (index == "hnsw" and result[param] == 32) or\
               (index == "lsh" and result[param] == 256) or\
               (index == "kd" and result[param] == 40) or\
               (index == "exact_sklearn"):
            # if True:
                # print(results[i])
                (indexes[index])[0].append(result["num_features"])
                (indexes[index])[1].append(result[metric])
    
    xy_vals = []
    line_labels = []
    for index in indexes:
        xy_vals.append(indexes[index])
        line_labels.append(index)

    x_label = "Number of Features"
    y_label = name_mapping[metric]

    save_to = PLOT_PATH + "data_structures/" + metric + "_vs_num_feature.png"
    make_plot(xy_vals, line_labels, x_label, y_label, title, save_to=save_to, legend_title="Indexes", xscale='log')


def generate_metric_vs_param_plots(results, metric, title="plot", param="mneighbors", index="hnsw", rep_dict=None):
    num_feat = {#  x   y
           "16": ([], []),
           "32": ([], []),
           "64": ([], []),
          "128": ([], []),
          "256": ([], []),
         "2048": ([], [])
    }
    if rep_dict is not None:
        num_feat = rep_dict

    for i in range(len(results)):
        result = results[i]
        if result[param] == -1: # Don't worry about exact search for now
            continue

        n = str(result["num_features"])
        (num_feat[n])[0].append(result[param])
        (num_feat[n])[1].append(result[metric])

    xy_vals = []
    line_labels = []
    for n in num_feat:
        xy_vals.append(num_feat[n])
        line_labels.append(n)

    x_label = param_mapping[param]
    y_label = name_mapping[metric]

    save_to = PLOT_PATH + index + "_" + metric + "_vs_"+ param +".png"
    make_plot(xy_vals, line_labels, x_label, y_label, title, save_to=save_to, legend_title="Feature Dim")


def make_plot(xy_vals, line_labels, x_label, y_label, title, save_to="plot.png", legend_title=None, xscale="linear"):
    """
    Given a list of pairs of lists such as [([x_1], [y_1]), ([x_2], [y_2]), ...], generates a single line
    plot for each pair, where the xy values of that line are the values in the list (eg. the list x_1 and
    the list y_1). The parameter line_labels are what we should call each line in the legend, and x_label,
    and y_label are the labels for the x and y axes, respectively.
    """
    assert len(xy_vals) == len(line_labels)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(xscale)

    for i in range(len(xy_vals)):
        x, y = xy_vals[i]
        line_label = line_labels[i]
        plt.plot(x, y, label=line_label)
    
    if legend_title is None:
        plt.legend()
    else:
        plt.legend(title=legend_title)
    plt.savefig(save_to)
    plt.clf()

# TODO: This can be deleted probably
# def generate_metric_vs_num_features_plots(results, metric, title="plot"):
#     ms = {      # x    y
#         "exact": ([], []),
#             "8": ([], []),
#            "16": ([], []),
#            "32": ([], [])
#     }
#     for i in range(len(results)):
#         result = results[i]
#         if result["M"] == -1:
#             m = "exact"
#         else:
#             m = str(result["M"])
#         (ms[m])[0].append(result["num_features"])
#         (ms[m])[1].append(result[metric])

#     xy_vals = []
#     line_labels = []
#     for m in ms:
#         xy_vals.append(ms[m])
#         line_labels.append(m)

#     x_label = "Number of Features"
#     y_label = name_mapping[metric]

#     save_to = "performance_plots/" + metric + "_vs_num_feature.png"
#     make_plot(xy_vals, line_labels, x_label, y_label, title, save_to=save_to, legend_title="Num Neighbors")



if __name__ == "__main__":
    main()