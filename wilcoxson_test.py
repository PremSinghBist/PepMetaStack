from scipy.stats import wilcoxon
import numpy as np
import pandas as pd


def wilcoxson_test(base_metrics, new_metrics):
    '''Performs Wilcoxon signed-rank test between two sets of performance metrics.'''
    stat, p_value = wilcoxon(base_metrics, new_metrics)
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")
    #Save statistic and p-value to a file results/wilcoxon_test_results.txt
    with open("./results/ecoli_random_seed_ablation/wilcoxon_test_results.txt", "w") as f:       
        f.write(f"Wilcoxon test statistic: {stat}, p-value: {p_value}\n")
    return stat, p_value


if __name__ == "__main__":

    RMSE_BASE_METRICS = [0.529, 0.526, 0.528, 0.533, 0.530, 0.529, 0.526, 0.528, 0.533, 0.530] #MSCMamba RMSE values
    PEPSTACK_METRICS = [0.4495410536250308,
                        0.4490482396917879,
                        0.44963173849410754,
                        0.4495877881131551,
                        0.44845228587948044,
                        0.45027820706674515,
                        0.4486379185784748,
                        0.44878298159219226,
                        0.44831189608136557,
                        0.4491994814360393
                        ] # PepMetastack Model RMSE values across 10 seeds
    

    stats, p_value = wilcoxson_test(RMSE_BASE_METRICS, PEPSTACK_METRICS)