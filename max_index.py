from data_generator import *

def max_acc_f(proc_remote_known, proc_local_known, comm_i, D, accuracies):
    if sum(proc_local_known) + comm_i > D:
        return accuracies[len(proc_local_known)],len(proc_local_known)
    for i in range(len(proc_remote_known)-len(proc_local_known)+1):
        if sum(proc_local_known) + comm_i + sum(proc_remote_known[len(proc_local_known):len(proc_remote_known)-i]) <= D:
            return accuracies[len(proc_remote_known)-i],len(proc_remote_known)-i

def max_acc_h(proc_remote_known, proc_local_known, comm_i, D, R, gamma_min, accuracies):
    for i in range(len(proc_remote_known)-len(proc_local_known)+1):
        if sum(proc_local_known) + (R/gamma_min)*sum(proc_remote_known[len(proc_local_known):len(proc_remote_known)-i]) <= D:
            return accuracies[len(proc_remote_known)-i],len(proc_remote_known)-i


def max_acc_g(proc_remote_known, proc_local_known, comm_i, D, accuracies):
    for i in range(len(proc_remote_known)-len(proc_local_known)+1):
        if sum(proc_local_known) + sum(proc_remote_known[len(proc_local_known):len(proc_remote_known)-i]) <= D:
            return accuracies[len(proc_remote_known)-i],len(proc_remote_known)-i
    print("Here")
    return accuracies[len(proc_remote_known)]




