from max_index import *
from data_generator import *
import math


def AODPart(proc_remote, proc_local, comms, D, R, gamma_min, accuracies):
  proc_local_known = []
  for i in range(len(proc_remote)):
    if sum(proc_local_known) > D:
      return accuracies[i-1],i-1
    comm_i = comms[i]
    f, point_f = max_acc_f(proc_remote, proc_local_known, comm_i, D, accuracies)
    g, point_g = max_acc_g(proc_remote, proc_local_known, comm_i, D, accuracies)
    h, point_h = max_acc_h(proc_remote, proc_local_known, comm_i, D, R, gamma_min, accuracies)
    if f > math.sqrt(g*h):
      return f, point_f
    proc_local_known.append(proc_local[i])
  return accuracies[len(proc_remote)], len(proc_remote)



def OPT(proc_remote, proc_local, comms, D, R, gamma_min, accuracies):
    accuracy_opt = accuracies[0]
    stage_opt = 0
    off_opt = 0

    for i in range(len(proc_remote)+1):
      if sum(proc_local[:len(proc_local)-i]) <= D:
        accuracy_temp, stage_temp = max_acc_f(proc_remote, proc_local[:len(proc_local)-i], comms[len(proc_local)-i], D, accuracies)
        if accuracy_temp > accuracy_opt:
          accuracy_opt = accuracy_temp
          stage_opt = stage_temp
          off_opt = len(proc_local)-i

    return accuracy_opt, stage_opt, off_opt

def Portend(proc_remote, proc_local, comms, proc_local_Port, comms_Port, D, R, gamma_min, accuracies):
  OPT_acc_Port, stage_Port, off_opt_port = OPT(proc_remote, proc_local_Port, comms_Port, D, R, gamma_min, accuracies)
  NeverOffload_acc, stage_NO = NeverOffload(proc_remote, proc_local, D, accuracies)
  if off_opt_port > stage_NO:
    Port_acc = NeverOffload_acc
    stage_PO = stage_NO
  else:
    Port_acc, stage_PO = max_acc_f(proc_remote, proc_local[:off_opt_port], comms[off_opt_port], D, accuracies)
  return Port_acc, stage_PO

def SPINN(proc_remote, proc_local, comms, proc_local_Port, comms_Port, D, R, gamma_min, SF, accuracies):
  OPT_acc_SPINN, stage_SPINN, off_opt_SPINN = OPT(proc_remote, SF*proc_local_Port, comms_Port, D, R, gamma_min, accuracies)
  NeverOffload_acc, stage_NO = NeverOffload(proc_remote, proc_local, D, accuracies)

  if off_opt_SPINN > stage_NO:
    SPINN_acc = NeverOffload_acc
    stage_SO = stage_NO
  else:
    SPINN_acc, stage_SO = max_acc_f(proc_remote, proc_local[:off_opt_SPINN], comms[off_opt_SPINN], D, accuracies)
  if off_opt_SPINN > 0:
    SF = sum(proc_local[:off_opt_SPINN])/sum(proc_local_Port[:off_opt_SPINN])
  return SPINN_acc, stage_SO, SF

def AlwaysOffload(proc_remote, proc_local, comms, D, accuracies):
  for i in range(len(proc_remote)+1):
    if comms[0]+sum(proc_remote[:len(proc_local)-i]) <= D:
      return accuracies[len(proc_remote)-i], len(proc_remote)-i
  return accuracies[0], 0

def NeverOffload(proc_remote, proc_local, D, accuracies):
  for i in range(len(proc_remote)+1):
    if sum(proc_local[:len(proc_local)-i]) <= D:
      return accuracies[len(proc_local[:len(proc_local)-i])],len(proc_local[:len(proc_local)-i])