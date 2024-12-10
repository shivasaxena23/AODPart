import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from algorithms import *
from data_generator import *

def experiment(proc_remote, input_data_real, pred_bin, accuracies, parameters):
  gamma_min = 0.5
  R = 1.5
  acc_exp = [[],[],[],[],[],[]]

  for k in parameters:
    D = math.pow(2,k)*sum(proc_remote)
    for g in range(40):
      AOD_bin = 0
      OPT_bin = 0
      Never_bin = 0
      Always_bin = 0
      Port_bin = 0
      SPINN_bin = 0
      SF = 1
      for instance in range(250):
          val = g*250+instance
          L = 4*input_data_real[0]/sum(proc_remote)
          proc_local = np.multiply(proc_remote,list(np.random.uniform(R,(R/gamma_min),len(proc_remote))))
          comms = get_comms(proc_remote, input_data_real, L)
          proc_local_Port = np.multiply(proc_remote,list([(R+(R/gamma_min))/2]*len(proc_remote)))
          bws_port= 0.5*L
          comms_Port = input_data_real/bws_port
          comms_Port = comms_Port.tolist()
          comms_Port.append(0)
          AODPart_acc, stage_AOD = AODPart(proc_remote, proc_local, comms, D, R, gamma_min, accuracies)
          OPT_acc, stage_OPT, off_opt = OPT(proc_remote, proc_local, comms, D, R, gamma_min, accuracies)
          NeverOffload_acc, stage_NO = NeverOffload(proc_remote, proc_local, D, accuracies)
          AlwaysOffload_acc, stage_AO = AlwaysOffload(proc_remote, proc_local, comms, D, accuracies)

          Port_acc, stage_PO = Portend(proc_remote, proc_local, comms, proc_local_Port, comms_Port, D, R, gamma_min, accuracies)

          SPINN_acc, stage_SO, SF = SPINN(proc_remote, proc_local, comms, proc_local_Port, comms_Port, D, R, gamma_min, SF, accuracies)

          AOD_bin += pred_bin[val,stage_AOD]
          OPT_bin += pred_bin[val,stage_OPT]
          Never_bin += pred_bin[val,stage_NO]
          Always_bin += pred_bin[val,stage_AO]
          Port_bin += pred_bin[val,stage_PO]
          SPINN_bin += pred_bin[val,stage_SO]

      AODPart_acc = AOD_bin/250
      OPT_acc = OPT_bin/250
      NeverOffload_acc = Never_bin/250
      AlwaysOffload_acc = Always_bin/250
      Port_acc = Port_bin/250
      SPINN_acc = SPINN_bin/250
      
      acc_exp[0].append([100*OPT_acc, k, "OPT"])
      acc_exp[1].append([100*AODPart_acc, k, "AODPart"])
      acc_exp[2].append([100*NeverOffload_acc, k, "NeverOffload"])
      acc_exp[3].append([100*AlwaysOffload_acc, k, "AlwaysOffload"])
      acc_exp[4].append([100*Port_acc, k, "Portend"])
      acc_exp[5].append([100*SPINN_acc, k, "SPINN"])
  return acc_exp


def plot(DNN_compute_values, DNN_prediction_values, input_data_real, pred_bin, accuracies, model, mode):
  DNN_compute_values_remote_expected = DNN_compute_values

  parameters = np.linspace(-1,1,13)

  acc_exp = experiment(DNN_compute_values_remote_expected, input_data_real, pred_bin, accuracies, parameters)

  compiled = []
  for i in acc_exp:
    for j in i:
      compiled.append(j)

  df_main1 = pd.DataFrame(compiled, columns = ['Accuracy', 'Deadline', 'Alg'])


  plt.rcParams["figure.figsize"] = [6,3]
  plt.rcParams["figure.dpi"] = 200
  plt.rcParams["figure.autolayout"] = True



  df_selected = df_main1.loc[df_main1['Deadline'] == 0]


  avg_acc_aodpart = df_selected.loc[df_selected['Alg'] == 'AODPart'].loc[:, 'Accuracy'].mean()
  avg_acc_portend = df_selected.loc[df_selected['Alg'] == 'Portend'].loc[:, 'Accuracy'].mean()
  avg_acc_SPINN = df_selected.loc[df_selected['Alg'] == 'SPINN'].loc[:, 'Accuracy'].mean()

  improvement_portend = (avg_acc_aodpart-avg_acc_portend)


  improvement_SPINN = (avg_acc_aodpart-avg_acc_SPINN)


  L = 4*input_data_real[0]/sum(DNN_compute_values_remote_expected)

  parameters = [round(elem, 2) for elem in parameters ]

  algs = ["OPT", "AODPart", "Portend","SPINN","NeverOffload","AlwaysOffload"]

  sns.set(font_scale=0.7,style='white')

  fig, ax = plt.subplots()


  d_style = {}
  for i in algs:
    d_style[i]=''

  d_style[algs[0]] = (10, 5)

  h = sns.lineplot(x="Deadline",y="Accuracy", hue="Alg", data=df_main1,style="Alg",linewidth=1, palette=['black', 'g','r','y','magenta', 'b'], markers=True, dashes=d_style, markersize=8, err_style="band", err_kws={'alpha':0.06})
  h.set_xticks(parameters) # <--- set the ticks first
  h.set_xlabel(r'$\log_2\delta$')
  h.set_ylabel('Average '+'Accuracy'+'%')
  h.ticklabel_format(useMathText=True)

  handles, labels = ax.get_legend_handles_labels()
  handles[0], handles[1] = handles[1], handles[0]
  labels[0], labels[1] = labels[1], labels[0]
  handles[2], handles[4] = handles[4], handles[2]
  labels[2], labels[4] = labels[4], labels[2]
  handles[3], handles[5] = handles[5], handles[3]
  labels[3], labels[5] = labels[5], labels[3]
  ax.legend(handles=handles[0:], labels=labels[0:], loc='lower right')




  # ax.grid(False)
  # plt.ticklabel_format(style='sci', axis='y')

  plt.savefig("plots/"+model+"/"+model+"_delta_"+mode+".pdf", bbox_inches='tight')
  return (improvement_portend, improvement_SPINN)

