import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
from algorithms import *
from data_generator import *

def experiment(proc_remote, input_data_real, pred_bin, accuracies, parameters):
  R = 1.5
  acc_exp = [[],[],[],[],[],[]]

  for gamma_min in parameters:
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
          D = sum(proc_remote)
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
      
      acc_exp[0].append([100*OPT_acc, gamma_min, "OPT"])
      acc_exp[1].append([100*AODPart_acc, gamma_min, "AODPart"])
      acc_exp[2].append([100*NeverOffload_acc, gamma_min, "NeverOffload"])
      acc_exp[3].append([100*AlwaysOffload_acc, gamma_min, "AlwaysOffload"])
      acc_exp[4].append([100*Port_acc, gamma_min, "Portend"])
      acc_exp[5].append([100*SPINN_acc, gamma_min, "SPINN"])
  return acc_exp


def plot(DNN_compute_values, DNN_prediction_values, input_data_real, pred_bin, accuracies, model, mode):
    
    DNN_compute_values_remote_expected = DNN_compute_values

    parameters = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 8/8]
    acc_exp = experiment(DNN_compute_values_remote_expected, input_data_real, pred_bin, accuracies, parameters)

    compiled = []
    for i in acc_exp:
        for j in i:
            compiled.append(j)

    df_main1 = pd.DataFrame(compiled, columns = ['Accuracy', 'g', 'Alg'])

    plt.rcParams["figure.figsize"] = [6,3]
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["figure.autolayout"] = True



    df_selected = df_main1.loc[df_main1['g'] == 0.5]


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

    h = sns.lineplot(x="g",y="Accuracy", hue="Alg", data=df_main1,style="Alg",linewidth=1, palette=['black', 'g','r','y','magenta', 'b'],
        markers=True, dashes=d_style, markersize=8, err_style="band", err_kws={'alpha':0.1}) #NEWALGALPHARAND
    h.set_xticks(parameters) # <--- set the ticks first
    h.set_xlabel(r'$\gamma$'+'$_\mathregular{min}$')
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


    plt.savefig("plots/"+model+"/"+model+"_gamma_"+mode+".pdf", bbox_inches='tight')
    return (improvement_portend, improvement_SPINN)
