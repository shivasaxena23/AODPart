import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
from algorithms import *
from data_generator import *

def experiment(proc_remote, input_data_real, pred_bin, accuracies, parameters, gamma_min_actual):
  R = 1.5
  acc_exp = []

  for k in parameters:
    D = sum(proc_remote)
    AOD_bin = 0
    OPT_bin = 0
    gamma_min = gamma_min_actual*(1+k)
    for g in range(40):
      for val in range(250):

          L = 4*input_data_real[0]/sum(proc_remote)
          proc_local = np.multiply(proc_remote,list(np.random.uniform(R,(R/gamma_min_actual),len(proc_remote))))
          comms = get_comms(proc_remote, input_data_real, L)
          AODPart_acc, stage_AOD = AODPart(proc_remote, proc_local, comms, D, R, gamma_min, accuracies)
          OPT_acc, stage_OPT, off_opt = OPT(proc_remote, proc_local, comms, D, R, gamma_min_actual, accuracies)


          AOD_bin += pred_bin[val,stage_AOD]
          OPT_bin += pred_bin[val,stage_OPT]


      AODPart_acc = AOD_bin/250
      OPT_acc = OPT_bin/250
      acc_exp.append([OPT_acc/AODPart_acc, k, "DNN "+str(gamma_min_actual)])

  return acc_exp

def plot(DNN_compute_values, DNN_prediction_values, input_data_real, pred_bin, accuracies, model):
    DNN_compute_values_remote_expected = DNN_compute_values

    parameters =  np.linspace(-0.3,0.3,7)
    acc_exp_1 = experiment(DNN_compute_values_remote_expected, input_data_real, pred_bin, accuracies, parameters, 0.25)
    acc_exp_2 = experiment(DNN_compute_values_remote_expected, input_data_real, pred_bin, accuracies, parameters, 0.5)
    acc_exp_3 = experiment(DNN_compute_values_remote_expected, input_data_real, pred_bin, accuracies, parameters, 0.75)
    acc_exp = acc_exp_1 + acc_exp_2 + acc_exp_3
    df_main1 = pd.DataFrame(acc_exp, columns = ['Emperical Performance Ratio', 'Estimation Error', 'Class'])
    plt.rcParams["figure.figsize"] = [6,3]
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["figure.autolayout"] = True
    algs = ["DNN 0.25", "DNN 0.5", "DNN 0.75"]

    rc('text', usetex=False)

    sns.set(font_scale=0.7,style='white')

    fig, ax = plt.subplots()


    d_style = {}
    for i in algs:
        d_style[i]=''

    h = sns.lineplot(x="Estimation Error",y="Emperical Performance Ratio", hue="Class", data=df_main1,style="Class",linewidth=1, palette=['black', 'g','r'],
        markers=True, dashes=d_style, markersize=8, err_style="band", err_kws={'alpha':0.1}) #NEWALGALPHARAND
    h.set_xticks(parameters) # <--- set the ticks first
    h.set_ylabel("Emperical Performance Ratio")
    h.set_xticklabels(['-30%','-20%','-10%','0','10%','20%','30%'])
    h.set_xlabel(r'$\epsilon$')

    handles, labels = ax.get_legend_handles_labels()
    labels[0] = r'$\gamma$'+'$_\mathregular{min} = 0.25$'
    labels[1] = r'$\gamma$'+'$_\mathregular{min} = 0.5$'
    labels[2] = r'$\gamma$'+'$_\mathregular{min} = 0.75$'
    ax.legend(handles=handles[0:], labels=labels[0:])

    plt.savefig("plots/"+model+"/"+model+"_estimation.pdf", bbox_inches='tight')