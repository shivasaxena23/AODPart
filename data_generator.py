import numpy as np


def initialize(model, exits):
    
    if model == "resnet18":    
        with open('data/resnet18/resnet18_remote_processing_latencies.npy', 'rb') as f:
            DNN_compute_values = np.load(f,allow_pickle=True)

        with open('data/resnet18/resnet18_prediction_results.npy', 'rb') as f:
            DNN_prediction_values = np.load(f,allow_pickle=True)
        
        with open('data/resnet18/resnet18_data_sizes.npy', 'rb') as f:
            input_data_real = np.load(f,allow_pickle=True)

        first = np.concatenate((np.full(1000, True),np.full(9000, False)))
        np.random.shuffle(first)
        pred_arr = DNN_prediction_values

        if exits == "trained":
            with open('data/resnet18/resnet18_exit_accuracies.npy', 'rb') as f:
                accuracies = np.load(f,allow_pickle=True)
            pred_bin = np.stack([first, first, pred_arr[:,0], pred_arr[:,0], pred_arr[:,1], pred_arr[:,1], pred_arr[:,2], pred_arr[:,2], pred_arr[:,3], pred_arr[:,3], pred_arr[:,4], pred_arr[:,4], pred_arr[:,5], pred_arr[:,5], pred_arr[:,6], pred_arr[:,6], pred_arr[:,7], pred_arr[:,7], pred_arr[:,8]], axis=1)
        else:
            accuracies = [10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80, 80, 90, 90, 95]
            array = []
            for i in range(len(accuracies)):
                first = np.concatenate((np.full(int(10000*round(accuracies[i]/100,3)), True),np.full(int(10000*round(1- accuracies[i]/100,3)), False)))
                np.random.shuffle(first)
                array.append(first)
            array = np.array(array)
            pred_bin = array.T
    else:
        with open('data/alexnet/alexnet_remote_processing_latencies.npy', 'rb') as f:
            DNN_compute_values = np.load(f,allow_pickle=True)

        with open('data/alexnet/alexnet_prediction_results.npy', 'rb') as f:
            DNN_prediction_values = np.load(f,allow_pickle=True)
        
        with open('data/alexnet/alexnet_data_sizes.npy', 'rb') as f:
            input_data_real = np.load(f,allow_pickle=True)

        first = np.concatenate((np.full(1000, True),np.full(9000, False)))
        np.random.shuffle(first)
        pred_arr = DNN_prediction_values

        if exits == "trained":
            with open('data/alexnet/alexnet_exit_accuracies.npy', 'rb') as f:
                accuracies = np.load(f,allow_pickle=True)
            pred_bin = np.stack([first, pred_arr[:,0], pred_arr[:,1], pred_arr[:,2], pred_arr[:,3], pred_arr[:,4], pred_arr[:,4], pred_arr[:,5]], axis=1)
        else:
            accuracies = [10, 30, 40, 50, 60, 70, 70, 75]
            array = []
            for i in range(len(accuracies)):
                first = np.concatenate((np.full(int(10000*round(accuracies[i]/100,3)), True),np.full(int(10000*round(1- accuracies[i]/100,3)), False)))
                np.random.shuffle(first)
                array.append(first)
            array = np.array(array)
            pred_bin = array.T        
        
    return DNN_compute_values, DNN_prediction_values, input_data_real, pred_bin, accuracies

def get_comms(proc_remote,input_data_real,L):
    bws= np.random.uniform(0.10,0.90,len(proc_remote))*L
    comms = input_data_real/bws
    comms = comms.tolist()
    comms.append(0)
    return comms




