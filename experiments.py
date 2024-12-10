import argparse
from data_generator import *
from R import plot as plot_R
from gamma_min import plot as plot_gamma_min
from delta import plot as plot_delta
from estimated import plot as plot_est


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="resnet18/alexnet")
parser.add_argument("--mode", help="trained/smooth/error")
args = parser.parse_args()

if args.mode != "trained" and args.mode != "smooth" and args.mode !=  "error":
    parser.print_help()
    exit(1)

if args.model != "resnet18" and args.model != "alexnet":
    parser.print_help()
    exit(1)

if args.mode == "error":
    DNN_compute_values, DNN_prediction_values, input_data_real, pred_bin, accuracies = initialize(args.model,"trained")
    plot_est(DNN_compute_values, DNN_prediction_values, input_data_real, pred_bin, accuracies, args.model)
else:
    improvement_portend_total = 0
    improvement_SPINN_total = 0
    DNN_compute_values, DNN_prediction_values, input_data_real, pred_bin, accuracies = initialize(args.model,args.mode)
    
    (improvement_portend, improvement_SPINN) = plot_R(DNN_compute_values, DNN_prediction_values, input_data_real, pred_bin, accuracies, args.model, args.mode)
    improvement_portend_total += improvement_portend 
    improvement_SPINN_total += improvement_SPINN
    
    (improvement_portend, improvement_SPINN) = plot_gamma_min(DNN_compute_values, DNN_prediction_values, input_data_real, pred_bin, accuracies, args.model, args.mode)
    improvement_portend_total += improvement_portend 
    improvement_SPINN_total += improvement_SPINN
    
    (improvement_portend, improvement_SPINN) = plot_delta(DNN_compute_values, DNN_prediction_values, input_data_real, pred_bin, accuracies, args.model, args.mode)
    improvement_portend_total += improvement_portend 
    improvement_SPINN_total += improvement_SPINN
    
    print("For "+args.model+" with "+args.mode+" exits, AODPart provides %.2f accuracy improvement over Portend and %.2f accuracy improvement over SPINN."%(improvement_portend_total/3,improvement_SPINN_total/3))