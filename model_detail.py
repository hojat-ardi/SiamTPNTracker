import os
import sys
import shutil
import torch
sys.path.append('./')
from lib.config.default import cfg, update_config_from_file
from lib.models.siamtpn.track import build_network
from torch.autograd import profiler

def clear_and_make_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_model_details(model, filepath):
    with open(filepath, 'w') as file:
        file.write("Model Details:\n")
        file.write("--------------------------------------------------------------------------------\n")
        total_params = 0
        total_trainable_params = 0
        for name, module in model.named_modules():
            for param_name, parameter in module.named_parameters(recurse=False):
                param = parameter.numel()
                trainable = parameter.requires_grad
                trainability = "Trainable" if trainable else "Frozen"
                file.write(f"{name}.{param_name:30} | Count: {param:10} | {trainability}\n")
                total_params += param
                if trainable:
                    total_trainable_params += param

        file.write("--------------------------------------------------------------------------------\n")
        file.write(f"Total parameters: {total_params}\n")
        file.write(f"Trainable parameters: {total_trainable_params}\n")
        file.write("--------------------------------------------------------------------------------\n")


def save_model_structure(model, filepath):
    with open(filepath, 'w') as file:
        def print_module(module, indent=0):
            for name, submodule in module.named_children():
                file.write(' ' * indent + f"{name}: {submodule.__class__.__name__}\n")
                print_module(submodule, indent + 4)

        file.write("Model Structure:\n")
        file.write("--------------------------------------------------------------------------------\n")
        print_module(model)
        file.write("--------------------------------------------------------------------------------\n")


def profile_model_performance(model, train_input, test_input, filepath):
    with profiler.profile(use_cuda=True, profile_memory=True, record_shapes=True) as prof:
        model(train_input, test_input)
    
    with open(filepath, 'a') as file:  # Append to the existing file
        file.write(prof.key_averages().table(sort_by="cuda_time_total"))
        file.write("\n--------------------------------------------------------------------------------\n")

def model_detail(local_rank=-1, save_dir=None, base_seed=None):
    # Update the default configs with config file
    update_config_from_file(cfg, 'experiments/{}.yaml'.format("shufflenet_l345_192"))
    
    # Create network
    net = build_network(cfg)
    # Wrap networks to distributed one
    net.cuda()

    # Prepare directory
    clear_and_make_directory(save_dir)

    # Save detailed model information to a file
    model_params_path = os.path.join(os.path.abspath(save_dir), "model_params.txt")
    model_structure_path = os.path.join(os.path.abspath(save_dir), "model_structure.txt")
    model_performance_path = os.path.join(os.path.abspath(save_dir), "model_performance.txt")

    if local_rank in [-1, 0]:  # Save only in the main process
        save_model_details(net, model_params_path)
        save_model_structure(net, model_structure_path)

        # Example input for performance profiling, adjust size according to your model input
        dummy_train_input = torch.randn(1, 3, 80, 80).cuda()  # Assume batch of 5 training images
        dummy_test_input = torch.randn(1, 3, 360, 360).cuda()   # Single test image
        profile_model_performance(net, dummy_train_input, dummy_test_input, model_performance_path)

model_detail(local_rank=-1, save_dir="./results/model_detail")

# ---------------------LINK TO /home/ardi/Desktop/project/SiamTPNTracker/lib/models/siamtpn/track.py---------------------
# def test_model():
#     # Update the default configs with config file
#     update_config_from_file(cfg, 'experiments/{}.yaml'.format("shufflenet_l345_192"))

#     # Create and configure the network
#     net = build_network(cfg)
#     net.cuda()

#     # Prepare dummy data
#     dummy_train_input = torch.randn(1, 3, 80, 80).cuda()
#     dummy_test_input = torch.randn(1, 3, 360, 360).cuda()

#     # Try running the forward pass
#     try:
#         output = net(dummy_train_input, dummy_test_input)
#         print("Final output shape:", output.shape)
#     except Exception as e:
#         print("Error during model forward pass:", str(e))

# if __name__ == "__main__":
#     test_model()
# -------------------------------------------------------------------------------------------------------

