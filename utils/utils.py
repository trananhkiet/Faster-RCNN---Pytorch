from gpuinfo import GPUInfo
import numpy as np
import sys
import os


def set_GPU(num_of_GPUs):
    # default_gpu = "-1"
    # print("[INFO] ***********************************************")
    # print(f"[INFO] You are using GPU(s): {default_gpu}")
    # print("[INFO] ***********************************************")
    # os.environ["CUDA_VISIBLE_DEVICES"] = default_gpu
    current_memory_gpu = GPUInfo.gpu_usage()[1]
    list_available_gpu = np.where(np.array(current_memory_gpu) < 1500)[0].astype('str').tolist()
    current_available_gpu = ",".join(list_available_gpu)
    # print(list_available_gpu)
    # print(current_available_gpu)
    # print(num_of_GPUs)
    if len(list_available_gpu) < num_of_GPUs:
        print("==============Warning==============")
        print("Your process had been terminated")
        print("Please decrease number of gpus you using")
        print(f"number of Devices available:\t{len(list_available_gpu)} gpu(s)")
        print(f"number of Device will use:\t{num_of_GPUs} gpu(s)")
        sys.exit()
    elif len(list_available_gpu) > num_of_GPUs and num_of_GPUs != 0:
        # redundant_gpu = len(list_available_gpu) - num_of_GPUs
        list_available_gpu = list_available_gpu[:num_of_GPUs]
        current_available_gpu = ",".join(list_available_gpu)
    elif num_of_GPUs == 0 or len(list_available_gpu)==0:
        current_available_gpu = "-1"

    print("[INFO] ***********************************************")
    print(f"[INFO] You are using GPU(s): {current_available_gpu}")
    print("[INFO] ***********************************************")
    os.environ["CUDA_VISIBLE_DEVICES"] = current_available_gpu