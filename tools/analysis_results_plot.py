from PIL.Image import merge
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import sys
sys.path.append('./')

from lib.test.analysis.plot_results_plot import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

# به جای استفاده از argparse، مستقیماً مقادیر را اینجا مشخص کنید:
trackers_names = ["siamtpn", "siamtpn","siamtpn", "siamtpn","siamtpn"
                  ,"siamtpn", "siamtpn","siamtpn", "siamtpn","siamtpn"
                  ,"siamtpn", "siamtpn","siamtpn", "siamtpn"]
params_names = ["SiamTPN","T-SiamTPN","Hift","FDNT","LPAT",
                "PRL_Track","SGDViT","SiamAPN","SiamAPN++","SiamSA",
                "UDAT_BAN","UDAT_CAR","TCTrack","TCTrack++"]
# trackers_names = ["siamtpn", "siamtpn","siamtpn","siamtpn"]
# params_names = ["base","base_temporal","base_temporal_CA","T-SiamTPN"]

runid = None
dataset_name = "uav"
epoch = 100

if len(trackers_names) != len(params_names):
    raise ValueError("Number of trackers and params must match.")

trackers = []

params_dict = {
    'checkpoint': epoch,
    'windows_factor': 0.5,
    'interval': 25,
    'debug': 0,
    'cpu': 0
}

for name, param in zip(trackers_names, params_names):
    trackers.extend(trackerlist(
        name=name,
        parameter_name=param,
        dataset_name=dataset_name,
        run_ids=runid,
        display_name=param,
        params_dict=params_dict
    ))

if "got10k" in dataset_name:
    report_name = 'got10k'
else:
    report_name = dataset_name

merge_results = False
dataset = get_dataset(dataset_name)
plot_results(
    trackers, dataset, report_name,
    merge_results=merge_results,
    plot_types=('success', 'norm_prec' , "prec"),
    skip_missing_seq=False,
    force_evaluation=True,
    plot_bin_gap=0.05
)
print_results(
    trackers, dataset, report_name,
    merge_results=merge_results,
    plot_types=('success', 'prec', 'norm_prec')
)
print_per_sequence_results(trackers, dataset, report_name, merge_results=merge_results)



import os
import numpy as np
from pathlib import Path
import sys

# att_path = "/home/ardi/Desktop/dataset/UAV123/anno/UAV123/att"
att_path = Path("/home/ardi/Desktop/dataset/UAV123/anno/att2")  # تبدیل رشته به Path

num_attr = 12  # تعداد Attributeها

att_all = np.zeros((len(dataset), num_attr))

for i, seq in enumerate(dataset):
    att_file = att_path / f"{seq.name}.txt"
    if not att_file.is_file():
        raise FileNotFoundError(f"Attribute file {att_file} not found!")
    seq_att = np.genfromtxt(att_file, delimiter=',')
    if seq_att.size != num_attr:
        raise ValueError(f"Attribute file {att_file} should have {num_attr} values.")
    att_all[i, :] = seq_att

# اکنون att_all آرایه ویژگی‌ها برای همه توالی‌ها داریم.
# شما می‌توانید این att_all را pickle کنید یا به صورت npy ذخیره کنید
# تا در code3 دوباره لود کنید.
import pickle
with open('att_all.pkl', 'wb') as f:
    pickle.dump(att_all, f)

print("Attribute data saved in att_all.pkl")



