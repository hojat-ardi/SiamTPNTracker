from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.got10k_path = '/home/ardi/Desktop/Dataset/GOT-10k/got-10k'
    settings.save_dir = './results/'
    settings.got_packed_results_path = './results/'
    settings.got_reports_path = './results/'
    settings.lasot_path = ''
    settings.otb_path = ''
    settings.result_plot_path = './results/result_plots/'
    settings.results_path = '/home/ardi/Desktop/project/SiamTPNTracker/results/UAV123' 
    settings.trackingnet_path = ''
    settings.uav_path ='/home/ardi/Desktop/Dataset/uav123/Dataset_UAV123/UAV123'
    settings.vot_path = ''

    return settings

