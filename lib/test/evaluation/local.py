from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.got10k_path = '/mnt/DATA/datasets/got-10k'
    settings.save_dir = './results/'
    settings.got_packed_results_path = './results/'
    settings.got_reports_path = './results/'
    settings.lasot_path = ''
    settings.otb_path = ''
    settings.result_plot_path = './results/result_plots/'
    settings.results_path = './results/tracking_results' 
    settings.trackingnet_path = ''
    settings.uav_path ='/mnt/DATA/datasets/UAV123'
    settings.vot_path = ''

    return settings

