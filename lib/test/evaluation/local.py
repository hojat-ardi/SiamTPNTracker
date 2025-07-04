from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.got10k_path = '/mnt/DATA/datasets/got-10k'
    settings.save_dir = './results/'
    settings.got_packed_results_path = './results/'
    settings.got_reports_path = './results/'
    settings.lasot_path = '/mnt/DATA/datasets/lasot'
    settings.otb_path = '/home/ardi/Desktop/dataset/otb100/raw'
    settings.result_plot_path = './results/result_plots/'
    settings.results_path = './results/tracking_results/uav123' 
    settings.trackingnet_path = ''
    settings.uav_path = '/home/ardi/Desktop/dataset/UAV123'
    settings.uav123_10fps_path = '/home/ardi/Desktop/dataset/UAV123_10fps'
    settings.vot_path = ''
    settings.uav20l_path = '/home/ardi/Desktop/dataset/UAV123'
    settings.visdrone_path = '/home/ardi/Desktop/dataset/visdrone/VisDrone2019-SOT-test-dev'
    settings.dtb70_path = '/home/ardi/Desktop/dataset/DTB70'
    settings.uavtrack_path = '/home/ardi/Desktop/dataset/UAVTrack112'
    settings.uavdt_path = "/home/ardi/Desktop/dataset/uavdt/sot"
    settings.uavtrackl_path = '/home/ardi/Desktop/dataset/UAVTrack112l'

    

    return settings


    
