
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/ardi/project/SiamTPNTracker/results'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/mnt/DATA/datasets/lasot'
        self.got10k_dir = '/mnt/DATA/datasets/got-10k/train'
        self.trackingnet_dir = '/mnt/DATA/datasets/TrackingNet'
        self.coco_dir = '/mnt/DATA/datasets/COCO2017/coco/coco'
        

                

