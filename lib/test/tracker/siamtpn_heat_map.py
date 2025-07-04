from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import numpy as np
import torch.nn.functional as F
import cv2
import os
from lib.models.siamtpn.track import build_network
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box

class SiamTPN(BaseTracker):
    def __init__(self, params):
        super(SiamTPN, self).__init__(params)
        network = build_network(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        self.cfg = params.cfg
        if params.cpu:
            self.network = network
        else:
            self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor(cpu=params.cpu)
        # for debug
        self.debug = self.params.debug
        self.frame_id = 0
        self.grids = self._generate_anchors(self.cfg.MODEL.ANCHOR.NUM, self.cfg.MODEL.ANCHOR.FACTOR, self.cfg.MODEL.ANCHOR.BIAS)
        self.window = self._hanning_window(self.cfg.MODEL.ANCHOR.NUM)
        self.hanning_factor = self.cfg.TEST.HANNING_FACTOR
        self.feat_sz_tar = self.cfg.MODEL.ANCHOR.NUM

    def initialize(self, image, info: dict):
        gt_box = torch.tensor(info['init_bbox'])
        z_patch_arr, _, z_amask_arr = sample_target(image, gt_box, self.params.template_factor,
                                                   output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            tem_feat = self.network.backbone(template)
            self.tem_feat = self.network.fpn(tem_feat)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor, output_sz=self.params.search_size
        )  # (x1, y1, w, h)

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            tar_feat = self.network.backbone(search)
            tar_feat = self.network.fpn(tar_feat)
            raw_scores, boxes = self.network.head(tar_feat, self.tem_feat)
            raw_scores = raw_scores.cpu()  # B, L, 2
            boxes = boxes.cpu()
            pred_boxes = boxes.reshape(-1, 4)
            lt = self.grids[:, :2] - pred_boxes[:, :2]
            rb = self.grids[:, :2] + pred_boxes[:, 2:]
            pred_boxes = torch.cat([lt, rb], -1).view(-1, 4)
            raw_scores = F.softmax(raw_scores, -1)[:, 1].view(self.feat_sz_tar, self.feat_sz_tar)
            raw_scores = raw_scores * (1 - self.hanning_factor) + self.hanning_factor * self.window
            max_v, ind = raw_scores.view(-1).topk(1)
            pred_box = pred_boxes[ind, :]  
        pred_box = (pred_box.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        # نگاشت هیت‌مپ به مختصات تصویر اصلی
        heatmap = raw_scores.numpy()  # هیت‌مپ خام
        full_heatmap = self.map_heatmap_back(heatmap, resize_factor, (H, W))
        # نرمال‌سازی هیت‌مپ نهایی
        if np.max(full_heatmap) > 0:
            heatmap_normalized = np.uint8(255 * full_heatmap / np.max(full_heatmap))  # نرمال‌سازی به [0, 255]
        else:
            heatmap_normalized = np.zeros_like(full_heatmap, dtype=np.uint8)

        # تبدیل هیت‌مپ به رنگ
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # ترکیب هیت‌مپ رنگی با تصویر اصلی
        image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # تبدیل تصویر به BGR برای OpenCV
        overlay = cv2.addWeighted(image_BGR, 0.6, heatmap_colored, 0.4, 0)

        # ذخیره تصویر ترکیبی
        output_dir = "/home/ardi/Desktop/project/SiamTPNTracker/results/tracking_results/map/siamtpn/shufflenet_l345_192/heatmap"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"frame_{self.frame_id:04d}.png")
        cv2.imwrite(output_path, overlay)  # ذخیره تصویر ترکیبی

        if self.debug:
            x1, y1, w, h = self.state
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=5)
            cv2.imshow("demo", image_BGR)
            key = cv2.waitKey(1)
            if key == ord('p'):
                cv2.waitKey(-1)

        # گرفتن نتیجه نهایی جعبه
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        # مرکز باکس قبلی
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        
        # استخراج مختصات باکس پیش‌بینی شده
        x1, y1, x2, y2 = pred_box
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        
        # محاسبه نیم‌عرض منطقه جستجو در مقیاس اصلی
        half_side = 0.5 * self.params.search_size / resize_factor
        
        # تبدیل مرکز باکس پیش‌بینی شده به مختصات واقعی
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        
        # تبدیل به فرمت [x, y, w, h] با مختصات واقعی
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_heatmap_back(self, heatmap: np.ndarray, resize_factor: float, image_size: tuple):
        """
        نگاشت هیت‌مپ رزایز شده به مختصات تصویر اصلی.
        
        Args:
            heatmap (np.ndarray): هیت‌مپ خام.
            resize_factor (float): فاکتور تغییر اندازه.
            image_size (tuple): ابعاد تصویر اصلی به صورت (H, W).
        
        Returns:
            np.ndarray: هیت‌مپ کامل با ابعاد تصویر اصلی.
        """
        H, W = image_size
        search_size = self.params.search_size

        # محاسبه نیم‌عرض جعبه جستجو در تصویر اصلی
        half_side = 0.5 * search_size / resize_factor

        # مرکز باکس قبلی
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]

        # موقعیت باکس جستجو در تصویر اصلی
        x1 = int(cx_prev - half_side)
        y1 = int(cy_prev - half_side)
        x2 = int(cx_prev + half_side)
        y2 = int(cy_prev + half_side)

        # اطمینان از اینکه موقعیت‌ها خارج از محدوده تصویر نباشند
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        # اندازه نهایی هیت‌مپ رزایز شده در تصویر اصلی
        heatmap_height = y2 - y1
        heatmap_width = x2 - x1

        # رزایز هیت‌مپ به اندازه باکس جستجو در تصویر اصلی
        heatmap_resized = cv2.resize(heatmap, (heatmap_width, heatmap_height), interpolation=cv2.INTER_LINEAR)

        # ایجاد هیت‌مپ کامل با ابعاد تصویر اصلی
        full_heatmap = np.zeros((H, W), dtype=np.float32)

        # قرار دادن هیت‌مپ رزایز شده در موقعیت صحیح
        full_heatmap[y1:y2, x1:x2] = heatmap_resized

        return full_heatmap

    def _hanning_window(self, num):
        hanning = np.hanning(num)
        window = np.outer(hanning, hanning)
        window = torch.from_numpy(window)
        return window

    def _generate_anchors(self, num=20, factor=1, bias=0.5):
        """
        generate anchors for each sampled point
        """
        x = np.arange(num)
        y = np.arange(num)
        xx, yy = np.meshgrid(x, y) 
        xx = (factor * xx + bias) / num 
        yy = (factor * yy + bias) / num
        xx = torch.from_numpy(xx).view(-1).float()
        yy = torch.from_numpy(yy).view(-1).float()
        grids = torch.stack([xx, yy], -1)  # N 2
        return grids

def get_tracker_class():
    return SiamTPN
