import os
import sys
import shutil
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table
from tabulate import tabulate

# افزودن پوشه والد به sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from lib.config.default import cfg, update_config_from_file
from lib.models.siamtpn.track import build_network

def clear_and_make_directory(path):
    """پاک کردن و ساخت دوباره دایرکتوری"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_model_details(model, filepath):
    """ذخیره جزئیات مدل در یک فایل متنی"""
    with open(filepath, 'w') as file:
        # محاسبه تعداد پارامترهای هر لایه
        params = parameter_count_table(model)
        file.write("Parameter Count per Module:\n")
        file.write("--------------------------------------------------------------------------------\n")
        file.write(params)
        file.write("\n--------------------------------------------------------------------------------\n")

def save_flops_details(model, inputs, filepath):
    """محاسبه و ذخیره FLOPs هر لایه"""
    flops = FlopCountAnalysis(model, inputs)
    flop_table = flop_count_table(flops, max_depth=4)  # تنظیم عمق نمایش به دلخواه
    with open(filepath, 'w') as file:
        file.write("FLOPs per Module:\n")
        file.write("--------------------------------------------------------------------------------\n")
        file.write(flop_table)
        file.write("\n--------------------------------------------------------------------------------\n")

def save_detailed_report(model, inputs, filepath):
    """ذخیره گزارش جزئیات FLOPs و پارامترها در قالب جدول"""
    # محاسبه FLOPs
    flops = FlopCountAnalysis(model, inputs)
    # دریافت اطلاعات FLOPs به صورت دیکشنری
    flops_per_module = flops.by_module()
    # محاسبه تعداد پارامترها
    params = {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
              for name, module in model.named_modules()}
    # آماده‌سازی داده‌ها برای جدول
    table_data = []
    for name in flops_per_module.keys():
        flops_value = flops_per_module[name]
        params_value = params.get(name, 0)
        flops_str = f"{flops_value / 1e6:.3f}M"
        params_str = f"{params_value / 1e3:.3f}K" if params_value < 1e6 else f"{params_value / 1e6:.3f}M"
        table_data.append([name, params_str, flops_str])

    # مرتب‌سازی داده‌ها بر اساس نام لایه
    table_data.sort(key=lambda x: x[0])

    # ساخت جدول
    table = tabulate(table_data, headers=["Module", "#Parameters", "#FLOPs"], tablefmt="github")

    # ذخیره جدول در فایل
    with open(filepath, 'w') as file:
        file.write(table)

def model_detail(local_rank=-1, save_dir=None, base_seed=None):
    # تنظیم مسیر مطلق برای فایل yaml
    yaml_file_path = '/home/ardi/Desktop/project/SiamTPNTracker/experiments/shufflenet_l345_192.yaml'

    # به‌روزرسانی تنظیمات با فایل config
    update_config_from_file(cfg, yaml_file_path)
    
    # ساخت مدل شبکه
    net = build_network(cfg)
    net.eval()
    net.cuda()  # انتقال مدل به CUDA

    # آماده‌سازی دایرکتوری ذخیره‌سازی
    clear_and_make_directory(save_dir)

    # ساخت ورودی‌های فرضی با ابعاد مورد نیاز مدل
# ساخت ورودی‌های فرضی با ابعاد مورد نیاز مدل
    train_input = torch.randn(1, 3, 128, 128).cuda() # انتقال ورودی‌ها به CUDA
                # torch.randn(1, 3, 224, 224).cuda(),
                # torch.randn(1, 3, 224, 224).cuda(),
                # torch.randn(1, 3, 224, 224).cuda(),
                # torch.randn(1, 3, 224, 224).cuda()]  # ورودی آموزش فرضی
    test_input = torch.randn(1, 3, 256, 256).cuda()  # ورودی تست فرضی
    inputs = (train_input, test_input)


    # تعیین مسیر فایل‌ها
    model_flops_path = os.path.join(os.path.abspath(save_dir), "model_flops.txt")
    detailed_report_path = os.path.join(os.path.abspath(save_dir), "detailed_report.txt")

    if local_rank in [-1, 0]:  # ذخیره فقط در فرآیند اصلی
        # ذخیره تعداد پارامترهای هر لایه
        # ذخیره FLOPs هر لایه
        save_flops_details(net, inputs, model_flops_path)
        # ذخیره گزارش جزئیات
        save_detailed_report(net, inputs, detailed_report_path)
# اجرای تابع اصلی
if __name__ == "__main__":
    model_detail(local_rank=-1, save_dir="./results/model_detail_flops")
