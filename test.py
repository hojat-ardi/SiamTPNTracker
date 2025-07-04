import os
import shutil

# مسیر پوشه اصلی که فایل‌ها در آن قرار دارند
source_dir = "/home/ardi/Desktop/dataset/UAV123/anno/UAV123/att"
# مسیر پوشه جدید که فایل‌ها در آن ذخیره خواهند شد
destination_dir = "/home/ardi/Desktop/dataset/UAV123/anno/UAV123/att2"

# اگر پوشه مقصد وجود ندارد، آن را ایجاد کنید
os.makedirs(destination_dir, exist_ok=True)

# پیمایش در فایل‌های پوشه مبدأ
for filename in os.listdir(source_dir):
    if filename.endswith(".txt"):  # بررسی اینکه فایل txt باشد
        # اضافه کردن 'uav_' به ابتدای نام فایل
        new_filename = f"uav_{filename}"
        # مسیر فایل مبدأ و مقصد
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(destination_dir, new_filename)
        # کپی فایل به مسیر جدید با نام تغییر یافته
        shutil.copy(src_path, dest_path)

print("تمام فایل‌ها با نام جدید ذخیره شدند.")
