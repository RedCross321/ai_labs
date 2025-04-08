import os
import shutil
import random


data_dir = 'data6'
tr_dir = 'tr'
val_dir = 'val'

tr_my_func = os.path.join(tr_dir, 'my_func')
tr_not_my_func = os.path.join(tr_dir, 'not_my_func')
val_my_func = os.path.join(val_dir, 'my_func')
val_not_my_func = os.path.join(val_dir, 'not_my_func')


os.makedirs(tr_my_func, exist_ok=True)
os.makedirs(tr_not_my_func, exist_ok=True)
os.makedirs(val_my_func, exist_ok=True)
os.makedirs(val_not_my_func, exist_ok=True)

for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    random.shuffle(files)

    train_files = files[:700]
    val_files = files[700:1000]

    if folder_name == '2':
        tr_target_dir = tr_my_func
        val_target_dir = val_my_func
    else:
        tr_target_dir = tr_not_my_func
        val_target_dir = val_not_my_func

    for filename in train_files:
        source = os.path.join(folder_path, filename)
        dest_name = f"{folder_name}_{filename}"
        destination = os.path.join(tr_target_dir, dest_name)
        shutil.copy2(source, destination)

    for filename in val_files:
        source = os.path.join(folder_path, filename)
        dest_name = f"{folder_name}_{filename}"
        destination = os.path.join(val_target_dir, dest_name)
        shutil.copy2(source, destination)

for directory in [tr_my_func, tr_not_my_func, val_my_func, val_not_my_func]:
    files_count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    print(f"{directory}: {files_count} файлов")

