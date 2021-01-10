import json
import os
from shutil import copyfile


TRAIN = "train"
VAL = "val"
PROJ_PATH = r"C:\Users\MohammadAbid\Documents\Excellenteam\MobilEye"
TO_SAVE = f"{PROJ_PATH}/Dir_data"
def make_empty_directories(directory_name, project_path):
    with os.scandir(f"{project_path}/gtFine/{directory_name}") as entries:
        for entry in entries:
            os.mkdir(f"{TO_SAVE}/{directory_name}/{entry.name}")
def move_json_and_label_files_to_directory(directory_name, project_path):
    with os.scandir(f"{project_path}/gtFine/{directory_name}") as entries:
        for entry in entries:
            with os.scandir(entry.path) as files:
                move_files(files, entry, directory_name, TO_SAVE)
def move_files(files, entry, directory_name, project_path):
    for file in files:
        if "labelIds" in file.name:
            label_path = file.path
            name = file.name
        if ".json" in file.name:
            with open(file.path) as json_file:
                json_object = json.load(json_file)
                objs = json_object["objects"]
            for dict_ in objs:
                if dict_["label"] == "traffic light":
                    file_path_json = f"{project_path}/{directory_name}/{entry.name}/{file.name}"
                    file_path_label = f"{project_path}/{directory_name}/{entry.name}/{name}"
                    copyfile(file.path, file_path_json)
                    copyfile(label_path, file_path_label)
                    break
def move_pictures_to_directory(directory_name, project_path):
    to_save_entries = os.listdir(f"{project_path}/leftImg8bit/{directory_name}")
    entries = os.listdir(f"{TO_SAVE}/{directory_name}")
    for to_save in to_save_entries:
        for entry in entries:
            if entry == to_save:
                pic_files = os.listdir(f"{project_path}/leftImg8bit/{directory_name}/{to_save}")
                files = os.listdir(f"{TO_SAVE}/{directory_name}/{entry}")
                for file in files:
                    for pic in pic_files:
                        if check_file(pic, file):
                            file_path_pic = f"{TO_SAVE}/{directory_name}/{entry}/{pic}"
                            copyfile(f"{project_path}/leftImg8bit/{directory_name}/{to_save}/{pic}", file_path_pic)
                            break
def check_file(pic, file):
    return "labelIds" in file and pic.split("_")[0] == file.split("_")[0] and pic.split("_")[1] == file.split("_")[1] \
           and pic.split("_")[2] == file.split("_")[2]
def mkdir_train_and_val(project_path, directory_name):
    os.mkdir(f"{project_path}/{directory_name}")
if not os.path.isdir(TO_SAVE):
    os.mkdir(TO_SAVE)
mkdir_train_and_val(TO_SAVE, TRAIN)
mkdir_train_and_val(TO_SAVE, VAL)
make_empty_directories(TRAIN, PROJ_PATH)
move_json_and_label_files_to_directory(TRAIN, PROJ_PATH)
make_empty_directories(VAL, PROJ_PATH)
move_json_and_label_files_to_directory(VAL, PROJ_PATH)
move_pictures_to_directory(TRAIN, PROJ_PATH)
move_pictures_to_directory(VAL, PROJ_PATH)