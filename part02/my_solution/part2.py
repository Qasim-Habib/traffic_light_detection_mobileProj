
from part02.my_solution import tfl_detection as det
from part02.my_solution import resources as src
import matplotlib.pyplot as plt
from part02.my_solution.jayouisy_alg import export_learn_data
import numpy as np



def get_tfl_objects( objects ):
    tfl_objects = []
    for obj in objects:
        if obj is not None and obj["label"] == "traffic light":
            tfl_objects.append( obj )
    return tfl_objects

def load_tfl_from_json( json_path ):
    try:
        with open( json_path ) as f:
            data = det.json.load(f)
            objects = data["objects"]
            tfl_objects = get_tfl_objects( objects )
            return tfl_objects
    except:
        return None

def load_image( image_path ):
    label_image = det.np.array( det.Image.open(image_path) )
    label_image = label_image.astype( det.np.uint8)
    return label_image

def load_all_tfl_objects_with_19( resource ):
    res_from_label = []
    tfl_objects = load_tfl_from_json( resource["json"] )
    if tfl_objects is not None:
        label_image = load_image( resource["label_img"] )
        for obj in tfl_objects:
            traffic_light=[]
            for point in obj["polygon"]:
                traffic_light.append(point)
            res_from_label.append(traffic_light)
    return res_from_label

def tfl_detect(image_path):
    red_x, red_y, green_x, green_y = det.find_tfl_lights(image_path, some_threshold=42)
    res=[]
    for i in range(len(red_x)):
        point = [red_x[i],red_y[i]]
        res.append(point)
    for i in range(len(green_x)):
        point=[green_x[i],green_y[i]]
        res.append(point)

    return res

def get_min_max_point(label_traffic):
    min_point=None
    max_point=None
    for point in label_traffic:
        if min_point == None and max_point == None:
            min_point=point
            max_point=point
        else:
            if point[0] > max_point[0] and point[1] > max_point[1]:
                max_point=point
            elif point[0] < max_point[0] and point[1] < max_point[1]:
                min_point=point
    return min_point,max_point

def is_valid_traffic_light(img,trafic_light):
    for point in trafic_light:
        if img[point[1],point[0]] == 19:
            return True
    return False

def is_point_in_square(limit_list,point):
    if point[0] >= limit_list[0][0] and point[1] >= limit_list[0][1]:
        if point[0] <= limit_list[1][0] and point[1] <= limit_list[1][1]:
            return True
    return False

def get_name_file(file_name):
    res=""
    filename=file_name.split("_")
    for i in range(3):
        if i > 0:
            res+="_"
        res+=filename[i]
    if filename[3] == "leftImg8bit.png":
        return res,"img"
    elif filename[4] == "polygons.json":
        return res,"json"
    elif filename[4] == "labelIds.png":
        return res,"label_img"


def make_imagesP02( resource ):
    points_from_label=load_all_tfl_objects_with_19( resource )
    img_label=load_image(resource[ "label_img"])
    src_image = load_image(resource["img"])
    points_from_detect=tfl_detect(src_image)
    valid_list=[]
    limit_list=[]
    for trafic_light in points_from_label:
        valid_list.append(is_valid_traffic_light(img_label,trafic_light))
        min_point,max_point=get_min_max_point(trafic_light)
        limit_list.append((min_point,max_point))
    for point in points_from_detect:
        isfound = False
        for i in range(len(points_from_label)):
            if valid_list[i] == True:
                if is_point_in_square(limit_list[i],point) == True:
                    isfound = True
                    trafic_img=det.np.zeros((81,81,3))
                    trafic_img = trafic_img.astype(det.np.uint8)
                    left = max(point[1]-40,0)
                    right = min (point[1]+41,src_image.shape[0])
                    up = max(point[0]-40,0)
                    down = min(point[0]+41,src_image.shape[1])
                    trafic_img[:(right-left),:(down-up)]= src_image[left:right,up:down]
                    export_learn_data("bins", [ 1 ] , trafic_img)
                    # plt.imshow(trafic_img)
                    # plt.show()
        if isfound == False:
            trafic_img=det.np.zeros((81,81,3))
            trafic_img = trafic_img.astype(det.np.uint8)
            left = max(point[1]-40,0)
            right = min (point[1]+41,src_image.shape[0])
            up = max(point[0]-40,0)
            down = min(point[0]+41,src_image.shape[1])
            trafic_img[:(right-left),:(down-up)]= src_image[left:right,up:down]
            export_learn_data("bins", [ 0 ] , trafic_img)
            # plt.imshow(trafic_img)
            # plt.show()
    # plt.imshow(src_image)
    # plt.show()



def export_learn_data_bins(path, file_name, bin_data):
    with open(f'{path}/{file_name}', 'ab') as data_file:
        with open(f'{path}/temp.bin', 'wb') as tmp:
            np.array(bin_data).reshape((-1,)).tofile(tmp)
        with open(f'{path}/temp.bin', 'rb') as tmp:
            data_file.write(tmp.read())


def crop_for_part04( image_id, src_image, points_from_detect ):
    for point in points_from_detect:
        trafic_img=det.np.zeros((81,81,3))
        trafic_img = trafic_img.astype(det.np.uint8)
        left = max(point[1]-40,0)
        right = min (point[1]+41,src_image.shape[0])
        up = max(point[0]-40,0)
        down = min(point[0]+41,src_image.shape[1])
        trafic_img[:(right-left),:(down-up)]= src_image[left:right,up:down]
        export_learn_data_bins("bins", "data" + image_id + ".bin" , trafic_img)



if __name__ == "__main__":
    # data={}
    with open( "imges.json" ) as f:
        data = det.json.load(f)
        for key, val in data.items():
            folder_name = val["img"].split("_")[0]
            path = r"C:\Users\MohammadAbid\Documents\Excellenteam\MobilEye\Dir_data\train\{}".format( folder_name )
            path += "\\"
            res = {
                "img": path + val["img"],
                "json": path + val["json"],
                "label_img": path + val["label_img"]
            }
            make_imagesP02( res )
