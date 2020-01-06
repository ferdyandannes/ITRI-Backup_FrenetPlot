import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

import h5py
import numpy as np
import os
import copy
import operator
from PIL import Image

import matplotlib.pyplot as plt

my_dpi = 100

def check_dir(dir_list):
    for d in dir_list:
        if not os.path.isdir(d):
            print('Create directory :\n' + d)
            os.makedirs(d)

def set_yticks_label():
    label_list =[]
    for i in range(0,61):
        if i%5==0:
            label_list.append(i)
        else:
            label_list.append("")

    return label_list

def set_yticks(bird_hight,r_hight,y_range,tick):
    tick_list = []    
    for i in range(0,tick):
        tick_list.append( (600-y_range*i)*r_hight)
    
    return tick_list

def get_transform_matrix(src, dst):
    # Get the transform matrix by OpenCV function. 
    # src : Coordinates of quadrangle vertices in the source image.
    # dst : Coordinates of the corresponding quadrangle vertices in the destination image.
    # return : Transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    return M

def draw_grid(grid):
    # input(grid) : A list of grid
    # return(img_bird_grid) : A bird view image of grids. More than one car in a grid, color the grid with red.
    #                         One car in a grid, color the grid with gray.
    img_bird_grid = np.ones((600, 175, 3), dtype=np.uint8)*255
    box_color = (200,200,200)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1 or grid[i][j] == 3:
                cv2.rectangle(img_bird_grid,(int(j*175/10), int(i*600/60)),
                              (int((j+1)*175/10), int((i+1)*600/60)),box_color,-1)
            elif grid[i][j] == 2 :
                cv2.rectangle(img_bird_grid,(int(j*175/10), int(i*600/60)),
                              (int((j+1)*175/10), int((i+1)*600/60)),(0,255,255),-1)
            elif grid[i][j] > 3:
                cv2.rectangle(img_bird_grid,(int(j*175/10), int(i*600/60)),
                              (int((j+1)*175/10), int((i+1)*600/60)),(0,0,255),-1)

    return img_bird_grid

def viz_frenet(data_dir):
    position_path = os.path.join(data_dir,"pseudo.txt")

    with open(position_path) as position:
        position_info = position.readlines()

    lines = len(position_info)

    # Save the position
    BEV_dir = data_dir+'Pseudo_Position/BEVh5/'
    BEVis_dir = data_dir+'Pseudo_Position/BEV/'
    check_dir([BEV_dir, BEVis_dir])

    # Read parameters for transform matrix of front view to bird view.
    with h5py.File(data_dir+'parameters.h5','r') as pf:
        src = pf['src'].value
        bird_hight2 = pf['bird_hight'].value
        bird_width2 = pf['bird_width'].value
        bird_channels = pf['bird_channels'].value

    # Calculate the transform matrix(front view to bird view).
    dst = np.float32([[0, 0], [bird_width2, 0], [0, bird_hight2], [bird_width2, bird_hight2]])
    M = get_transform_matrix(src, dst)

    new_bird_width2, new_bird_hight2 = 175,600

    r_width = new_bird_width2/bird_width2
    r_hight = new_bird_hight2/bird_hight2
    plt.ion()
    fig = plt.figure('Bird View Trajectory',  figsize=(300/my_dpi, 1080/my_dpi))
    tick_list = set_yticks(bird_hight2,r_hight,10,60)## every x pixel a grid total y grids  x * y =500
    y_ticks_label = set_yticks_label()
    plt.yticks(tick_list, y_ticks_label)

    # plt.xticks([17.5,35,52.5,70,87.5],['-3.5',"",'0',"",'3.5'])
    plt.xticks([17.5, 35, 52.5, 70, 87.5, 105, 122.5, 140, 157.5],['-7', "", '-3.5', "", '0', "", '3.5', "", '7'])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().invert_yaxis()
    plt.grid(True)

    for line in range(lines):
        print("Line = ", line)

        info = position_info[line].strip().split()

        nomor_frame = info[0]
        ego_x = float(info[1])
        ego_y = float(info[2])

        pseu_id = int(info[3])
        pseu_x = float(info[4])
        pseu_y = float(info[5])

        
        gap_y = ego_y - pseu_y
        gap_x = ego_x - pseu_x

        # Matplotlib BEV format
        if gap_x >= 0:
            pos_x = -abs(gap_x)
        elif gap_x < 0:
            pos_x = abs(gap_x)

        if gap_y > 0:
            pos_y = -abs(gap_y)
        elif gap_y < 0:
            pos_y = abs(gap_y)

        # Matplotlib ori format
        pos_x = pos_x
        pos_y = 60 - pos_y
        print("pos_x = ", pos_x)
        print("pos_y = ", pos_y)

        # Revert into the correct position
        pos_x = ((175 - 0) / (8.75 + 8.75))*(pos_x - 8.75) + 175
        pos_y = ((600 - 0) / (60 - 0))*(pos_y - 60) + 600
        print("pos_x = ", pos_x)
        print("pos_y = ", pos_y)

        # Plotting
        bxmin = int(pos_x-5)
        bxmax = int(pos_x+5)
        bymin = int(pos_y-10)
        bymax = int(pos_y)

        # draw object position
        grid = np.zeros((60+2,10+2), dtype = int)
        img_bird_grid = draw_grid(grid)
        img_bird = np.ones((bird_hight2, bird_width2, 3), np.uint8)*255
        cv2.rectangle(img_bird,(bxmin, bymin),(bxmax, bymax),(195,6,147),-1)

        img_bird[np.all(img_bird == [255,255,255], axis = -1)] = img_bird_grid[np.all(img_bird == [255,255,255], axis = -1)]
        bird_result = img_bird
        bird_result = cv2.cvtColor(bird_result,cv2.COLOR_BGR2RGB)

        # refresh plt figure
        if line == 0:
            im = plt.imshow(bird_result)
        else:
            im.set_data(bird_result)

        info_file = h5py.File(BEV_dir+nomor_frame+'.h5','w')
        info_file.create_dataset('frame_id', data = nomor_frame)
        info_file.create_dataset('pos_x', data = pos_x)
        info_file.create_dataset('pos_y', data = pos_y)

        print("")
        plt.savefig(BEVis_dir+nomor_frame+'.png')
        plt.pause(0.000001)

def viz_frenet2(data_dir):
    position_path = os.path.join(data_dir,"pseudo.txt")

    with open(position_path) as position:
        position_info = position.readlines()

    lines = len(position_info)

    helper_dir = data_dir+'Helper/img/'
    helper = os.listdir(helper_dir)
    helper.sort()

    helper_dir2 = data_dir+'Helper/grid/'
    helper2 = os.listdir(helper_dir2)
    helper2.sort()

    # Save the position
    BEV_dir = data_dir+'Pseudo_Position/Comb_two/'
    check_dir([BEV_dir])

    # Read parameters for transform matrix of front view to bird view.
    with h5py.File(data_dir+'parameters.h5','r') as pf:
        src = pf['src'].value
        bird_hight2 = pf['bird_hight'].value
        bird_width2 = pf['bird_width'].value
        bird_channels = pf['bird_channels'].value

    # Calculate the transform matrix(front view to bird view).
    dst = np.float32([[0, 0], [bird_width2, 0], [0, bird_hight2], [bird_width2, bird_hight2]])
    M = get_transform_matrix(src, dst)

    new_bird_width2, new_bird_hight2 = 175,600

    r_width = new_bird_width2/bird_width2
    r_hight = new_bird_hight2/bird_hight2
    plt.ion()
    fig = plt.figure('Bird View Trajectory',  figsize=(300/my_dpi, 1080/my_dpi))
    tick_list = set_yticks(bird_hight2,r_hight,10,60)## every x pixel a grid total y grids  x * y =500
    y_ticks_label = set_yticks_label()
    plt.yticks(tick_list, y_ticks_label)

    # plt.xticks([17.5,35,52.5,70,87.5],['-3.5',"",'0',"",'3.5'])
    plt.xticks([17.5, 35, 52.5, 70, 87.5, 105, 122.5, 140, 157.5],['-7', "", '-3.5', "", '0', "", '3.5', "", '7'])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().invert_yaxis()
    plt.grid(True)

    for line in range(lines):
        print("line = ", line)

        info = position_info[line].strip().split()

        nomor_frame = info[0]
        ego_x = float(info[1])
        ego_y = float(info[2])

        pseu_id = int(info[3])
        pseu_x = float(info[4])
        pseu_y = float(info[5])

        
        gap_y = ego_y - pseu_y
        gap_x = ego_x - pseu_x

        # Matplotlib BEV format
        if gap_x >= 0:
            pos_x = -abs(gap_x)
        elif gap_x < 0:
            pos_x = abs(gap_x)

        if gap_y > 0:
            pos_y = -abs(gap_y)
        elif gap_y < 0:
            pos_y = abs(gap_y)

        # Matplotlib ori format
        pos_x = pos_x
        pos_y = 60 - pos_y
        print("pos_x = ", pos_x)
        print("pos_y = ", pos_y)

        # Revert into the correct position
        pos_x = ((175 - 0) / (8.75 + 8.75))*(pos_x - 8.75) + 175
        pos_y = ((600 - 0) / (60 - 0))*(pos_y - 60) + 600
        print("pos_x = ", pos_x)
        print("pos_y = ", pos_y)

        # Plotting
        bxmin = int(pos_x-5)
        bxmax = int(pos_x+5)
        bymin = int(pos_y-10)
        bymax = int(pos_y)

        # draw object position
        grid = np.zeros((60+2,10+2), dtype = int)
        with h5py.File(helper_dir2+nomor_frame+'.h5','r') as pf:
            grid = pf['grid'].value

        img_bird_grid = draw_grid(grid)

        #img_bird = np.ones((bird_hight2, bird_width2, 3), np.uint8)*255
        img_bird = cv2.imread(helper_dir+nomor_frame+'.png')
        cv2.rectangle(img_bird,(bxmin, bymin),(bxmax, bymax),(195,6,147),-1)

        img_bird[np.all(img_bird == [255,255,255], axis = -1)] = img_bird_grid[np.all(img_bird == [255,255,255], axis = -1)]
        bird_result = img_bird
        bird_result = cv2.cvtColor(bird_result,cv2.COLOR_BGR2RGB)

        # refresh plt figure
        if line == 0:
            im = plt.imshow(bird_result)
        else:
            im.set_data(bird_result)

        print("")
        plt.savefig(BEV_dir+nomor_frame+'.png')
        plt.pause(0.000001)

def viz_front(data_dir):
    image_dir = data_dir+'Images/'
    images = os.listdir(image_dir)
    images.sort()

    bev_dir = data_dir+'Pseudo_Position/BEVh5/'
    bev = os.listdir(bev_dir)
    bev.sort()

    with h5py.File(data_dir+'parameters.h5','r') as pf:
        src = pf['src'].value
        bird_hight2 = pf['bird_hight'].value
        bird_width2 = pf['bird_width'].value
        bird_channels = pf['bird_channels'].value

    # Save the position
    front_dir = data_dir+'Pseudo_Position/Front/'
    check_dir([front_dir])

    dst = np.float32([[0, 0], [bird_width2, 0], [0, bird_hight2], [bird_width2, bird_hight2]])
    M = get_transform_matrix(src, dst)

    for frame_id in range(len(images)-1):
        print("frame_id = ", frame_id)
        imgnum = str(frame_id).zfill(4)

        if frame_id == 0:
            continue
        elif frame_id == 1:
            continue
        # Check if file is exist or not
        img_num2 = imgnum.lstrip('0')
        img_num2 = int(img_num2)
        imgnum2 = str(img_num2).zfill(4)
        if os.path.exists(bev_dir+imgnum2+'.h5') == False:
            continue

        # Read RGB Image
        img_path = image_dir+images[frame_id]
        img = cv2.imread(img_path)

        # Warp Perspective
        warped_img = cv2.warpPerspective(img, M, (bird_width2, bird_hight2))

        ###########
        # First Mask
        # Read the path trajectory
        with h5py.File(bev_dir+imgnum2+'.h5','r') as ra:
            pos_x = ra['pos_x'].value
            pos_y = ra['pos_y'].value

        mask = np.zeros((bird_hight2,bird_width2,3), np.uint8)
        cv2.rectangle(mask, (int(pos_x) - 17, int(pos_y) - 30), (int(pos_x) + 17, int(pos_y) + 10), (255,0,0), cv2.FILLED)

        # Inverse Display
        Minv = cv2.getPerspectiveTransform(dst, src)
        inverse_img = cv2.warpPerspective(mask, Minv, (1280, 385))
        cover_image = cv2.addWeighted(img, 1, inverse_img, 0.3, 0)

        fw = cv2.cvtColor(inverse_img, cv2.COLOR_BGR2GRAY)
        _, fw_mask = cv2.threshold(fw, 1, 255, cv2.THRESH_BINARY)
        _, contours, hier = cv2.findContours(fw_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Value for bounding box
        if 0 <= pos_y < 100:
            fw_value = 15
        elif 100 <= pos_y < 200:
            fw_value = 30
        elif 200 <= pos_y < 300:
            fw_value = 45
        elif 300 <= pos_y < 400:
            fw_value = 60
        elif 400 <= pos_y < 500:
            fw_value = 75
        elif 500 <= pos_y < 600:
            fw_value = 90 

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(cover_image,(x,y-fw_value),(x+w,y+h),(0,0,255),1)

        cv2.imshow("cover_image", cover_image)
        cv2.imwrite(front_dir+imgnum2+'.png', cover_image)
        cv2.waitKey(0)

def resize_gambar(data_dir):
    input_dir = os.path.join(data_dir, "Pseudo_Position/BEV/")
    save_dir = os.path.join(data_dir, "Pseudo_Position/BEV_resize/")
    input_files = os.listdir(input_dir)

    # Tambahan aing
    input_files.sort()

    check_dir([save_dir])
    for input_file in input_files:
        print("input_file = ", input_file)
        frame = os.path.join(input_dir, input_file)
        img = cv2.imread(frame)
        resize_img = cv2.resize(img, (320, 1385), interpolation=cv2.INTER_CUBIC)
        text = "Car Coordinate"
        cv2.putText(resize_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, 4)
        save_file = os.path.join(save_dir, input_file)
        cv2.imwrite(save_file,resize_img)

def resize_gambar2(data_dir):
    input_dir = os.path.join(data_dir, "Pseudo_Position/Comb_two/")
    save_dir = os.path.join(data_dir, "Pseudo_Position/Comb_two_resize/")
    input_files = os.listdir(input_dir)

    # Tambahan aing
    input_files.sort()

    check_dir([save_dir])
    for input_file in input_files:
        print("input_file = ", input_file)
        frame = os.path.join(input_dir, input_file)
        img = cv2.imread(frame)
        resize_img = cv2.resize(img, (320, 1385), interpolation=cv2.INTER_CUBIC)
        text = "Car Coordinate"
        cv2.putText(resize_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, 4)
        save_file = os.path.join(save_dir, input_file)
        cv2.imwrite(save_file,resize_img)

def combine1(img1_path, img2_path, save):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    width1, height1 = img1.size
    width2, height2 = img2.size
    width = width1
    height = height1+height2
    com_img = Image.new('YCbCr', (width, height))
    loc1 = (0, 0)
    loc2 = (0, height1)
    com_img.paste(img1, loc1)
    com_img.paste(img2, loc2)
    save_name = save
    com_img.save(save_name,"JPEG")

def combine2(img1_path, img2_path, save):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    width1, height1 = img1.size
    width2, height2 = img2.size
    width = width1+width2
    height = height1
    com_img = Image.new('YCbCr', (width, height))
    loc1 = (0, 0)
    loc2 = (width1, 0)
    com_img.paste(img1, loc1)
    com_img.paste(img2, loc2)
    save_name = save
    com_img.save(save_name, "JPEG")

def combine3(img1_path, img2_path, save):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    width1, height1 = img1.size
    width2, height2 = img2.size
    width = width1+width2
    height = height2
    com_img = Image.new('RGB', (width, height), color = (255, 255, 255))
    #com_img.show()
    loc1 = (0, 300)
    loc2 = (width1, 0)
    com_img.paste(img1, loc1)
    com_img.paste(img2, loc2)
    save_name = save
    com_img.save(save_name, "JPEG")

def combine_semua(data_dir):
    rgb = os.path.join(data_dir,"Images/")
    front = os.path.join(data_dir,"Pseudo_Position/Front/")
    track_bev = os.path.join(data_dir,"Pseudo_Position/Comb_two_resize/")
    pseudo_bev = os.path.join(data_dir,"Pseudo_Position/BEV_resize/")

    save1 = os.path.join(data_dir , "Pseudo_Position/Combine_1/")
    save2 = os.path.join(data_dir , "Pseudo_Position/Combine_2/")
    save3 = os.path.join(data_dir , "Pseudo_Position/Combine/")
    check_dir([save1, save2, save3])

    front_files = os.listdir(front)
    front_files.sort()

    for front_file in front_files:
        print("front_file = ", front_file)
        nama_file = front_file.split('.')
        img1 = os.path.join(rgb, front_file)
        img2 = os.path.join(front, front_file)
        save_file1 = os.path.join(save1 , nama_file[0] + ".jpg")
        combine1(img1,img2,save_file1)

        img3 = os.path.join(track_bev, front_file)
        img4 = os.path.join(pseudo_bev, front_file)
        save_file2 = os.path.join(save2 , nama_file[0] + ".jpg")
        combine2(img3,img4,save_file2)

        img5 = os.path.join(save1, nama_file[0] + ".jpg")
        img6 = os.path.join(save2, nama_file[0] + ".jpg")
        save_file3 = os.path.join(save3 , nama_file[0] + ".jpg")
        combine3(img5,img6,save_file3)


if __name__ == '__main__':
    data_dir = "/media/ferdyan/LocalDiskE/Hasil/dataset/Long/3/"
    viz_frenet(data_dir)
    viz_frenet2(data_dir)
    viz_front(data_dir)
    resize_gambar(data_dir)
    resize_gambar2(data_dir)
    combine_semua(data_dir)
