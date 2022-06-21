import math as m
import numpy as np


# ================= 3D data list =================
def get_3D_point(lins_x, lins_y):
    '''
    :param lins_x: 3D point x array
    :param lins_y: 3D point y array
    :return: 3D point array
    '''
    size = len(lins_x) * len(lins_y)
    point_z = np.zeros(size)

    x_mesh, y_mesh = np.meshgrid(lins_x, lins_y)
    point_x = x_mesh.ravel()  # 2차원 -> 1차원
    point_y = y_mesh.ravel()

    # 1차원 벡터를 열 벡터로 인식한 뒤, 열 방향으로 합쳐줌
    point_3D = np.column_stack((point_x, point_y, point_z))

    return point_3D  # 3*size


# ================= Extrinsic parameter =================
def set_extrinsic_parameter(x: float = 0.0,
                            y: float = 0.0,
                            z: float = 0.0,
                            pan: float = 0.0,
                            tilt: float = 0.0):
    '''
    :param x: camera position x
    :param y: camera position y
    :param z: camera position z
    :param pan: pan angle
    :param tilt: tilt angle
    :return: extrincsic parameter list
    '''
    ex_param = [x, y, z, pan, tilt]
    print("-----Extrinsic Parmeter-----")
    print("camera position: ({}, {}, {})".format(x, y, z))
    print("pan : {}".format(pan))
    print("tilt : {}".format(tilt))

    return ex_param


# ================= Intrinsic parameter =================#
def set_intrinsic_parameter(fx: float = 0.0,
                            fy: float = 0.0,
                            cx: float = 0.0,
                            cy: float = 0.0):
    '''
    :param fx: focal length x
    :param fy: focal length y
    :param cx: principal point x
    :param cy: principal point y
    :return: intrinsic parameter list
    '''
    in_param = [fx, fy, cx, cy]
    print("-----Intrinsic Parameter-----")
    print("focal length : ({}, {})".format(fx, fy))
    print("principal point : ({}, {})".format(cx, cy))

    return in_param


# ================= Set extrinsic Matrix =================
def extrinsic_matrix(e):
    R = np.zeros((3, 3))  # Extrinsic matrix

    pan = e[3] * m.pi / 180
    tilt = e[4] * m.pi / 180

    # Extrinsic matrix R
    R[0][0] = m.cos(pan)
    R[0][1] = m.sin(pan)
    R[0][2] = 0
    R[1][0] = -m.sin(tilt) * m.sin(pan)
    R[1][1] = m.sin(tilt) * m.cos(pan)
    R[1][2] = -m.cos(tilt)
    R[2][0] = -m.cos(tilt) * m.sin(pan)
    R[2][1] = m.cos(tilt) * m.cos(pan)
    R[2][2] = m.sin(tilt)

    return R


# ================= Set intrinsic Matrix =================
def intrinsic_matrix(i):
    K = np.zeros((3, 3))  # Intrinsic matrix

    # Set intrinsic
    fx = i[0]
    fy = i[1]
    cx = i[2]
    cy = i[3]

    # Intrinsic matrix K
    K[0][0] = fx
    K[0][1] = 0
    K[0][2] = cx
    K[1][0] = 0
    K[1][1] = fy
    K[1][2] = cy
    K[2][0] = 0
    K[2][1] = 0
    K[2][2] = 1

    return K


# ================= world to camera 변환 =================
def world_to_camera(world_point, e):
    '''
    :param world_point: 3D Point array
    :param e: Extrinsic parameter list
    :return: Camera_coord array
    '''
    cam_p = np.zeros(3)  # camera position
    R = extrinsic_matrix(e) # world coord -> camera coord transform

    cam_p[0] = e[0]
    cam_p[1] = e[1]
    cam_p[2] = e[2]

    # 모든 3D 좌표에 대해서 camera 좌표로 변환
    # Xc = R[0][0]*(px - cam_p[0]) + R[0][1]*(py - cam_p[1]) + R[0][2]*(pz - cam_p[2])
    # Yc = R[1][0]*(px - cam_p[0]) + R[1][1]*(py - cam_p[1]) + R[1][2]*(pz - cam_p[2])
    # Zc = R[2][0]*(px - cam_p[0]) + R[2][1]*(py - cam_p[1]) + R[2][2]*(pz - cam_p[2])

    temp = world_point - cam_p
    # Extrinsic matrix compute
    # R x (world_point)^T -> (3*3) x (3*size) = 3*size 이후 다시 transpose -> size*3
    camera_arr = np.transpose(R.dot(np.transpose(temp)))  # size*3

    return camera_arr


# ================= camera to pixel 변환 =================
def camera_to_pixel(camera_arr, i):
    '''
    :param camera_arr: Camera point array
    :param i: Intrinsic parameter list
    :return: Pixel_coord array
    '''
    K = intrinsic_matrix(i)

    # Normalized 좌표로 만들기 위해 Zc=0 인 부분 1로 바꿔 예외 처리
    # for i in range(len(camera_arr)):
    #     if camera_arr[i][2] == 0:
    #         camera_arr[i][2] = 1

    n_x = camera_arr[:, 0] / camera_arr[:, 2]  # Xc / Zc
    n_y = camera_arr[:, 1] / camera_arr[:, 2]  # Yc / Zc
    homo_z = camera_arr[:, 2] / camera_arr[:, 2]  # 1

    normal = np.column_stack((n_x, n_y, homo_z))  # size*2
    # print('normal\n', normal)

    # Intrinsic matrix compute
    pixel_arr = np.transpose(K.dot(np.transpose(normal))).astype(int)  # size*3

    return pixel_arr


# ================= Point-wise Intensity =================
def point_wise_intensity(point_arr, max_x):
    '''
    :param point_arr: 3D point array
    :return: Point-wise intensity array
    '''
    rgb_arr = np.zeros((len(point_arr), 3))
    #rgb_arr[:, 2].fill(255)  # RGB : [0, 0, 255]

    # 양쪽 끝 열만 빨간색
    for i in range(len(point_arr)):
        x = point_arr[i][0]
        if (max_x/2)-1 < x < (max_x/2)+1:
            rgb_arr[i] = [255, 255, 0]
        else:
            rgb_arr[i] = [0, 0, 0]

    return rgb_arr


# ================= Image visualizing =================
def visualizing_image(pixel, rgb_arr):
    '''
    :param pixel: pixel array
    :param rgb_arr: point-wise array
    :return: final_image
    '''
    # img_w = 640
    # img_h = 480
    img_w = 2592//4
    img_h = 1944//4
    # plt.imshow()는 정수형만 표현하므로 dtype = uin8
    result_img = np.full((img_h, img_w, 3), 0, dtype=np.uint8)
    # result_img.fill()

    for i in range(len(pixel)):
        if 0 <= pixel[i][0] < img_w and 0 <= pixel[i][1] < img_h:
            # 해당 pixel의 intensity 대입
            result_img[pixel[i][1], pixel[i][0]] = rgb_arr[i]

    return result_img


# ================= Camera axis =================
def make_axis(e):
    tar = np.zeros((3, 3)) # basis
    R = extrinsic_matrix(e) # Extrinsic matrix
    axis_len = 2 # 좌표축 길이

    tar[0] = [axis_len, 0, 0]
    tar[1] = [0, axis_len, 0]
    tar[2] = [0, 0, axis_len]

    axis = np.transpose(np.transpose(R).dot(np.transpose(tar)))

    return axis
