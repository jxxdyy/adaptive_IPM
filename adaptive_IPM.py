import numpy as np
import math as m
import matplotlib.pyplot as plt
import world_to_camera as wtc
from mpl_toolkits.mplot3d import Axes3D

class adaptive_IPM:
    def __init__(self):
        self.num = 500
        self.max_x = 20
        self.max_y = 30
        self.k = 94.34

        self.fx = 481.20
        self.fy = 480.00
        self.cx = 319.50
        self.cy = 239.50

        self.cam_x = 10.0
        self.cam_y = -5.0
        self.cam_z = 5.0  # cam_z > 0
        self.pan = 0.0
        self.tilt = -15.0

        self.pitch = 0.0  # pitch motion change

        self.new_tilt = self.tilt + self.pitch
        #self.fr = self.fx*(self.w_ccd / 2*self.cx)
        self.fr = self.fx*(1/self.k)
        self.theta = -self.tilt * m.pi / 180
        self.theta_p = -self.pitch * m.pi / 180

        # ================= Set ground data  =================
        self.x_value = np.linspace(1, self.max_x, self.num)  # linspace(a, b, c) : a ~ b 까지 c등분 한 좌표들
        self.y_value = np.linspace(1, self.max_y, self.num)

        # ================= world to camera function =================
        # 3D point
        self.point_arr = wtc.get_3D_point(self.x_value, self.y_value)
        # Intensidy
        self.intensity_arr = wtc.point_wise_intensity(self.point_arr, self.max_x)
        # Extrinsic & Intrinsic parameter set
        self.extrinsic_param = wtc.set_extrinsic_parameter(self.cam_x, self.cam_y, self.cam_z, self.pan, self.new_tilt)
        print("")
        self.intrinsic_param = wtc.set_intrinsic_parameter(self.fx, self.fy, self.cx, self.cy)
        # Camera
        self.camera_arr = wtc.world_to_camera(self.point_arr, self.extrinsic_param)
        # Pixel
        self.pixel_arr = wtc.camera_to_pixel(self.camera_arr, self.intrinsic_param)


    # =========================== Ground point visualizing ===========================
    def Ground_visualing(self):
        x_mesh = self.point_arr[:, 0]
        y_mesh = self.point_arr[:, 1]

        # Middle line
        line_arr = self.point_arr[(x_mesh > (self.max_x / 2) - 1) & (x_mesh < (self.max_x / 2) + 1)]
        lx_mesh = line_arr[:, 0]
        ly_mesh = line_arr[:, 1]
        # s : point size, c : point color
        plt.scatter(x_mesh, y_mesh, s=1, c='black')
        plt.scatter(lx_mesh, ly_mesh, s=1, c='yellow')

        # x, y축의 범위 지정
        plt.xlim(0, self.x_value[-1] + 2)
        plt.ylim(0, self.y_value[-1] + 2)

        plt.title("Ground point - XY plane")


    # =========================== 3D Graph visualizing ===========================
    def ThreeD_graph_visualing(self):
        x_mesh = self.point_arr[:, 0]
        y_mesh = self.point_arr[:, 1]

        # Middle line
        line_arr = self.point_arr[(x_mesh > (self.max_x / 2) - 1) & (x_mesh < (self.max_x / 2) + 1)]
        lx_mesh = line_arr[:, 0]
        ly_mesh = line_arr[:, 1]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x_mesh, y_mesh, s=1, c='black')  # Ground
        ax.scatter(lx_mesh, ly_mesh, s=1, c='yellow')
        ax.scatter(self.cam_x, self.cam_y, self.cam_z, marker=8, s=1, c='r')  # Camera position

        basis = wtc.make_axis(self.extrinsic_param)
        ax.text(self.cam_x, self.cam_y, self.cam_z + 1, 'Camera : ({}, {}, {})'.format(self.cam_x, self.cam_y, self.cam_z))

        ax.plot([self.cam_x, basis[0][0] + self.cam_x], [self.cam_y, basis[0][1] + self.cam_y], [self.cam_z, basis[0][2] + self.cam_z],
                'r')  # X_axis red
        ax.plot([self.cam_x, basis[1][0] + self.cam_x], [self.cam_y, basis[1][1] + self.cam_y], [self.cam_z, basis[1][2] + self.cam_z],
                'g')  # Y_axis green
        ax.plot([self.cam_x, basis[2][0] + self.cam_x], [self.cam_y, basis[2][1] + self.cam_y], [self.cam_z, basis[2][2] + self.cam_z],
                'b')  # Z_axis blue
        # ax.scatter(basis[0][0] + cam_x, basis[0][1] + cam_y, basis[0][2] + cam_z, s=50, c='r')
        # ax.scatter(basis[1][0] + cam_x, basis[1][1] + cam_y, basis[1][2] + cam_z, s=50, c='g')
        # ax.scatter(basis[2][0] + cam_x, basis[2][1] + cam_y, basis[2][2] + cam_z, s=50, c='b')

        # optical_axis yellow
        # ax.plot([cam_x, basis[2][0] + cam_x], [cam_y, basis[2][1] + cam_y+7], [cam_z-0.1, basis[2][2] + cam_z-0.1], 'y', linestyle=':')

        ax.set_xlim(0, self.x_value[-1] + 2)
        ax.set_ylim(-5, self.y_value[-1] + 2)
        ax.set_zlim(0, self.x_value[-1] + 2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.title('Camera Position & 3D Graph')


    # =========================== Normalized point visualizing ===========================
    def Normalized_visualizing(self):
        n_x = self.pixel_arr[:, 0]
        n_y = self.pixel_arr[:, 1]
    
        plt.figure()
        plt.scatter(n_x, -n_y, s=10, c='b')

        plt.title('Normalized Point')


    # =========================== Image visualizing ===========================
    def Image_visulizing(self):
        # Create an image using intensity
        result_img = wtc.visualizing_image(self.pixel_arr, self.intensity_arr)

        plt.figure()
        plt.imshow(result_img)
        plt.title("Camera Image - Pixel (640*480)")


    def v2r(self, v):
        r = (1/self.k)*(self.cy + 0.5 - v)
        return r


    def u2c(self, u):
        c = (1/self.k)*(u - self.cx + 0.5)
        return c

    # image 내부에 투영된 points의 pixel좌표
    def image_pixel(self):
        # image 내부에 들어오는 pixel
        img_pixels = self.pixel_arr[((0 <= self.pixel_arr[:, 0]) & (self.pixel_arr[:, 0] < 640)) &
                               ((0 <= self.pixel_arr[:, 1]) & (self.pixel_arr[:, 1] < 480))]

        # print(img_pixels)
        u = img_pixels[:, 0]
        v = img_pixels[:, 1]

        return u, v


    def line_pixel(self):
        line_u = []
        line_v = []
        img_u, img_v = self.image_pixel()

        black = np.zeros(3, dtype=int)
        result_img = wtc.visualizing_image(self.pixel_arr, self.intensity_arr)

        for i in range(len(img_u)):
            # Pixel 값이 (0, 0, 0)이 아닌 pixel들을 분류
            if not np.array_equal(result_img[img_v[i], img_u[i]], black):
                line_u.append(img_u[i])
                line_v.append(img_v[i])

        return line_u, line_v


    def IPM_X(self, y_pixels):
        X_list = []
        h = self.cam_z
        for i in range(len(y_pixels)):
            v = y_pixels[i]
            theta_v = -m.atan2(self.v2r(v), self.fr)

            X = h * (1 / m.tan(self.theta + theta_v))
            X_list.append(X)

        return X_list


    def IPM_Y(self, x_pixels, y_pixels, X):
        Y_list = []
        for i in range(len(y_pixels)):
            u = x_pixels[i]

            Y = -X[i] * self.u2c(u) / self.fr
            Y_list.append(Y)

        return Y_list


    def adaptive_IPM_X(self, y_pixels):
        X_list = []
        h = self.cam_z
        for i in range(len(y_pixels)):
            v = y_pixels[i]
            theta_v = -m.atan2(self.v2r(v), self.fr)

            X = h * (1 / m.tan(self.theta + self.theta_p + theta_v))
            X_list.append(X)

        return X_list


    def adaptive_IPM_Y(self, x_pixels, y_pixels, X):
        Y_list = []
        for i in range(len(y_pixels)):
            u = x_pixels[i]
            v = y_pixels[i]
            theta_v = -m.atan2(self.v2r(v), self.fr)

            Y = -(m.cos(theta_v) / m.cos(self.theta + self.theta_p + theta_v)) * X[i] * self.u2c(u) / self.fr
            #Y = -X[i] * self.u2c(u) / self.fr
            Y_list.append(Y)

        return Y_list


    def IPM_visualizing(self):
        u, v = self.image_pixel()
        line_u, line_v = self.line_pixel()

        X = self.IPM_X(v)
        Y = self.IPM_Y(u, v, X)

        line_X = self.IPM_X(line_v)
        line_Y = self.IPM_Y(line_u, line_v, line_X)

        plt.figure()
        plt.scatter(X, Y, s=1, c='black')
        plt.scatter(line_X, line_Y, s=1, c='yellow')

        plt.title("IPM Plane / Pitch motion : {}".format(self.pitch))


    def Adaptive_IPM_visualizing(self):
        u, v = self.image_pixel()
        line_u, line_v = self.line_pixel()

        X = self.adaptive_IPM_X(v)
        Y = self.adaptive_IPM_Y(u, v, X)

        line_X = self.adaptive_IPM_X(line_v)
        line_Y = self.adaptive_IPM_Y(line_u, line_v, line_X)

        plt.figure()
        plt.scatter(X, Y, s=1, c='black')
        plt.scatter(line_X, line_Y, s=1, c='yellow')

        plt.title("Adaptive IPM Plane / Pitch motion : {}".format(self.pitch))


if __name__ == "__main__":
    print("=================== Do Adaptive IPM ===================")
    IPM1 = adaptive_IPM()

    IPM1.Ground_visualing()
    IPM1.ThreeD_graph_visualing()
    IPM1.Image_visulizing()

    IPM1.IPM_visualizing()
    IPM1.Adaptive_IPM_visualizing()

    plt.show()
