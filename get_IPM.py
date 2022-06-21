import rclpy
import cv2
import numpy as np
import math as m
import matplotlib.pyplot as plt
import transforms3d as tf
from rclpy.node import Node
import world_to_camera as wtc
import pcl
#from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from sensor_msgs.msg import ChannelFloat32
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class Adaptive_IPM(Node):
    
    def __init__(self):
        super().__init__('adaptive_IPM')
        self.subscription = self.create_subscription(CompressedImage, '/cam_f/image/compressed', self.get_IPM, 10)
        self.subscription = self.create_subscription(Imu, '/imu', self.imu_quat2euler, 10)
        self.subscription = self.create_subscription(Odometry, '/odomgyro', self.odom_quat2euler, 10)
        self.subscription  # prevent unused variable warning
        self.IPM_publisher = self.create_publisher(PointCloud, '/IPM_points', 10)
        
        self.roll = 0
        self.pitch = 0
        self.pitch_list = []
        self.yaw = 0
        
        # front camera parameters
        self.fx = 1307.928446923253
        self.fy = 1305.90567944212
        self.cx = 1335.329531505523
        self.cy = 976.0733898764446
        
        self.h = 0.58 # camera z position
        self.tilt = -0.008
        self.theta = -self.tilt
        self.theta_p = 0
        
        # virtual camera for visualizing
        self.cam_x = 2.0
        self.cam_y = 0.0
        self.cam_z = 4
        self.v_pan = -90.0
        self.v_tilt = -90.0
        self.extrinsic_param = wtc.set_extrinsic_parameter(self.cam_x, self.cam_y, self.cam_z, self.v_pan, self.v_tilt)
        print("")
        self.intrinsic_param = wtc.set_intrinsic_parameter(self.fx, self.fy, self.cx, self.cy)
        
        
    def imu_quat2euler(self, msg):
        #print()
        quaternion = (msg.orientation.w,
                      msg.orientation.x, 
                      msg.orientation.y, 
                      msg.orientation.z)
        
        self.roll, self.pitch, self.yaw = tf.euler.quat2euler(quaternion)
        #print(self.roll, self.pitch, self.yaw)
        self.pitch_list.append(self.pitch)
        #print(np.average(self.pitch_list))
        self.theta_p = -self.pitch
        
        
    def odom_quat2euler(self, msg):
        quaternion = (msg.pose.pose.orientation.w,
                      msg.pose.pose.orientation.x, 
                      msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z) 
        
        self.roll, self.pitch, self.yaw = tf.euler.quat2euler(quaternion)
        #print(self.roll, self.pitch, self.yaw)
        self.theta_p = -self.pitch
        
            
    def get_IPM(self, msg):
        bridge = CvBridge()
        front_image = bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        #front_image = np.uint8(front_image)
        resize_f_img = cv2.resize(front_image, (front_image.shape[1]//2, front_image.shape[0]//2), interpolation=cv2.INTER_CUBIC)
        img_width = front_image.shape[1]
        img_height = front_image.shape[0]
        
        
        # ROI
        mask = np.zeros_like(front_image)
        roi_u1, roi_v1 = 0, img_height
        roi_u2, roi_v2 = 0, img_height*7//10
        roi_u3, roi_v3 = img_width//4, img_height*3//5
        roi_u4, roi_v4 = img_width*3//4, img_height*3//5
        roi_u5, roi_v5 = img_width, img_height*7//10
        roi_u6, roi_v6 = img_width, img_height
        vertices = np.array([[(roi_u1, roi_v1), (roi_u2, roi_v2), (roi_u3, roi_v3), 
                              (roi_u4, roi_v4), (roi_u5, roi_v5), (roi_u6, roi_v6)]], dtype=np.int32)
        
        cv2.fillPoly(mask, vertices, (255, 255, 255))
        ROI_area = cv2.bitwise_and(front_image, mask)
        
        # ROI image -> 해당 intesity만 받아올 때 써볼 수 있지 않을까?
        roi_f_img = ROI_area[roi_v3:roi_v6, roi_u1:roi_u6]
        roi_intensity = np.reshape(roi_f_img, (roi_f_img.size//3, 3))
        
        
        # =================================== IPM 적용 부분 ===================================
        # pixel idx 얻기
        u_arr, v_arr = np.meshgrid(np.arange(roi_u1, roi_u6), np.arange(roi_v3, roi_v6))
        
        # IPM 적용시킬 pixel 
        u = np.reshape(u_arr, (u_arr.size, 1))
        v = np.reshape(v_arr, (v_arr.size, 1))

        # image coordinates translation
        v2r = lambda v:(self.cy + 0.5 - v)
        u2c = lambda u:(u - (self.cx + 0.5))
        r = v2r(v)
        c = u2c(u)      
        
        # derive X & Y
        theta_v = -np.arctan2(r, self.fx)
        X = self.h * (1 / np.tan(self.theta + theta_v))
        Y = -(np.cos(theta_v) / np.cos(self.theta + theta_v)) * X * c / (self.fx)
        Z = np.zeros_like(X)
        X_Y = np.column_stack((X, Y, Z))
        print(len(X))
        
        
        # =================================== virtual camera로 visualizing ===================================
        camera_XY = wtc.world_to_camera(X_Y, self.extrinsic_param)
        pixel_XY = wtc.camera_to_pixel(camera_XY, self.intrinsic_param)
        print(len(pixel_XY))
        BEV_image = wtc.visualizing_image(pixel_XY, roi_intensity)
        
        cv2.namedWindow("BEV_image", 0);
        cv2.resizeWindow("BEV_image", BEV_image.shape[1]//2, BEV_image.shape[0]//2)
        cv2.imshow("BEV_image", BEV_image)
        cv2.waitKey(1)


        # =================================== Publishing PointCloud ===================================
        IPM_points = PointCloud()
        rgb_ch = ChannelFloat32()
        rgb_ch.name = 'intensity'
        
        
        # for i in range(X.size):
        #     print(X.size)
        #     data = Point32()
        #     data.x = X[i][0]
        #     data.y = Y[i][0]
        #     rgb_color = roi_intensity[i][2]
        #     #print(get_in.values)
        #     IPM_points.points.append(data)
        #     rgb_ch.values.append(rgb_color)
        
        
        # IPM_points.channels.append(rgb_ch)
        # IPM_points.header.frame_id = "ipm"
        # self.IPM_publisher.publish(IPM_points)


        # plt.scatter(X, Y, s=1, c='black')
        # plt.show()
        
        
def main(args=None):
    rclpy.init(args=args)

    adaptive_IPM = Adaptive_IPM()
    rclpy.spin(adaptive_IPM)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    adaptive_IPM.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    print("do_main?")
    main()