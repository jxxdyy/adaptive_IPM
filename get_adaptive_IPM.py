import rclpy
import cv2
import numpy as np
import math as m
import matplotlib.pyplot as plt
import transforms3d as tf
from rclpy.node import Node
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
        
    def get_image(self, msg):
        bridge = CvBridge()
        #self.get_logger().info('I heard: {}'.format(msg))
        front_image = bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        # resize image
        resize_f_img = cv2.resize(front_image, (front_image.shape[1]//2, front_image.shape[0]//2), interpolation=cv2.INTER_NEAREST)
        #print(front_image)
        print(front_image.shape)

        cv2.namedWindow("front_image", 0);
        cv2.resizeWindow("front_image", front_image.shape[1]//2, front_image.shape[0]//2)
        cv2.imshow("front_image", front_image)
        # cv2.namedWindow("resize_front_image", 0);
        # cv2.resizeWindow("resize_front_image", resize_f_img.shape[1], resize_f_img.shape[0])
        # cv2.imshow("resize_front_image", resize_f_img)
        cv2.waitKey(1)
        
        
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
        print(front_image.dtype)
        #front_image = np.uint8(front_image)
        img_width = front_image.shape[1]
        img_height = front_image.shape[0]
        resize_f_img = cv2.resize(front_image, (img_width//2, img_height//2), interpolation=cv2.INTER_CUBIC)
        #print(resize_f_img.shape)
        
        # ROI
        roi_u1, roi_v1 = img_width//6, img_height*4//7
        roi_u2, roi_v2 = img_width*5//6, img_height
        
        roi_u3, roi_v3 = img_width*(1/4), img_height*(5/8)
        roi_u4, roi_v4 = img_width*(3/4), img_height*(5/8)
        roi_u5, roi_v5 = img_width, img_height*(3/4)
        roi_u6, roi_v6 = img_height, img_height
        cv2.rectangle(front_image, (roi_u1, roi_v1), (roi_u2, roi_v2), (255,0,0), 2)
        
        # ROI image -> 해당 intesity만 받아올 때 써볼 수 있지 않을까?
        roi_f_img = front_image[roi_v1:roi_v2, roi_u1:roi_u2]
        roi_intensity = np.reshape(roi_f_img, (roi_f_img.size//3, 3))
        #roi_intensity = np.array(roi_intensity, dtype=np.uint8)
        #print(front_image.shape)
        #print(roi_f_img.shape)
        #print(roi_intensity)
    
        cv2.namedWindow("front_image", 0);
        cv2.resizeWindow("front_image", front_image.shape[1]//2, front_image.shape[0]//2)
        cv2.imshow("front_image", front_image)
        cv2.namedWindow("roi_f_img", 0);
        cv2.resizeWindow("roi_f_img", roi_f_img.shape[1]//2, roi_f_img.shape[0]//2)
        cv2.imshow("roi_f_img", roi_f_img)
        # cv2.namedWindow("resize_front_image", 0);
        # cv2.resizeWindow("resize_front_image", resize_f_img.shape[1], resize_f_img.shape[0])
        # cv2.imshow("resize_front_image", resize_f_img)
        cv2.waitKey(1)
        
    
        #print(self.theta_p)
        u_arr, v_arr = np.meshgrid(np.arange(roi_u1, roi_u2), np.arange(roi_v1, roi_v2))
        
        # IPM 적용시킬 pixel 
        u = np.reshape(u_arr, (u_arr.size, 1))
        v = np.reshape(v_arr, (v_arr.size, 1))
        #pixel_arr = np.column_stack((u, v))
    
        # IPM 적용 부분
        v2r = lambda v:(self.cy + 0.5 - v)
        u2c = lambda u:(u - (self.cx + 0.5))
        r = v2r(v)
        c = u2c(u)      
         
        theta_v = -np.arctan2(r, self.fx)
        X = self.h * (1 / np.tan(self.theta + theta_v))
        Y = -(np.cos(theta_v) / np.cos(self.theta + theta_v)) * X * c / self.fx
        X_Y = np.column_stack((X, Y))
        
        #print("X size :", X.size)
        #print("Y size :", Y.size)
        #print(X_Y)
        
        IPM_points = PointCloud()
        rgb_ch = ChannelFloat32()
        rgb_ch.name = 'rgb'
        
        for i in range(X.size):
            data = Point32()
            data.x = X[i][0]
            data.y = Y[i][0]
            rgb_color = roi_intensity[i][2]
            #print(get_in.values)
            IPM_points.points.append(data)
            rgb_ch.values.append(rgb_color)
        
        
        IPM_points.channels.append(rgb_ch)
        IPM_points.header.frame_id = "ipm"
        self.IPM_publisher.publish(IPM_points)
        
        
        # conventional IPM
        # # 이중 for문이라 연산량 up..
        # for v in range(roi_v1, roi_v2):
        #     for u in range(roi_u1, roi_u2):
        #         theta_v = -m.atan2(v2r(v), self.fx)
            
        #         X = self.h * (1 / m.tan(self.theta + theta_v))
        #         Y = -(m.cos(theta_v) / m.cos(self.theta + theta_v)) * X * u2c(u) / self.fx
        #         #Y = -X * u2c(u) /self.fx
        #         point = Point32()
        #         point.x = X
        #         point.y = Y
        #         #print(point)
        #         IPM_points.points.append(point)

        # print(X)
        # print(Y)
        # IPM_points.header.frame_id = "ipm"
        # #print(IPM_points)   
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