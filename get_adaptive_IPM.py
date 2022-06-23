import rclpy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import transforms3d as tf
from rclpy.node import Node
import world_to_camera as wtc
import pcl
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
#from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point32
from sensor_msgs.msg import ChannelFloat32
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class Adaptive_IPM(Node):
    
    def __init__(self):
        super().__init__('adaptive_IPM')
        self.subscription = self.create_subscription(CompressedImage, '/cam_f/image/compressed', self.get_IPM, 10)
        #self.subscription = self.create_subscription(Imu, '/imu', self.imu_quat2euler, 10)
        self.subscription = self.create_subscription(Odometry, '/odomgyro', self.odom_quat2euler, 10)
        self.subscription  # prevent unused variable warning
        self.IPM_publisher = self.create_publisher(PointCloud2, '/IPM_points', 10)
        
        self.roll = 0
        self.pitch = 0
        self.pitch_list = []
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
        self.cam_z = 3
        self.v_pan = -90.0
        self.v_tilt = -90.0
        self.extrinsic_param = wtc.set_extrinsic_parameter(self.cam_x, self.cam_y, self.cam_z, self.v_pan, self.v_tilt)
        print("")
        self.intrinsic_param = wtc.set_intrinsic_parameter(self.fx//4, self.fy//4, self.cx//4, self.cy//4)
        
        
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
        print(self.pitch)
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
    
    
    def point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions.
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        """
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

        data = points.astype(dtype).tobytes()
        fields = [sensor_msgs.PointField(
            name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyzrgb')]
        header = std_msgs.Header(frame_id=parent_frame)       
            
            
        return sensor_msgs.PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 6),  # Every point consists of three float32s.
            row_step=(itemsize * 6 * points.shape[0]),
            data=data
        )        
          
            
    def get_IPM(self, msg):
        bridge = CvBridge()
        front_image = bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        #front_image = np.uint8(front_image)
        resize_f_img = cv2.resize(front_image, (front_image.shape[1]//4, front_image.shape[0]//4), interpolation=cv2.INTER_CUBIC)
        img_width = resize_f_img.shape[1]
        img_height = resize_f_img.shape[0]
        
        
        # ROI
        mask = np.zeros_like(resize_f_img)
        roi_u1, roi_v1 = img_width//6, img_height
        roi_u2, roi_v2 = img_width//6, img_height*2//3
        roi_u3, roi_v3 = img_width*5//6, img_height*2//3
        roi_u6, roi_v6 = img_width*5//6, img_height
        vertices = np.array([[(roi_u1, roi_v1), (roi_u2, roi_v2), (roi_u3, roi_v3), (roi_u6, roi_v6)]], dtype=np.int32)
        
        cv2.fillPoly(mask, vertices, (255, 255, 255))
        ROI_area = cv2.bitwise_and(resize_f_img, mask)
        # 파란색 직사각형 표시
        #cv2.rectangle(resize_f_img, (roi_u1, roi_v2), (roi_u6, roi_v6), (255,0,0), 2)
        
        # ROI image -> 해당 intesity만 받아올 때 써볼 수 있지 않을까?
        roi_f_img = ROI_area[roi_v3:roi_v6, roi_u1:roi_u6]
        roi_intensity = np.reshape(roi_f_img, (roi_f_img.size//3, 3))
        #print(front_image.shape)
        #print(roi_f_img.shape)
        print(roi_intensity.shape)
    
        # cv2.namedWindow("front_image", 0);
        # cv2.resizeWindow("front_image", front_image.shape[1]//2, front_image.shape[0]//2)
        # cv2.imshow("front_image", front_image)
    
        cv2.namedWindow("roi_f_img", 0);
        cv2.resizeWindow("roi_f_img", roi_f_img.shape[1], roi_f_img.shape[0])
        cv2.imshow("roi_f_img", roi_f_img)
        
        # cv2.namedWindow("resize_front_image", 0);
        # cv2.resizeWindow("resize_front_image", resize_f_img.shape[1], resize_f_img.shape[0])
        # cv2.imshow("resize_front_image", resize_f_img)
        
        
        #print(self.theta_p)
        # =================================== Adaptive IPM 적용 부분 ===================================
        # pixel idx 얻기
        u_arr, v_arr = np.meshgrid(np.arange(roi_u1, roi_u6), np.arange(roi_v3, roi_v6))
        
        # IPM 적용시킬 pixel 
        u = np.reshape(u_arr, (u_arr.size, 1))
        v = np.reshape(v_arr, (v_arr.size, 1))
        #pixel_arr = np.column_stack((u, v))
    
        # image coordinates translation
        v2r = lambda v:(self.cy//4 + 0.5 - v)
        u2c = lambda u:(u - (self.cx//4 + 0.5))
        r = v2r(v)
        c = u2c(u)      
        
        # derive X & Y
        theta_v = -np.arctan2(r, self.fx//4)
        print('theta_p :', self.theta_p)
        X = self.h * (1 / np.tan(self.theta + self.theta_p + theta_v))
        Y = -(np.cos(theta_v) / np.cos(self.theta + self.theta_p + theta_v)) * X * c / (self.fx//4)
        Z = np.zeros_like(X)
        X_Y = np.column_stack((X, Y, Z, roi_intensity[:,0], roi_intensity[:,1], roi_intensity[:,2]))
        print("X size :", X.size)
        # print("Y size :", np.min(Y))
        # print(X_Y)
        
        
        # =================================== virtual camera로 visualizing ===================================
        # camera_XY = wtc.world_to_camera(X_Y, self.extrinsic_param)
        # pixel_XY = wtc.camera_to_pixel(camera_XY, self.intrinsic_param)
        # BEV_image = wtc.visualizing_image(pixel_XY, roi_intensity)
        
        # cv2.namedWindow("BEV_image", 0);
        # cv2.resizeWindow("BEV_image", BEV_image.shape[1], BEV_image.shape[0])
        # cv2.imshow("BEV_image", BEV_image)
        cv2.waitKey(1)


        # =================================== Publishing PointCloud ===================================
        # IPM_points = PointCloud()
        # rgb_ch = ChannelFloat32()
        # rgb_ch.name = 'intensity'
        
        
        # for i in range(X.size):
        #     data = Point32()
        #     data.x = X[i][0]
        #     data.y = Y[i][0]
        #     rgb_color = roi_intensity[i][2]
        #     IPM_points.points.append(data)
        #     rgb_ch.values.append(rgb_color)
        
        # IPM_points.channels.append(rgb_ch)
        # IPM_points.header.frame_id = "ipm"
        # self.IPM_publisher.publish(IPM_points)


        # plt.scatter(X, Y, s=1, c='black')
        # plt.show()
        
        IPM_points = self.point_cloud(X_Y, 'ipm')
        self.IPM_publisher.publish(IPM_points)
        
        
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