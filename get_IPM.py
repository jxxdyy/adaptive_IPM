import rclpy
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import transforms3d as tf
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
#from sensor_msgs.msg import Image
from IPM_virtual_data import world_to_camera as wtc
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class Adaptive_IPM(Node):
    
    def __init__(self):
        super().__init__('adaptive_IPM')
        self.subscription = self.create_subscription(CompressedImage, '/cam_f/image/compressed', self.get_IPM, 10)
        self.subscription  # prevent unused variable warning
        self.IPM_publisher = self.create_publisher(PointCloud2, '/IPM_points', 10)
        
        self.imu_pitch = 0
        self.odom_pitch = 0
        self.pitch_list = []
        # front camera parameters
        self.fx = 1307.928446923253
        self.fy = 1305.90567944212
        self.cx = 1335.329531505523
        self.cy = 976.0733898764446
        
        self.h = 0.561 # camera z position
        self.tilt = -0.0126
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
    
        
    def get_image(self, msg):
        bridge = CvBridge()
        #self.get_logger().info('I heard: {}'.format(msg))
        front_image = bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        # resize image
        resize_f_img = cv2.resize(front_image, (front_image.shape[1]//2, front_image.shape[0]//2), interpolation=cv2.INTER_NEAREST)
        #print(front_image)
        print(front_image.shape)
        
        hsv_front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2HSV)
        
        lower_range = (70, 10, 55)
        upper_range = (90, 40, 75)
        
        img_mask = cv2.inRange(hsv_front_image, lower_range, upper_range)
        img_result = cv2.bitwise_and(front_image, front_image, mask=img_mask)

        cv2.namedWindow("front_image", 0);
        cv2.resizeWindow("front_image", front_image.shape[1]//2, front_image.shape[0]//2)
        cv2.imshow("front_image", front_image)
        
        # cv2.namedWindow("hsv_front_image", 0);
        # cv2.resizeWindow("hsv_front_image", front_image.shape[1]//2, front_image.shape[0]//2)
        # cv2.imshow("hsv_front_image", hsv_front_image)
        
        # cv2.namedWindow("img_result", 0);
        # cv2.resizeWindow("img_result", img_result.shape[1]//2, img_result.shape[0]//2)
        # cv2.imshow("img_result", img_result)
        cv2.waitKey(1)
        
        
    def visualizing_image(self, points_field, intensity):
        """ Create a IPM result image

        Args:
            points_field (_array_): N*3 Points array ([X, Y, Z])
            intensity (_array_): N*3 RGB array ([R, G ,B])

        Returns:
            _type_: RGB image
        """
        points = points_field[:,:3]
        
        camera_XY = wtc.world_to_camera(points, self.extrinsic_param)
        pixel_XY = wtc.camera_to_pixel(camera_XY, self.intrinsic_param)
        BEV_image = wtc.visualizing_image(pixel_XY, intensity)
        
        return BEV_image
    
        
    def point_cloud2(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx6 array of xyz positions & 'bgr' color value
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
            for i, n in enumerate('xyzbgr')]

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
        

    def threshold_pointcloud(self, points_field, type):
        """ Do threshold using pointcloud intensity

        Args:
            points_field (_array_): N*6 Points array ([X, Y, Z, temp],  temp = N*6 bgr or hsv array)
            type (_string_): threshold mode

        Returns:
            _type_: thresholded points_field
        """
        if type == 'bgr':
            lower_ran = (65, 65, 50)
            upper_ran = (80, 80, 68)
            
            b = (points_field[:, 3] > lower_ran[0]) & (points_field[:, 3] < upper_ran[0])
            g = (points_field[:, 4] > lower_ran[1]) & (points_field[:, 4] < upper_ran[1])
            r = (points_field[:, 5] > lower_ran[2]) & (points_field[:, 5] < upper_ran[2])
            new_points_field = points_field[(b & g & r)]
            
            return new_points_field
        
        elif type == 'hsv':
            lower_ran = (75, 10, 50)
            upper_ran = (85, 40, 90)
            
            h = (points_field[:, 3] > lower_ran[0]) & (points_field[:, 3] < upper_ran[0])
            s = (points_field[:, 4] > lower_ran[1]) & (points_field[:, 4] < upper_ran[1])
            v = (points_field[:, 5] > lower_ran[2]) & (points_field[:, 5] < upper_ran[2])
              
            new_points_field = points_field[(h & s & v)]
            
            return new_points_field
        
            
    def get_IPM(self, msg):
        #start_time = time.time_ns()
        bridge = CvBridge()
        front_image = bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        #hsv_front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2HSV)
        #front_image = np.uint8(front_image)
        resize_f_img = cv2.resize(front_image, (front_image.shape[1]//2, front_image.shape[0]//2), interpolation=cv2.INTER_CUBIC)

        img_width = resize_f_img.shape[1]
        img_height = resize_f_img.shape[0]

        
        # ROI
        mask = np.zeros_like(resize_f_img)
        roi_u1, roi_v1 = 0, img_height
        roi_u2, roi_v2 = 0, img_height*9//10
        roi_u3, roi_v3 = img_width//4, img_height*2//3
        roi_u4, roi_v4 = img_width*3//4, img_height*2//3
        roi_u5, roi_v5 = img_width, img_height*9//10
        roi_u6, roi_v6 = img_width, img_height
        vertices = np.array([[(roi_u1, roi_v1), (roi_u2, roi_v2), (roi_u3, roi_v3), 
                              (roi_u4, roi_v4), (roi_u5, roi_v5), (roi_u6, roi_v6)]], dtype=np.int32)
        
        cv2.fillPoly(mask, vertices, (255, 255, 255))
        ROI_area = cv2.bitwise_and(resize_f_img, mask)
        # 파란색 직사각형 표시
        cv2.rectangle(resize_f_img, (roi_u1, roi_v3), (roi_u6, roi_v6), (255,0,0), 2)
        
        # ROI image -> 해당 intesity만 받아올 때 써볼 수 있지 않을까?
        roi_f_img = ROI_area[roi_v3:roi_v6, roi_u1:roi_u6]
        roi_intensity = np.reshape(roi_f_img, (roi_f_img.size//3, 3))

        # cv2.namedWindow("front_image", 0);
        # cv2.resizeWindow("front_image", front_image.shape[1]//2, front_image.shape[0]//2)
        # cv2.imshow("front_image", front_image)

        # cv2.namedWindow("roi_f_img", 0);
        # cv2.resizeWindow("roi_f_img", roi_f_img.shape[1]//2, roi_f_img.shape[0]//2)
        # cv2.imshow("roi_f_img", roi_f_img)
        
        cv2.namedWindow("ROI_area", 0);
        cv2.resizeWindow("ROI_area", ROI_area.shape[1]*4//5, ROI_area.shape[0]*4//5)
        cv2.imshow("ROI_area", ROI_area)
        cv2.waitKey(1)
        

        
        # =================================== Adaptive IPM 적용 부분 ===================================
        # pixel idx 얻기
        u_arr, v_arr = np.meshgrid(np.arange(roi_u1, roi_u6), np.arange(roi_v3, roi_v6))

        # IPM 적용시킬 pixel 
        u = np.reshape(u_arr, (u_arr.size, 1))
        v = np.reshape(v_arr, (v_arr.size, 1))
        #pixel_arr = np.column_stack((u, v))
    
        # image coordinates translation
        v2r = lambda v:(self.cy//2 + 0.5 - v)
        u2c = lambda u:(u - (self.cx//2 + 0.5))
        r = v2r(v)
        c = u2c(u)      
 
        
        # derive X & Y
        theta_v = -np.arctan(r/(self.fx//2))
        print('imu pitch :', self.imu_pitch)
        print('odom pitch :', self.odom_pitch)
        print('theta_p :', self.theta_p)

        X = self.h * (1 / np.tan(self.theta + theta_v))
        Y = -(np.cos(theta_v) / np.cos(self.theta + theta_v)) * X * c / (self.fx//2)
        Z = np.zeros_like(X)

        points_field = np.concatenate((X, Y, Z, roi_intensity), axis=1)

        
        # =================================== virtual camera로 visualizing ===================================
        # BEV_image = self.visualizing_image(points_field, roi_intensity)
        
        # cv2.namedWindow("BEV_image", 0);
        # cv2.resizeWindow("BEV_image", BEV_image.shape[1]//2, BEV_image.shape[0]//2)
        # cv2.imshow("BEV_image", BEV_image)


        # =================================== Publishing PointCloud ===================================
        #tile_points_field = self.threshold_pointcloud(points_field, type='bgr')
        
        #print("tile_point size :", len(tile_points_field))
        print("point size :", len(points_field))
        
        IPM_points = self.point_cloud2(points_field, 'ipm')
        self.IPM_publisher.publish(IPM_points)
        
        
        #end_time = time.time_ns()
        #print('코드 실행 시간: %10ds' % (end_time - start_time))
        print('---------------------------------------------')
        
        
def main(args=None):
    rclpy.init(args=args)

    adaptive_IPM = Adaptive_IPM()
    rclpy.spin(adaptive_IPM)
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    adaptive_IPM.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()