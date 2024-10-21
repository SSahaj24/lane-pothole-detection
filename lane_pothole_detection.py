import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ZedNode(Node):
    def __init__(self):
        super().__init__('zed_node')
        self.bridge = CvBridge()

        # Constants
        self.MASK_THRESHOLD = 4.5/8
        self.LEN_THRESHOLD = 100

        # Publishers (correct topics if needed)
        self.lane_publisher = self.create_publisher(Image, '/lane_img', 1)
        self.pothole_publisher = self.create_publisher(Image, '/pothole_img', 1)

        # Subscribers (correct topics if needed)
        self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)

    def preprocess_image(self, img):
        """
        Basic thresholding and preprocessing of the image

        Parameters:
            img (numpy.ndarray) : Image received from camera 

        Returns:
            numpy.ndarray : Preprocessed image (after grayscaling, binary thresholding, blurring)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply a binary threshold to the grayscale image (120 is threshold, 255 is max value)
        binary = cv2.inRange(gray, 120, 255)

        # Smooth out noise
        blurred = cv2.medianBlur(cv2.blur(binary, (3, 3)), 3)
        
        # Mask out the top half of the image
        cutoff_height = int(blurred.shape[0] * self.MASK_THRESHOLD)
        blurred[:cutoff_height, :] = 0
        
        return blurred


    def get_contours(self, binary_img):
        """
        Identifies the contours of the potholes by filtering based on aspect ratio and utilized area of the bounding box.
        In the remaining image, lane detection is done by identifying the 2 biggest contours.

        Parameters:
            binary_img (numpy.ndarray) : Image received after preprocessing 

        Returns:
            lane_contours(numpy.ndarray)
            pothole_contours(numpy.ndarray)
        """
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.get_logger().info(f'Found {len(contours)} contours')
        
        if not contours or len(contours)==0:
            return np.zeros((0,1,2), dtype=np.int32), []  # Return empty contour with correct shape
        
        pothole_contours = []
        filtered_contours = []
        
        # Debug each contour's shape
        for idx, contour in enumerate(contours):
            self.get_logger().debug(f'Contour {idx} shape: {contour.shape}')
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)
            rect_area = w * h
            extent = float(area) / rect_area

            # Filter contours based on aspect ratio and extent
            if 1 < aspect_ratio < 5 and 0.5 < extent < 0.8 and 200 < area:
                pothole_contours.append(contour)
                self.get_logger().debug(f'Added to pothole_contours, shape: {contour.shape}')
            elif len(contour) > self.LEN_THRESHOLD:
                filtered_contours.append(contour)
                self.get_logger().debug(f'Added to filtered_contours, shape: {contour.shape}')
        
        # Debug filtered contours
        self.get_logger().info(f'Found {len(filtered_contours)} filtered contours for lanes')
        self.get_logger().info(f'Found {len(pothole_contours)} pothole contours')
        
        # Handle lane contours
        if len(filtered_contours)>=2:
            filtered_contours.sort(key=lambda x:len(x), reverse=True)
            lane_contours = filtered_contours[:2]
            self.get_logger().info(f'Selected 2 lane contour shape: {[c.shape for c in lane_contours]}')
        elif len(filtered_contours)==1:
            lane_contours = filtered_contours
            self.get_logger().info(f"Only 1 lane found with shape : {filtered_contours[0].shape}")
        else:
            lane_contours = np.zeros((0,1,2), dtype=np.int32)  # Empty array with correct shape
            self.get_logger().info('No lane contours found, using empty array')
        
        return lane_contours, pothole_contours

    def draw_contours(self, shape, contours):
        """
        Draws the contours on a blank image and returns it

        Parameters:
            shape
            contours

        Returns:
            img(numpy.ndarray)
        """
        img = np.zeros(shape, dtype=np.uint8)
        
        # Handle single contour vs list of contours
        if isinstance(contours, np.ndarray) and len(contours.shape) == 3:
            # Single contour
            self.get_logger().debug(f'Drawing single contour with shape: {contours.shape}')
            if len(contours) > 0:
                cv2.drawContours(img, [contours], -1, (255, 255, 255), -1)
        elif isinstance(contours, list):
            # List of contours
            self.get_logger().debug(f'Drawing {len(contours)} contours')
            if contours:
                cv2.drawContours(img, contours, -1, (255, 255, 255), -1)
                
        return img
        

    def get_lane_pothole_images(self, img):
        """
        Runs contour detection on camera image and returns the contours on a black background (or blank image if nothing detected)
        
        Parameters:
            img(numpy.ndarray)
        
        Returms:
            lane_image(numpy.ndarray)
            pothole_image(numpy.ndarray)
        """
        processed = self.preprocess_image(img)
        self.get_logger().info(f'Processed image shape: {processed.shape}')
        # cv2.imshow("processed_image]", processed)
        # cv2.waitKey(1)
        
        try:
            lane_contours, pothole_contours = self.get_contours(processed)
            self.get_logger().info(f'Lane contours type: {type(lane_contours)}, shape: {lane_contours.shape if isinstance(lane_contours, np.ndarray) else "list"}')
            self.get_logger().info(f'Pothole contours length: {len(pothole_contours)}')
            
            lane_image = self.draw_contours((img.shape[0], img.shape[1], 3), lane_contours)
            pothole_image = self.draw_contours((img.shape[0], img.shape[1], 3), pothole_contours)
            
            # Comment out if needed
            cv2.imshow("Lanes and Potholes", np.hstack([lane_image, pothole_image]))
            cv2.waitKey(1)
            
            return lane_image, pothole_image
        except Exception as e:
            self.get_logger().error(f'Error in get_lanes_potholes: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            # Return blank images in case of error
            blank = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return blank, blank
        
    def camera_callback(self, data):
        """
        Camera callback processes the camera input and publishes lane image and pothole image to rostopics.
        
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imshow("cvimage", cv_image)
            lane_image, pothole_image = self.get_lane_pothole_images(cv_image)
            self.lane_publisher.publish(self.bridge.cv2_to_imgmsg(lane_image, "rgb8"))
            self.pothole_publisher.publish(self.bridge.cv2_to_imgmsg(pothole_image, "rgb8"))
            
        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ZedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
