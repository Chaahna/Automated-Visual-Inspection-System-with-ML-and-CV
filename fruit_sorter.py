import traceback

import cv2
import numpy as np
import time
import os,sys
import math
import platform

from pymycobot.mycobot import MyCobot

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


IS_CV_4 = cv2.__version__[0] == '4'
__version__ = "1.0"
# Adaptive seeed


class Object_detect():

    def __init__(self, camera_x = 150, camera_y = 5):
        # inherit the parent class
        super(Object_detect, self).__init__()
        # declare mecharm270
        self.mc = None

        # Define dataset paths
        self.data_dir_train = 'dataset/train'
        self.data_dir_test = "dataset/test"
        self.categories = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]
        # Image size for resizing
        self.img_size = 100

        # Load the training dataset
        self.X_train, self.y_train = self.load_data(self.data_dir_train)

# Load the testing dataset
        self.X_test, self.y_test = self.load_data(self.data_dir_test)


        # Extract features for all training images
        self.train_features = [self.extract_features(img) for img in self.X_train]
        self.train_features = np.array(self.train_features)

    # Extract features for all testing images

        self.test_features = [self.extract_features(img) for img in self.X_test]
        self.test_features = np.array(self.test_features)


        print(f"Number of training images: {len(self.X_train)}")
        print(f"Training Features : {self.train_features}")






        # 移动角度
        self.move_angles = [
            [0, 0, 0, 0, 90, 0],  # init the point
            [-33.31, 2.02, -10.72, -0.08, 95, -54.84],  # point to grab
        ]

        # 移动坐标
        self.move_coords = [
            [96.5, -101.9, 185.6, 155.25, 19.14, 75.88], # above the red bucket
            [180.9, -99.3, 184.6, 124.4, 30.9, 80.58], # above the green bucket
            [77.4, 122.1, 179.2, 151.66, 17.94, 178.24],# above the blue bucket
            [2.2, 128.5, 171.6, 163.27, 10.58, -147.25] # yellow
        ]

        # which robot: USB* is m5; ACM* is wio; AMA* is raspi
        self.robot_m5 = os.popen("ls /dev/ttyUSB*").readline()[:-1]
        self.robot_wio = os.popen("ls /dev/ttyACM*").readline()[:-1]
        self.robot_raspi = os.popen("ls /dev/ttyAMA*").readline()[:-1]
        self.robot_jes = os.popen("ls /dev/ttyTHS1").readline()[:-1]
        self.raspi = False
        if "dev" in self.robot_m5:
            self.Pin = [2, 5]
        elif "dev" in self.robot_wio:
            # self.Pin = [20, 21]
            self.Pin = [2, 5]

            for i in self.move_coords:
                i[2] -= 20
        elif "dev" in self.robot_raspi or "dev" in self.robot_jes:
            import RPi.GPIO as GPIO
            GPIO.setwarnings(False)
            self.GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(20, GPIO.OUT)
            GPIO.setup(21, GPIO.OUT)
            GPIO.output(20, 1)
            GPIO.output(21, 1)
            self.raspi = True
        if self.raspi:
            self.gpio_status(False)


        # choose place to set cube
        self.color = 0
        # parameters to calculate camera clipping parameters
        self.x1 = self.x2 = self.y1 = self.y2 = 0
        # set cache of real coord
        self.cache_x = self.cache_y = 0

        # use to calculate coord between cube and mycobot
        self.sum_x1 = self.sum_x2 = self.sum_y2 = self.sum_y1 = 0
        # The coordinates of the grab center point relative to the mycobot
        self.camera_x, self.camera_y = camera_x, camera_y
        # The coordinates of the cube relative to the mycobot
        self.c_x, self.c_y = 0, 0
        # The ratio of pixels to actual values
        self.ratio = 0
        # Get ArUco marker dict that can be detected.
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Get ArUco marker params.
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # 初始化背景减法器
        self.mog =cv2.bgsegm.createBackgroundSubtractorMOG()




    # pump_control pi
    def gpio_status(self, flag):
        if flag:
            self.GPIO.output(20, 0)
            self.GPIO.output(21, 0)
        else:
            self.GPIO.output(20, 1)
            self.GPIO.output(21, 1)

    # 开启吸泵 m5
    def pump_on(self):
        # 让2号位工作
        # self.mc.set_basic_output(2, 0)
        # 让5号位工作
        self.mc.set_basic_output(5, 0)

    # 停止吸泵 m5
    def pump_off(self):
        # 让2号位停止工作
        self.mc.set_basic_output(2, 1)
        # 让5号位停止工作
        self.mc.set_basic_output(5, 1)

    def check_position(self, data, ids):
        """
        循环检测是否到位某个位置
        :param data: 角度或者坐标
        :param ids: 角度-0，坐标-1
        :return:
        """
        try:
            while True:
                res = self.mc.is_in_position(data, ids)
                # print('res', res)
                if res == 1:
                    time.sleep(0.1)
                    break
                time.sleep(0.1)
        except Exception as e:
            e = traceback.format_exc()
            print(e)

    # Grasping motion
    def move(self, x, y, color):
        # send Angle to move mecharm270
        print(color)
        self.mc.send_angles(self.move_angles[0], 30)
        self.check_position(self.move_angles[0], 0)

        # send coordinates to move mycobot
        self.mc.send_coords([x, y,  110, -176.1, 2.4, -125.1], 40, 1) # usb :rx,ry,rz -173.3, -5.48, -57.9


        # self.mc.send_coords([x, y, 150, 179.87, -3.78, -62.75], 25, 0)
        # time.sleep(3)

        self.mc.send_coords([x, y, 70, -176.1, 2.4, -125.1], 40, 1)
        self.check_position([x, y, 70, -176.1, 2.4, -125.1], 1)

        # open pump
        if "dev" in self.robot_m5 or "dev" in self.robot_wio:
            self.pump_on()
        elif "dev" in self.robot_raspi or "dev" in self.robot_jes:
            self.gpio_status(True)
        time.sleep(1.5)

        tmp = []
        while True:
            if not tmp:
                tmp = self.mc.get_angles()
            else:
                break
        time.sleep(0.5)

        # print(tmp)
        self.mc.send_angles([tmp[0], 17.22, -32.51, tmp[3], 97, tmp[5]],30) # [18.8, -7.91, -54.49, -23.02, -0.79, -14.76]
        self.check_position([tmp[0], 17.22, -32.51, tmp[3], 97, tmp[5]], 0)

        self.mc.send_coords(self.move_coords[color], 40, 1)

        self.check_position(self.move_coords[color], 1)

        # close pump

        if "dev" in self.robot_m5 or "dev" in self.robot_wio:
            self.pump_off()
        elif "dev" in self.robot_raspi or "dev" in self.robot_jes:
            self.gpio_status(False)
        time.sleep(5)

        self.mc.send_angles(self.move_angles[1], 30)
        self.check_position(self.move_angles[1], 0)

    # decide whether grab cube
    def decide_move(self, x, y, color):
        print(x, y, self.cache_x, self.cache_y)
        # detect the cube status move or run
        if (abs(x - self.cache_x) + abs(y - self.cache_y)) / 2 > 5:  # mm
            self.cache_x, self.cache_y = x, y
            return
        else:
            self.cache_x = self.cache_y = 0
            # 调整吸泵吸取位置，y增大,向左移动;y减小,向右移动;x增大,前方移动;x减小,向后方移动
            self.move(x, y, color)



    # init mecharm270
    def run(self):

        if "dev" in self.robot_wio :
            self.mc = MyCobot(self.robot_wio, 115200)
        elif "dev" in self.robot_m5:
            self.mc = MyCobot(self.robot_m5, 115200)
        elif "dev" in self.robot_raspi:
            self.mc = MyCobot(self.robot_raspi, 1000000)
        if not self.raspi:
            self.pub_pump(False, self.Pin)
        self.mc.send_angles([-33.31, 2.02, -10.72, -0.08, 95, -54.84], 30)
        time.sleep(3)

    # draw aruco
    def draw_marker(self, img, x, y):
        # draw rectangle on img
        cv2.rectangle(
            img,
            (x - 20, y - 20),
            (x + 20, y + 20),
            (0, 255, 0),
            thickness=2,
            lineType=cv2.FONT_HERSHEY_COMPLEX,
        )
        # add text on rectangle
        cv2.putText(img, "({},{})".format(x, y), (x, y),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (243, 0, 0), 2,)

    # get points of two aruco
    def get_calculate_params(self, img):
        # Convert the image to a gray image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect ArUco marker.
        corners, ids, rejectImaPoint = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        """
        Two Arucos must be present in the picture and in the same order.
        There are two Arucos in the Corners, and each aruco contains the pixels of its four corners.
        Determine the center of the aruco by the four corners of the aruco.
        """
        if len(corners) > 0:
            if ids is not None:
                if len(corners) <= 1 or ids[0] == 1:
                    return None
                x1 = x2 = y1 = y2 = 0
                point_11, point_21, point_31, point_41 = corners[0][0]
                x1, y1 = int((point_11[0] + point_21[0] + point_31[0] + point_41[0]) / 4.0), int(
                    (point_11[1] + point_21[1] + point_31[1] + point_41[1]) / 4.0)
                point_1, point_2, point_3, point_4 = corners[1][0]
                x2, y2 = int((point_1[0] + point_2[0] + point_3[0] + point_4[0]) / 4.0), int(
                    (point_1[1] + point_2[1] + point_3[1] + point_4[1]) / 4.0)
                return x1, x2, y1, y2
        return None

    # set camera clipping parameters
    def set_cut_params(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        print(self.x1, self.y1, self.x2, self.y2)

    # set parameters to calculate the coords between cube and mecharm270
    def set_params(self, c_x, c_y, ratio):
        self.c_x = c_x
        self.c_y = c_y
        self.ratio = 220.0/ratio

    # calculate the coords between cube and mecharm270
    def get_position(self, x, y):
        return ((y - self.c_y)*self.ratio + self.camera_x), ((x - self.c_x)*self.ratio + self.camera_y)

    """
    Calibrate the camera according to the calibration parameters.
    Enlarge the video pixel by 1.5 times, which means enlarge the video size by 1.5 times.
    If two ARuco values have been calculated, clip the video.
    """
    def transform_frame(self, frame):
        # enlarge the image by 1.5 times
        fx = 1.5
        fy = 1.5
        frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy,
                           interpolation=cv2.INTER_CUBIC)
        if self.x1 != self.x2:
            # the cutting ratio here is adjusted according to the actual situation
            frame = frame[int(self.y2*0.78):int(self.y1*1.1),
                          int(self.x1*0.86):int(self.x2*1.08)]
        return frame


   # Load data function
    def load_data(self, data_dir):
        data = []
        labels = []
        for category in self.categories:
            path = os.path.join(data_dir, category)
            if not os.path.exists(path):
                print(f"Warning: Directory {path} does not exist.")
                continue
            label = 0 if "fresh" in category else 1
            for img in os.listdir(path):
                try:
                    img_path = os.path.join(path, img)
                    img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img_array is None:
                        print(f"Warning: Failed to load image {img_path}.")
                        continue
                    img_resized = cv2.resize(img_array, (self.img_size, self.img_size))
                    data.append(img_resized)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
        if len(data) == 0 or len(labels) ==0:
            raise ValeError(f"No data found in directory {data_dir}.")
        return np.array(data), np.array(labels)




# Feature extraction using color and texture analysis
    def extract_features(self, image):
    # Convert image to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate histogram for hue channel
        hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    # Normalize histogram
        hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()

    # Texture analysis using Laplacian variance (sharpness)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Combine features
        features = np.concatenate([hist_hue, [laplacian_var]])
        return features

    def train_model(self):
            # Train a K-Nearest Neighbors classifier
       knn = KNeighborsClassifier(n_neighbors=3)
       knn.fit(self.train_features, self.y_train)

       return knn



    # Function to classify new fruit images
    def classify_fruit(self, knn, image):
        img_resized = cv2.resize(image, (self.img_size, self.img_size))

        img_features = self.extract_features(img_resized).reshape(1, -1)

        prediction = knn.predict(img_features)

        freshness = "Fresh" if prediction[0] == 0 else "Rotten"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y = 0,0
        if len(contours)>0:
            #find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            x = x + w //2 # center x-coordinate
            y = y + h //2 # center y-coordinate


        if freshness == "Fresh" :
            self.color = 1
            print(f"The fruit is {freshness}")

        elif freshness == "Rotten":
            self.color = 2
            print(f"The fruit is {freshness}")





        if abs(x) + abs(y) > 0:
            return x, y
        else:
            return None



if __name__ == "__main__":

    # open the camera
    if platform.system() == "Linux":
        cap_num = 0
        cap = cv2.VideoCapture(cap_num)
        cap.set(3, 640)
        cap.set(4, 480)

        if not cap.isOpened():
            cap.open()

    # init a class of Object_detect
    detect = Object_detect()

	# init mecharm270
    detect.run()

    _init_ = 20
    init_num = 0
    nparams = 0
    num = 0
    real_sx = real_sy = 0
    while cv2.waitKey(1) < 0:
       # read camera
        _, frame = cap.read()
        # deal img
        frame = detect.transform_frame(frame)
        if _init_ > 0:
            _init_ -= 1
            continue

        # calculate the parameters of camera clipping
        if init_num < 20:
            if detect.get_calculate_params(frame) is None:
                cv2.imshow("figure", frame)
                continue
            else:
                x1, x2, y1, y2 = detect.get_calculate_params(frame)
                detect.draw_marker(frame, x1, y1)
                detect.draw_marker(frame, x2, y2)
                detect.sum_x1 += x1
                detect.sum_x2 += x2
                detect.sum_y1 += y1
                detect.sum_y2 += y2
                init_num += 1
                continue
        elif init_num == 20:
            detect.set_cut_params(
                (detect.sum_x1)/20.0,
                (detect.sum_y1)/20.0,
                (detect.sum_x2)/20.0,
                (detect.sum_y2)/20.0,
            )
            detect.sum_x1 = detect.sum_x2 = detect.sum_y1 = detect.sum_y2 = 0
            init_num += 1
            continue

        # calculate params of the coords between cube and mecharm270
        if nparams < 10:
            if detect.get_calculate_params(frame) is None:
                cv2.imshow("figure", frame)
                continue
            else:
                x1, x2, y1, y2 = detect.get_calculate_params(frame)
                detect.draw_marker(frame, x1, y1)
                detect.draw_marker(frame, x2, y2)
                detect.sum_x1 += x1
                detect.sum_x2 += x2
                detect.sum_y1 += y1
                detect.sum_y2 += y2
                nparams += 1
                continue
        elif nparams == 10:
            nparams += 1
            # calculate and set params of calculating real coord between cube and mecharm270
            detect.set_params(
                (detect.sum_x1+detect.sum_x2)/20.0,
                (detect.sum_y1+detect.sum_y2)/20.0,
                abs(detect.sum_x1-detect.sum_x2)/10.0 +
                abs(detect.sum_y1-detect.sum_y2)/10.0
            )
            print("ok")
            continue

        # get detect result
        # detect_result = detect.color_detect(frame)
        # print('调用检测')
        knn_model = detect.train_model()


        detect_result = detect.classify_fruit(knn_model,frame)
        #print(f"The Fruit is: {detect_result}")
        if detect_result is None:
            cv2.imshow("figure", frame)
            continue
        else:
            x, y = detect_result
            # calculate real coord between cube and mecharm270
            real_x, real_y = detect.get_position(x, y)
            if num == 20:

                detect.decide_move(real_sx/20.0, real_sy/20.0, detect.color)
                num = real_sx = real_sy = 0

            else:
                num += 1
                real_sy += real_y
                real_sx += real_x

        cv2.imshow("figure", frame)

        # close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
