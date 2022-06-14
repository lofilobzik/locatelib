from concurrent.futures import thread
import sys
sys.path.insert(0, '/home/konstantin/RoboDK/Python')
from robodk.robolink import *
from robodk.robomath import *

import time
from locatelib import *
import threading, _thread

RDK = Robolink()

robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception('No robot selected or available')

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
back_image = cv2.imread('bg.png')

class App:
    def __init__(self):        
        self.main_thread = threading.Thread(target=self.main)
        self.main_thread.start()
        
    def main(self):
        self.t1 = threading.Thread(target=self.camera)
        self.t1.start()

        self.t2 = threading.Thread(target=self.userinput)
        self.t2.start()
    
    def camera(self):
        while True:
            image = get_image(ip='77.37.184.204')
            straight_image = straighten_image(image, aruco_dict)
            
            if straight_image is None:
                cv2.imshow('image', image)
            else:
                bin_mask = find_bin_mask_difference(straight_image, back_image)
                self.coords, rect = get_coords(bin_mask, 287, 200)
                self.coords = get_robot_coords(self.coords) # change this
                
                # for x, y, w, h in rect:
                #     cv2.rectangle(straight_image, (x, y), (x+w, y+h), (255, 255, 255), 2, cv2.LINE_AA)
                
                for i in range(len(self.coords)):
                    cX = rect[i][1][0]
                    cY = rect[i][1][1]
                    cv2.putText(straight_image, f'{i}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)  
                    
                cv2.imshow('image', straight_image)

            if cv2.waitKey(500) == 27:
                break
        
        self.exit()
            
    def userinput(self):
        while True:
            try:
                num = int(input('Enter object num: '))
                print(f'Moving robot to {self.coords[num]}')
                self.move_robot(self.coords[num][0], self.coords[num][1], 100)
                self.move_robot(self.coords[num][0], self.coords[num][1], 25)
                self.move_robot(self.coords[num][0], self.coords[num][1], 100)
            except ValueError:
                print('ValueError')
            except IndexError:
                print('IndexError')
            
    def move_robot(self, x, y, z):
        target = UR_2_Pose([x, y, z, 0, 3.142, 0])
        try:
            robot.MoveL(target)
        except TargetReachError:
            pass
        
    def exit(self): # stop main thread
        cv2.destroyAllWindows()
        _thread.interrupt_main()
            
app = App()
