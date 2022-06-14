import sys
sys.path.insert(0, '/home/konstantin/RoboDK/Python')
from robodk.robolink import *
from robodk.robomath import *

RDK = Robolink()

robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception('No robot selected or available')

# print(robot.Pose())
while True:
    target = UR_2_Pose([-300, 0, 100, 0, 3.142, 0])
    robot.MoveL(target)
    target = UR_2_Pose([-300, 0, 0, 0, 3.142, 0])
    robot.MoveL(target)
    