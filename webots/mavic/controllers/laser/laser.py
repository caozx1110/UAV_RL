"""laser controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import LidarPoint
from controller import Lidar

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
Lidar = robot.getDevice("LDS-01")
Lidar.enable(timestep) #启动更新
Lidar.enablePointCloud() #启动激光雷达点云更新

# Lidar.isPointCloudEnabled() #判断是否已经启动点云更新，是则返回true
# print(Lidar.getRangeImage()) # 读取激光雷达捕获的最后一个范围图像的内容，一维列表
# print(Lidar.getRangeImageArray()) #与上同，但返回的是二维列表

Cloud = Lidar.getPointCloud() #获取点数组
print(Lidar.getNumberOfPoints()) #获取总数

FOLDER = './data/'

def WriteIntoFile(pc, filename):
    with open(filename, 'w') as f:
        for p in pc:
            f.write(str(p.x) + ' ' + str(p.y) + ' ' + str(p.z) + '\n')
    
    print('Write into file: ' + filename)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    try:
        Cloud = Lidar.getPointCloud() #获取点数组
        WriteIntoFile(Cloud, FOLDER + str(Cloud[1].time) + '.txt')
        # print(Cloud[1].time)
    except Exception as e:
        print(e)
    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
Lidar.disable()
Lidar.disablePointCloud() #关闭激光雷达点云更新
