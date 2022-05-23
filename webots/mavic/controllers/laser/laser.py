"""laser controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Lidar, LidarPoint, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
import os
import numpy as np
import cv2

FOLDER = './data/'

def SIGN(x):
    return (x > 0) - (x < 0)

def CLAMP(value, low, high):
    return min(high, max(low, value))

def WriteIntoFile(pc, filename):
    # if folder is not existing, make new folder
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
    with open(filename, 'w') as f:
        for p in pc:
            f.write(str(p.x) + ' ' + str(p.y) + ' ' + str(p.z) + '\n')

    print('Write into file: ' + filename)


if __name__ == '__main__':
    # create the Robot instance.
    robot = Robot()

    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # You should insert a getDevice-like function in order to get the
    # instance of a device of the robot. Something like:
    #  motor = robot.getDevice('motorname')
    #  ds = robot.getDevice('dsname')
    #  ds.enable(timestep)

    # device initialization
    camera = robot.getDevice('camera')
    camera.enable(timestep)
    front_left_led = robot.getDevice('front left led')
    front_right_led = robot.getDevice('front right led')
    imu = robot.getDevice('inertial unit')
    imu.enable(timestep)
    gps = robot.getDevice('gps')
    gps.enable(timestep)
    compass = robot.getDevice('compass')
    compass.enable(timestep)
    gyro = robot.getDevice('gyro')
    gyro.enable(timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)
    camera_roll_motor = robot.getDevice('camera roll')
    camera_pitch_motor = robot.getDevice('camera pitch')
    front_left_motor = robot.getDevice('front left propeller')
    front_right_motor = robot.getDevice('front right propeller')
    rear_left_motor = robot.getDevice('rear left propeller')
    rear_right_motor = robot.getDevice('rear right propeller')
    motors = [
        front_left_motor, front_right_motor, rear_left_motor, rear_right_motor
    ]
    for i in range(4):
        motors[i].setPosition(np.inf)
        motors[i].setVelocity(1.0)

    # Lidar initialization
    lidar = robot.getDevice("LDS-01")
    lidar.enable(timestep)  #启动更新
    lidar.enablePointCloud()  #启动激光雷达点云更新
    # Lidar.isPointCloudEnabled() #判断是否已经启动点云更新，是则返回true
    # print(Lidar.getRangeImage()) # 读取激光雷达捕获的最后一个范围图像的内容，一维列表
    # print(Lidar.getRangeImageArray()) #与上同，但返回的是二维列表
    # Cloud = Lidar.getPointCloud() #获取点数组
    # print(Lidar.getNumberOfPoints()) #获取总数

    print("Start the drone...")

    while robot.step(timestep) != -1:
        if robot.getTime() > 1.0:
            break

    print("You can control the drone with your computer keyboard:\n")
    print("- 'up': move forward.\n")
    print("- 'down': move backward.\n")
    print("- 'right': turn right.\n")
    print("- 'left': turn left.\n")
    print("- 'shift + up': increase the target altitude.\n")
    print("- 'shift + down': decrease the target altitude.\n")
    print("- 'shift + right': strafe right.\n")
    print("- 'shift + left': strafe left.\n")

    # Constants, empirically found.
    k_vertical_thrust = 68.5  # with this thrust, the drone lifts.
    k_vertical_offset = 0.6  # Vertical offset where the robot actually targets to stabilize itself.
    k_vertical_p = 3.0  # P constant of the vertical PID.
    k_roll_p = 50.0  # P constant of the roll PID.
    k_pitch_p = 30.0  # P constant of the pitch PID.

    # Variables.
    target_altitude = 1.0  # The target altitude. Can be changed by the user.

    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        # Read the sensors:
        # Enter here functions to read sensor data, like:
        #  val = ds.getValue()
        time = robot.getTime()

        '''camera img'''
        # cameraData = np.array(camera.getImageArray())
        # print(cameraData.shape)

        # Retrieve robot position using the sensors.
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        altitude = gps.getValues()[2]
        roll_acceleration = gyro.getValues()[0]
        pitch_acceleration = gyro.getValues()[1]
        
        # Blink the front LEDs alternatively with a 1 second rate.
        led_state = ((int(time)) % 2) > 0
        front_left_led.set(led_state)
        front_right_led.set(led_state)

        # Stabilize the Camera by actuating the camera motors according to the gyro feedback.
        camera_roll_motor.setPosition(-0.115 * roll_acceleration)
        camera_pitch_motor.setPosition(-0.1 * pitch_acceleration)

        # Transform the keyboard input to disturbances on the stabilization algorithm.
        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0
        key = keyboard.getKey()
        while key > 0:
            if key == Keyboard.UP:
                pitch_disturbance = -2.0
            elif key == Keyboard.DOWN:
                pitch_disturbance = 2.0
            elif key == Keyboard.RIGHT:
                yaw_disturbance = -1.3
            elif key == Keyboard.LEFT:
                yaw_disturbance = 1.3
            elif key == Keyboard.SHIFT + Keyboard.RIGHT:
                roll_disturbance = -1.0
            elif key == Keyboard.SHIFT + Keyboard.LEFT:
                roll_disturbance = 1.0
            elif key == Keyboard.SHIFT + Keyboard.UP:
                target_altitude += 0.05
                print("target altitude: %f [m]" % target_altitude)
            elif key == Keyboard.SHIFT + Keyboard.DOWN:
                target_altitude -= 0.05
                print("target altitude: %f [m]" % target_altitude)
            
            key = keyboard.getKey()

        '''lidar'''
        try:
            Cloud = lidar.getPointCloud()  #获取点数组
            # TODO: write into file
            # WriteIntoFile(Cloud, FOLDER + str(Cloud[1].time) + '.txt')
            # print(Cloud[1].time)
        except Exception as e:
            print(e)
        # Process sensor data here.
        # Compute the roll, pitch, yaw and vertical inputs.
        roll_input = k_roll_p * CLAMP(roll, -1.0, 1.0) + roll_acceleration + roll_disturbance
        pitch_input = k_pitch_p * CLAMP(pitch, -1.0, 1.0) + pitch_acceleration + pitch_disturbance
        yaw_input = yaw_disturbance
        clamped_difference_altitude = CLAMP(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
        vertical_input = k_vertical_p * pow(clamped_difference_altitude, 3.0)

        # Actuate the motors taking into consideration all the computed inputs.
        front_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
        front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
        rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
        rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

        # Enter here functions to send actuator commands, like:
        #  motor.setPosition(10.0)
        front_left_motor.setVelocity(front_left_motor_input)
        front_right_motor.setVelocity(-front_right_motor_input)
        rear_left_motor.setVelocity(-rear_left_motor_input)
        rear_right_motor.setVelocity(rear_right_motor_input)
        # print(front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input)
        pass

    # Enter here exit cleanup code.
    lidar.disable()
    lidar.disablePointCloud()  #关闭激光雷达点云更新
