from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
import numpy as np
from torch import threshold
# Supervisor 继承了 Robot
from controller import Supervisor, Robot
from controller import Lidar, LidarPoint, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
from gym.spaces import Box, Discrete


class UAV(RobotSupervisor):
    """class uav robot"""

    def __init__(self, steps_per_episode=200):
        super().__init__()
        '''robot initialization'''
        self.robot = self.getSelf()
        self.timestep = int(self.getBasicTimeStep())

        # device initialization
        self.camera = self.getDevice('camera')
        self.camera.enable(self.timestep)
        self.front_left_led = self.getDevice('front left led')
        self.front_right_led = self.getDevice('front right led')
        self.imu = self.getDevice('inertial unit')
        self.imu.enable(self.timestep)
        self.gps = self.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.getDevice('compass')
        self.compass.enable(self.timestep)
        self.gyro = self.getDevice('gyro')
        self.gyro.enable(self.timestep)
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)
        self.camera_roll_motor = self.getDevice('camera roll')
        self.camera_pitch_motor = self.getDevice('camera pitch')
        self.front_left_motor = self.getDevice('front left propeller')
        self.front_right_motor = self.getDevice('front right propeller')
        self.rear_left_motor = self.getDevice('rear left propeller')
        self.rear_right_motor = self.getDevice('rear right propeller')
        self.motors = [
            self.front_left_motor, self.front_right_motor,
            self.rear_left_motor, self.rear_right_motor
        ]

        # v: + - - + 垂直起飞
        self.Sign = [+1, -1, -1, +1]
        for i in range(4):
            self.motors[i].setPosition(np.inf)
            self.motors[i].setVelocity(self.Sign[i])

        # Lidar initialization
        self.lidar = self.getDevice("LDS-01")
        self.lidar.enable(self.timestep)  #启动更新
        self.lidar.enablePointCloud()  #启动激光雷达点云更新

        print("UAV initialized")
        '''RL initialization'''
        # observation space, TODO: a rgb picture
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        self.observation_space = Box(low=np.zeros((width, height, 3)),
                                     high=255 * np.ones((width, height, 3)),
                                     dtype=np.uint8)
        # action space, TODO: the four motors' velocity
        # a typical motor velocity is 70.0
        self.action_space = Box(low=np.zeros(4),
                                high=200 * np.ones(4),
                                dtype=np.float32)
        # parameters
        self.StepsPerEpisode = steps_per_episode
        # store
        self.EpisodeScore = 0
        self.EpisodeScoreList = []

    def get_observations(self):
        """
        get observations from the robot

        :return: the camera image 3 x 240 x 400, the imu data
        """
        # 400 x 240 x 3
        camera_data_whc = np.array(self.camera.getImageArray())
        # print(camera_data_whc.shape)
        # 3 x 240 x 400
        camera_data_chw = np.transpose(camera_data_whc, (2, 1, 0))
        rpy_data = np.array(self.imu.getRollPitchYaw())

        return [camera_data_chw, rpy_data]

    def get_reward(self, action=None):
        """
        get reward from the robot

        :param action: the action to be executed
        :return: the reward
        """
        # TODO
        return 1
    
    @staticmethod
    def RestrictAngle(angle):
        """
        restrict the angle to [-pi, pi]
        """
        while angle > np.pi:
            angle = angle - 2 * np.pi
        while angle < -np.pi:
            angle = angle + 2 * np.pi
        return angle

    def is_done(self):
        """
        check if the episode is done, end condition

        :return: True if the episode is done
        """
        # the sum of angle out of 30 degrees, stop
        rpy = self.imu.getRollPitchYaw()
        threshold_rpy = np.pi / 6
        # print(rpy)
        if abs(rpy[0]) + abs(rpy[1]) + abs(self.RestrictAngle(rpy[2] - np.pi)) > threshold_rpy:
            return True

        # TODO
        return False

    def solved(self):
        """
        check if the episode is solved, end condition

        :return: True if the episode is solved
        """
        # TODO
        threshold_score = 200
        if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
            if np.mean(
                    self.episodeScoreList[-100:]
            ) > threshold_score:  # Last 100 episodes' scores average value
                return True
        return False

    def get_default_observation(self):
        """
        get the default observation
        
        :return: the default observation
        """
        return [np.zeros((3, self.camera.getHeight(), self.camera.getWidth())), np.zeros((3,))]

    def apply_action(self, action):
        """
        apply the action to the robot

        :param action: the action to be executed
        """
        action = action[0]
        print("action: ", action)
        for i in range(4):
            self.motors[i].setPosition(np.inf)
            self.motors[i].setVelocity(int(action[i]) * self.Sign[i])

    def render(self, mode='human'):
        """
        render the robot
        """
        # TODO
        print("no need to render")

    def get_info(self):
        """
        get information from the robot
        """
        # TODO
        return None


if __name__ == '__main__':
    uav = UAV()
