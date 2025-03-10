from controller import Supervisor,Display
from controller import Motor, Camera, GPS, InertialUnit, Gyro, Lidar
import matplotlib.pyplot as plt     #导入matplotlib库绘图
import numpy as np
import math
# 初始化 Supervisor
supervisor = Supervisor()
timeStep = int(supervisor.getBasicTimeStep())


MAP_SIZE = 100
GRID_RESOLUTION = 0.1  #定义每个栅格的大小（相对实际大小）
occupancy_grid = np.zeros((MAP_SIZE, MAP_SIZE))  # 0: 空闲, 1: 障碍
display = supervisor.getDevice("display")
display_width = display.getWidth()
display_height = display.getHeight()


#过圈处理
quanshu = 0
last_angle = 0
def angle_quan(angle_now):
    global quanshu,last_angle
    if angle_now < 0.5 and last_angle > 6.0:
        quanshu += 1
    if angle_now > 6.0 and last_angle < 0.5:
        quanshu -= 1    
    last_angle = angle_now
    return quanshu

#角度转换
def angle_trans(angle_real):
    # angle_real = -angle_real
    #将角度转化为0到2pi，方便过圈处理
    if angle_real < 0:
        angle_real += 6.283
    angle_real += angle_quan(angle_real)*3.1415*2.0
    angle_real = angle_real + 3.14159/2.0

    return angle_real

#更新地图数据
def update_map(lidar_data, robot_x, robot_y, robot_theta):
    global occupancy_grid
    for i in range(len(lidar_data)):  # 遍历激光雷达每个角度 
        angle = math.radians(i) # 角度转换为弧度（默认雷达所有线条数为360）
        angle = 3.14159*1.0 - angle  # 调整角度方向
        distance = lidar_data[i]
        
        if distance < 1.3:  # 只记录有限距离的数据
            # 计算障碍物的相对竞技场的坐标
            obs_x = robot_x + distance * math.cos(angle + robot_theta)
            obs_y = robot_y + distance * math.sin(angle + robot_theta)
            print(robot_x,robot_y,distance,angle,robot_theta,obs_x,obs_y)
            # 映射到栅格地图（左下角为0,0）
            map_x = int(obs_x / GRID_RESOLUTION) + MAP_SIZE // 2
            map_y = int(-obs_y / GRID_RESOLUTION) + MAP_SIZE // 2
            #若计算所得障碍坐标合法
            if 0 <= map_x < MAP_SIZE and 0 <= map_y < MAP_SIZE:
                occupancy_grid[map_x, map_y] = 1  # 标记障碍物

# **在 Display 上绘制地图**
def draw_map():
    display.setColor(0xFFFFFF)  # 设置背景色（白色）
    display.fillRectangle(0, 0, display_width, display_height)  # 清空Display

    display.setColor(0x000000)  # 设置障碍物颜色（黑色）
    
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            if occupancy_grid[x, y] == 1:
                # 计算映射到 Display 屏幕上的坐标
                screen_x = int(x * display_width / MAP_SIZE)
                screen_y = int(y * display_height / MAP_SIZE)
                display.drawPixel(screen_x, screen_y)  # 画一个像素点
    







# 定义PID控制器类
class PIDController:
    def __init__(self, p, i, d):
        self.p = p
        self.i = i
        self.d = d
        self.max_integral = 10
        self.prev_error = 0    #上一次误差
        self.integral = 0      #积分
    
    def compute(self, error):     #pid计算
        self.integral += error
        #限制积分最大值
        if self.integral > self.max_integral:
            self.integral = self.max_integral
        elif self.integral < -self.max_integral:
            self.integral = -self.max_integral
        derivative = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.integral + self.d * derivative

# 创建PID控制器实例
pid_turn = PIDController(10.0, 0.0, 0.1)     #控制角度的pid





# 获取轮子
MW_L = supervisor.getDevice("MW_L")
MW_R = supervisor.getDevice("MW_R")
MW_L.setPosition(float('inf'))
MW_R.setPosition(float('inf'))
# 获取传感器
gps = supervisor.getDevice("gps")
gps.enable(timeStep)
imu = supervisor.getDevice("imu")
imu.enable(timeStep)
gyro = supervisor.getDevice("gyro")
gyro.enable(timeStep)
# 获取摄像头
camera_F = supervisor.getDevice("camera_F")
camera_B = supervisor.getDevice("camera_B")
camera_L = supervisor.getDevice("camera_L")
camera_R = supervisor.getDevice("camera_R")
camera_F.enable(timeStep)
camera_B.enable(timeStep)
camera_L.enable(timeStep)
camera_R.enable(timeStep)
# 获取雷达和距离传感器
lidar = supervisor.getDevice("lidar")
lidar.enable(timeStep)
lidar.enablePointCloud()





#声明全局变量
speed_set_L = 0
speed_set_R = 0
# 设置轮子速度
def wheel_run():
    global speed_set_L, speed_set_R
    MW_L.setVelocity(speed_set_L)
    MW_R.setVelocity(speed_set_R)

# 控制小车前进、后退、转向
def move_forward(speed_set):
    global speed_set_L, speed_set_R
    speed_set_L -= speed_set
    speed_set_R -= speed_set

def move_backward(speed_set):
    global speed_set_L, speed_set_R
    speed_set_L += speed_set
    speed_set_R += speed_set

def set_angle(yaw_set, yaw_now):
    global speed_set_L, speed_set_R
    error = yaw_set - yaw_now
    speed_turn_set = pid_turn.compute(error)
    speed_set_L -= speed_turn_set
    speed_set_R += speed_turn_set





# 设置绘图
fig, ax2 = plt.subplots(1,2,figsize=(12,5))  # 创建两个子图
# 设置第二个图形 (IMU yaw 角度)
ax2[0].set_title("IMU Data (Yaw Angle)")
ax2[0].set_xlabel("Time Step")
ax2[0].set_ylabel("Angle (rad)")
line_imu, = ax2[0].plot([], [], label='Yaw Angle')
ax2[0].legend()

# 数据列表
imu_pitch_data_list = []
times = []

ax2[1].imshow(occupancy_grid, cmap="gray_r")   #使用灰度反转颜色映射

# 开启交互模式
plt.ion()

time = 0
# 主控制循环
while supervisor.step(timeStep) != -1:
    # 获取传感器数据（例如 GPS 或加速度计）来做决策
    gps_data = gps.getValues()
    gyro_d = gyro.getValues()
    gyro_data = gyro_d[2]       #yaw角速度   弧度制
    imu_data = imu.getRollPitchYaw()
    lidar_data = lidar.getRangeImage()
    yaw_now = angle_trans(imu_data[2])


    #每次计算前清零
    speed_set_L = 0 
    speed_set_R = 0
    # 根据PID输出调整速度
    # set_angle(1, imu_data[2])
    # move_forward(-2)

    wheel_run()   #控制轮子电机


    #打印debug信息
 #   print(lidar_data[0], lidar_data[90], lidar_data[180], lidar_data[270], lidar_data[359])
    #print(gps_data[0], gps_data[1])

    update_map(lidar_data, gps_data[0], gps_data[1], yaw_now)  # 更新地图数据
    draw_map()
    # 收集数据
    imu_pitch_data_list.append(yaw_now)  # yaw 角度
    times.append(len(times))  # 将当前时间步数作为时间戳
    # 实时更新图形
    line_imu.set_data(times, imu_pitch_data_list)  # 更新IMU pitch数据图形
     # 设置 x 轴范围
    ax2[0].set_xlim(0, len(times))
    # 动态调整 y 轴范围
    ax2[0].relim()  # 重新计算 y 轴的范围
    ax2[0].autoscale_view()  # 自动调整 y 轴范围
    plt.pause(0.001)  # 暂停，让图形实时更新
    
# 关闭交互模式，保持图形窗口display_map
plt.ioff()
ax2[1].imshow(occupancy_grid, cmap="gray_r")   #使用灰度反转颜色映射
plt.show()
    

