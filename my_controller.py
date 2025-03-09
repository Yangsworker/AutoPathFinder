from controller import Supervisor
from controller import Motor, Camera, GPS, InertialUnit, Gyro, Lidar
import matplotlib.pyplot as plt     #导入matplotlib库绘图
import math

# 初始化 Supervisor
supervisor = Supervisor()
timeStep = int(supervisor.getBasicTimeStep())





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
fig, ax2 = plt.subplots()  # 创建两个子图，2行1列
# 设置第二个图形 (IMU yaw 角度)
ax2.set_title("IMU Data (Yaw Angle)")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Angle (rad)")
line_imu, = ax2.plot([], [], label='Yaw Angle')
ax2.legend()
# 开启交互模式
plt.ion()
# 数据列表
imu_pitch_data_list = []
times = []





# 主控制循环
while supervisor.step(timeStep) != -1:
    # 获取传感器数据（例如 GPS 或加速度计）来做决策
    gps_data = gps.getValues()
    gyro_d = gyro.getValues()
    gyro_data = gyro_d[2]       #yaw角速度   弧度制
    imu_data = imu.getRollPitchYaw()
    lidar_data = lidar.getRangeImage()
    
    #每次计算前清零
    speed_set_L = 0 
    speed_set_R = 0


    # 根据PID输出调整速度
    set_angle(1, imu_data[2])
    move_forward(-2)

    wheel_run()   #控制轮子电机




    # 收集数据
    imu_pitch_data_list.append(imu_data[2])  # yaw 角度
    times.append(len(times))  # 将当前时间步数作为时间戳
    
    # 实时更新图形
    line_imu.set_data(times, imu_pitch_data_list)  # 更新IMU pitch数据图形
     # 设置 x 轴范围
    ax2.set_xlim(0, len(times))
    # 动态调整 y 轴范围
    ax2.relim()  # 重新计算 y 轴的范围
    ax2.autoscale_view()  # 自动调整 y 轴范围
    plt.pause(0.001)  # 暂停，让图形实时更新
    
# 关闭交互模式，保持图形窗口
plt.ioff()
plt.show()
    

