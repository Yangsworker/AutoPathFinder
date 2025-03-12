from controller import Supervisor,Display
from controller import Motor, Camera, GPS, InertialUnit, Gyro, Lidar
import matplotlib.pyplot as plt     #导入matplotlib库绘图
import matplotlib.colors as mcolors
import queue
import numpy as np
import math
# 初始化 Supervisor
supervisor = Supervisor()
timeStep = int(supervisor.getBasicTimeStep())


MAP_SIZE = 100
GRID_RESOLUTION = 0.1  #定义每个栅格的大小（相对实际大小）
occupancy_grid = np.zeros((MAP_SIZE, MAP_SIZE))    # 0: 空闲, 1: 障碍  
exploration_grid = np.zeros((MAP_SIZE, MAP_SIZE))  # 0 代表未探索  1 表示已探索
exploration_time = np.full((MAP_SIZE, MAP_SIZE), -300)  # 记录每个网格的探索时间(为探索使用-1表示)
visual_grid = np.zeros((MAP_SIZE, MAP_SIZE))   #可视化数据
display = supervisor.getDevice("display")
display_width = display.getWidth()
display_height = display.getHeight()
path = []   #用来记录机器人移动路径
goal_x = -4   #机器人目标点
goal_y = -4


# **过圈处理**
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
# **角度转换**
def angle_trans(angle_real):
    #将角度转化为0到2pi，方便过圈处理
    if angle_real < 0:
        angle_real += 6.283
    angle_real += angle_quan(angle_real)*3.1415*2.0
    angle_real = angle_real + 3.14159/2.0
    return angle_real
# **更新地图数据**
def update_map(lidar_data, robot_x, robot_y, robot_theta):
    global occupancy_grid
    for i in range(len(lidar_data)):  # 遍历激光雷达每个角度 
        angle = math.radians(i) # 角度转换为弧度（默认雷达所有线条数为360）
        angle = 3.14159*1.0 - angle  # 调整角度方向
        distance = lidar_data[i]        
        if distance < 2.0:  # 只记录有限距离的数据
            # 计算障碍物的相对竞技场的坐标
            obs_x = robot_x + distance * math.cos(angle + robot_theta)
            obs_y = robot_y + distance * math.sin(angle + robot_theta)
            #print(robot_x,robot_y,distance,angle,robot_theta,obs_x,obs_y)
            # 映射到栅格地图（左下角为0,0）
            map_x = int(obs_x / GRID_RESOLUTION) + MAP_SIZE // 2
            map_y = int(-obs_y / GRID_RESOLUTION) + MAP_SIZE // 2
            #若计算所得障碍坐标合法
            if 0 <= map_x < MAP_SIZE and 0 <= map_y < MAP_SIZE:
                occupancy_grid[map_x, map_y] = 1  # 标记障碍物
# **在 Display 上绘制地图**
def draw_map(robot_x,robot_y):
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
    #绘制机器人坐标所在
    display.setColor(0xFF0000)  # 设置机器人颜色（红色）
    display.drawPixel(int(robot_x / GRID_RESOLUTION) + MAP_SIZE // 2, int(-robot_y / GRID_RESOLUTION) + MAP_SIZE // 2)  
    display.setColor(0x0000FF)  # 轨迹颜色（蓝色）
    for (x, y) in path:
        screen_x = int(x / GRID_RESOLUTION) + MAP_SIZE // 2
        screen_y = int(-y / GRID_RESOLUTION) + MAP_SIZE // 2
        display.drawPixel(screen_x, screen_y)


lidar_data = []
def distance(x1, y1, x2, y2):
    x_err = x1 - x2
    y_err = y1 - y2
    sum = x_err*x_err+y_err*y_err
    return np.sqrt(sum)
def is_in_lidar_range(robot_x, robot_y, laser_range, x, y):
    dist = distance(robot_x, robot_y, x, y)
    return dist <= laser_range
# **更新探索区域**
def update_exploration(robot_x, robot_y,yaw_angle):   #填入gps信息
    global lidar_dataeturn 
    x_min = robot_x - 2.0
    y_min = robot_y - 2.0
    x_max = robot_x + 2.0
    y_max = robot_y + 2.0
    x_min = max(-5, x_min)
    y_min = max(-5, y_min)
    x_max = min(5, x_max)
    y_max = min(5, y_max)
    for x in np.arange(x_min, x_max, GRID_RESOLUTION):      #根据角度获取对应雷达范围
        for y in np.arange(y_min, y_max, GRID_RESOLUTION):
            yaw_angle = -yaw_angle
            while yaw_angle < 0:
                yaw_angle += 6.283
            while yaw_angle > 6.283:
                yaw_angle -= 6.283
            angle = np.arctan2(y - robot_y, x - robot_x)   #获取对应雷达角度
            angle = 1.0*3.14159 - angle - yaw_angle
            if angle > 6.283:
                angle -= 6.283
            angle = np.round(np.rad2deg(angle))  # 转换并取整
            angle = min(359, angle)  # 最大值为359
            if is_in_lidar_range(robot_x, robot_y, get_lidar_dis(angle), x, y):
                x_t = int(x / GRID_RESOLUTION) + MAP_SIZE // 2
                y_t = int(-y / GRID_RESOLUTION) + MAP_SIZE // 2
                x_t = max(0, min(MAP_SIZE - 1, x_t))
                y_t = max(0, min(MAP_SIZE - 1, y_t))
                exploration_grid[x_t, y_t] = 1  # 标记为已探索
                exploration_time[x_t, y_t] += 1  # 记录探索时间
                exploration_time[x_t, y_t] = min(-50, exploration_time[x_t, y_t])  # 最大值为1
def get_lidar_dis(angle):
    global lidar_data
    if lidar_data[int(angle)] > 1.8:
        return 1.8
    else:
        return lidar_data[int(angle)]



def test(x,y,robot_x,robot_y,yaw_angle):
    yaw_angle = -yaw_angle
    while yaw_angle < 0:
        yaw_angle += 6.283
    while yaw_angle > 6.283:
        yaw_angle -= 6.283
    angle = math.atan2(y - robot_y, x - robot_x)   #获取对应雷达角度
    print(angle)
    angle = 1.0*3.14159 - angle - yaw_angle
    
    if angle > 6.283:
        angle -= 6.283
    angle = np.round(np.rad2deg(angle))  # 转换并取整
    angle = min(359, angle)  # 最大值为359
    print(distance(x,y,robot_x,robot_y),get_lidar_dis(angle),(distance(x,y,robot_x,robot_y) < get_lidar_dis(angle)))
    if is_in_lidar_range(robot_x, robot_y, get_lidar_dis(angle), x, y):
        print("YES")
        x = int(x / GRID_RESOLUTION) + MAP_SIZE // 2
        y = int(-y / GRID_RESOLUTION) + MAP_SIZE // 2
        exploration_grid[x, y] = 1  # 标记为已探索
        exploration_time[x, y] += 1  # 记录探索时间
        exploration_time[x, y] = min(-50, exploration_time[x, y])  # 最大值为1
    print(angle,exploration_time[int(x), int(y)])






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
pid_turn = PIDController(1.0, 0.0, 0.1)     #控制角度的pid



def mapToreal(x,y):
    return (int(x / GRID_RESOLUTION) + MAP_SIZE // 2,int(-y / GRID_RESOLUTION) + MAP_SIZE // 2)
def realTomap(x,y):
    return ((x - MAP_SIZE // 2)*GRID_RESOLUTION, (y - MAP_SIZE // 2)*GRID_RESOLUTION*-1.0)



# 获取轮子
MW_L = supervisor.getDevice("MW_L")
MW_R = supervisor.getDevice("MW_R")
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
# 设置轮子速度(需要先把位置设置为inf)
def wheel_run():
    global speed_set_L, speed_set_R
    MW_L.setPosition(float('inf'))
    MW_R.setPosition(float('inf'))
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
    speed_turn_set = max(-4.0, min(4.0, speed_turn_set))
    speed_set_L -= speed_turn_set
    speed_set_R += speed_turn_set
def wheel_stop():
    global speed_set_L, speed_set_R
    speed_set_L = 0
    speed_set_R = 0




visited = []
path_one = []
queue = []
found = 0
# **广度优先搜索 (BFS) 计算路径**
def bfs(start):
    global found,visited,exploration_grid
    
    path_one.clear()
    visited.clear()
    # 初始化队列，从起始点开始
    queue.append([start])  # 初始路径为[start]
    path_one.append(start)
    visited.append(start)  # 标记为已访问
    x_time_max = start[0] + 50
    x_time_min = start[0] - 50
    y_time_max = start[1] + 50
    y_time_min = start[1] - 50
    while queue:
    #while found < 100:
        path = queue.pop(0)   # 取出队列中的第一个路径
        x, y = path[-1]       # 获取路径的最后一个点

        # 扫描当前点的周围区域（四个方向）
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        neighbors = [(nx, ny) for nx, ny in neighbors if max(1,x_time_min) <= nx <= min(98,x_time_max) and max(1,y_time_min) <= ny <= min(98,y_time_max)]
        for neighbor in neighbors:
            x_t,y_t = neighbor
            if neighbor not in visited and occupancy_grid[int(x_t),int(y_t)] == 0:  # 未访问过且不是障碍物
                visited.append(neighbor)    #满足以上条件即加入路径
                if exploration_grid[int(x_t),int(y_t)] == 0:
                    path_one.append(neighbor)
                #print(path,found,neighbor)
                queue.append([neighbor])
    #print(path_one)
    found = 1
    return visited








 #** 计算目标角度 **#
def get_angle(robot_x,robot_y,goal_x,goal_y,yaw_angle):
    angle = np.arctan2(goal_y - robot_y, goal_x - robot_x)
    angle_set = angle+(quanshu)*3.1415*2.0
    if abs(angle_set - yaw_angle) > 3.1415:
        if angle_set > yaw_angle:
            angle_set = angle_set - 3.1415*2.0
        else:
            angle_set = angle_set + 3.1415*2.0
    return angle_set
def ifget_goal(robot_x,robot_y,goal_x,goal_y):
    if distance(robot_x,robot_y,goal_x,goal_y) < 0.1:
        return True
    else:
        return False
    







# 设置绘图
fig, ax = plt.subplots(figsize=(8,8))  # 创建两个子图
# 开启交互模式
plt.ion()
time_run = 0
time_Cnt = 0
front_distance = 0
lidar_angle = 0
yaw_check = 0 #在遇到障碍物时，记录当前yaw角度
yep = 0
goal = (0,0)
# 主控制循环
while supervisor.step(timeStep) != -1:
    time_run = time_run + timeStep / 1000.0
    # 获取传感器数据（例如 GPS 或加速度计）来做决策
    gps_data = gps.getValues()
    gyro_d = gyro.getValues()
    gyro_data = gyro_d[2]       #yaw角速度   弧度制
    imu_data = imu.getRollPitchYaw()
    lidar_data = lidar.getRangeImage()
    yaw_now = angle_trans(imu_data[2])
    update_exploration(gps_data[0], gps_data[1],yaw_now)       # 更新探索区域
    update_map(lidar_data, gps_data[0], gps_data[1], yaw_now)  # 更新地图数据

    #每次计算前清零
    speed_set_L = 0 
    speed_set_R = 0
    # **检查障碍物**
    front_distance = lidar_data[180]
    for i in range(135, 225, 1):
        if lidar_data[i] < front_distance:
            front_distance = lidar_data[i]
            lidar_angle = i   #记录最近的障碍物角度
    #print(front_distance,lidar_angle)        

    
    

    if time_Cnt > 100 or time_Cnt == 0:
        wheel_stop()
        wheel_run()
        print("以刷新点",found)
        start = mapToreal(gps_data[0], gps_data[1])
        bfs(start)
        time_Cnt = 0
    time_Cnt += 1
    for (x,y) in path_one:
        yep = 0
        if exploration_grid[int(x),int(y)] == 0:
            goal = (x,y)
            goal = realTomap(goal[0],goal[1])
            yep = 1
            break
    if yep == 0:
        print("FINISH")
        break  
    #else:
        #print(goal)
    #test(2,3,gps_data[0],gps_data[1],yaw_now)        
    
    #控制机器人运动至所到位置 
    set_angle(get_angle(gps_data[0],gps_data[1],goal[0],goal[1],yaw_now),yaw_now)
    if front_distance < 0.5:
        angle_turn = (180 - lidar_angle)/180.0*3.1415*6.0
        if lidar_angle < 180:
            speed_set_L-=angle_turn
            speed_set_R+=angle_turn
        else:
            speed_set_L+=angle_turn
            speed_set_R-=angle_turn
    
    move_forward(6)
    wheel_run()   #控制轮子电机

    draw_map(gps_data[0], gps_data[1])                         # 在 Display 上绘制地图

    
    #打印debug信息

    # 收集数据
    path.append([gps_data[0], gps_data[1]])  # 记录机器人移动路径
    
# 关闭交互模式，保持图形窗口display_map
plt.ioff()
for x in range(MAP_SIZE):
    for y in range(MAP_SIZE):
        if occupancy_grid[x, y] == 1:
            visual_grid[x, y] = 50  # 障碍物（黑色）
        elif exploration_grid[x, y] == 1:
            visual_grid[x, y] = exploration_time[x,y]  # 已探索区域（浅色）
        else:
            visual_grid[x, y] = -300  # 未探索区域（深灰色）
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    "custom_map", [(0, "black"), (0.7, "gray"), (1, "red")]
)    #自定义颜色映射Optimize visual effects
im = ax.imshow(visual_grid, cmap=cmap_custom,origin="lower",vmin=-300,vmax=50)   #使用灰度反转颜色映射
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Exploration Level")
plt.show()
    

# **todo: 需要先实现给定点，机器人导航至目标点，误差小于一定值之间返回；
# 使用BFS算法计算单个路径规划
# 实时进行广度优先搜索**
