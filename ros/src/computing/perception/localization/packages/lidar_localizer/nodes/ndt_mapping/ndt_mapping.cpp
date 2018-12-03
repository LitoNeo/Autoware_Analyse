/*
 *  Copyright (c) 2015, Nagoya University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name of Autoware nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 Localization and mapping program using Normal Distributions Transform

 Yuki KITSUKAWA
 */

#define OUTPUT  // If you want to output "position_log.txt", "#define OUTPUT".

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ndt_cpu/NormalDistributionsTransform.h>
#include <pcl/registration/ndt.h>
#ifdef CUDA_FOUND
#include <ndt_gpu/NormalDistributionsTransform.h>
#endif
#ifdef USE_PCL_OPENMP
#include <pcl_omp_registration/ndt.h>
#endif

#include <autoware_config_msgs/ConfigNdtMapping.h>
#include <autoware_config_msgs/ConfigNdtMappingOutput.h>

#include <time.h>

struct pose
{
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

enum class MethodType  // 定义使用NDT的方法
{
  PCL_GENERIC = 0,  // 使用pcl自带ndt算法
  PCL_ANH = 1,      // 使用ndt_cpu方式的ndt算法
  PCL_ANH_GPU = 2,  // 使用ndt_gpu方式的算法
  PCL_OPENMP = 3,   // 使用ndt_omp方式的算法
};
static MethodType _method_type = MethodType::PCL_GENERIC;  // 默认赋值_method_type为pcl自带方法

// global variables  // 定义各种可能用到的位置pose
static pose previous_pose, guess_pose, guess_pose_imu, guess_pose_odom, guess_pose_imu_odom
static pose current_pose, current_pose_imu, current_pose_odom, current_pose_imu_odom
static pose ndt_pose, added_pose, localizer_pose;

static ros::Time current_scan_time;
static ros::Time previous_scan_time;
static ros::Duration scan_duration;

// 定义各种差异值(两次采集数据之间的差异,包括点云位置差异,imu差异,odom差异,imu-odom差异)
static double diff = 0.0;
static double diff_x = 0.0, diff_y = 0.0, diff_z = 0.0, diff_yaw;  // current_pose - previous_pose // 定义两帧点云差异值 --以确定是否更新点云等
static double offset_imu_x, offset_imu_y, offset_imu_z, offset_imu_roll, offset_imu_pitch, offset_imu_yaw;
static double offset_odom_x, offset_odom_y, offset_odom_z, offset_odom_roll, offset_odom_pitch, offset_odom_yaw;
static double offset_imu_odom_x, offset_imu_odom_y, offset_imu_odom_z, offset_imu_odom_roll, offset_imu_odom_pitch,
    offset_imu_odom_yaw;

// 定义速度值 --包括实际速度值,和imu取到的速度值
static double current_velocity_x = 0.0;
static double current_velocity_y = 0.0;
static double current_velocity_z = 0.0;

static double current_velocity_imu_x = 0.0;
static double current_velocity_imu_y = 0.0;
static double current_velocity_imu_z = 0.0;

static pcl::PointCloud<pcl::PointXYZI> map;  // 此处定义地图  --global?local?

static pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
static cpu::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> anh_ndt;  // cpu方式  --可以直接调用吗??
#ifdef CUDA_FOUND
static gpu::GNormalDistributionsTransform anh_gpu_ndt;
#endif
#ifdef USE_PCL_OPENMP
static pcl_omp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> omp_ndt;
#endif

// Default values  // 公共ndt参数设置
static int max_iter = 30;        // Maximum iterations
static float ndt_res = 1.0;      // Resolution
static double step_size = 0.1;   // Step size
static double trans_eps = 0.01;  // Transformation epsilon

// Leaf size of VoxelGrid filter.   ---该处定义的是一个默认的栅格大小
static double voxel_leaf_size = 2.0;

// 定义各种时间变量  -- 用以计算帧率的吧  // 同时由于使用了static,因此全部赋值为0
static ros::Time callback_start, callback_end, t1_start, t1_end, t2_start, t2_end, t3_start, t3_end, t4_start, t4_end,
    t5_start, t5_end;
static ros::Duration d_callback, d1, d2, d3, d4, d5;

// 定义Publisher
static ros::Publisher ndt_map_pub;        // 地图发布
static ros::Publisher current_pose_pub;   // 位置发布
static ros::Publisher guess_pose_linaer_pub;  // TODO:这是个啥发布????
static geometry_msgs::PoseStamped current_pose_msg, guess_pose_msg;

static ros::Publisher ndt_stat_pub;
static std_msgs::Bool ndt_stat_msg;  // 确认是否是ndt的第一帧图像 bool类型

static int initial_scan_loaded = 0;

static Eigen::Matrix4f gnss_transform = Eigen::Matrix4f::Identity();  // 保存GPS信号的变量

static double min_scan_range = 5.0;  // min和max用于第一次进行点云过滤(截取两个同心圆内的)
static double max_scan_range = 200.0;
static double min_add_scan_shift = 1.0;  // 定义将点云添加到locaMap里的最小间隔值  --应该是添加到localMap吧??

static double _tf_x, _tf_y, _tf_z, _tf_roll, _tf_pitch, _tf_yaw;
static Eigen::Matrix4f tf_btol, tf_ltob;  // ?????

// 重要变量参数 :: 用以指示是否使用imu,是否使用odom
static bool _use_imu = false;  
static bool _use_odom = false;
static bool _imu_upside_down = false;  // 用以解决坐标系方向(正负变换)问题 (比如x变更为-x等)

static bool _incremental_voxel_update = false;  // ??????????

static std::string _imu_topic = "/imu_raw";  // 定义imu消息的topic

// 设置变量,用以接受ndt配准后的参数
static double fitness_score;
static bool has_converged;
static int final_num_iteration;
static double transformation_probability;

// 设置变量,用以接受imu和odom消息
static sensor_msgs::Imu imu;
static nav_msgs::Odometry odom;

// 读写文件流相关
static std::ofstream ofs; 写csv文件
static std::string filename;


// param_callback 用以接受一个带输出的msg消息,并取该消息的内容设置NDT参数  -- 聪明
//TODO: 1: 传入的是一个msg格式的文件,即把参数文件定义为msg消息
static void param_callback(const autoware_config_msgs::ConfigNdtMapping::ConstPtr& input)
{
  ndt_res = input->resolution;
  step_size = input->step_size;
  trans_eps = input->trans_epsilon;
  max_iter = input->max_iterations;
  voxel_leaf_size = input->leaf_size;
  min_scan_range = input->min_scan_range;
  max_scan_range = input->max_scan_range;
  min_add_scan_shift = input->min_add_scan_shift;

  std::cout << "param_callback" << std::endl;
  std::cout << "ndt_res: " << ndt_res << std::endl;
  std::cout << "step_size: " << step_size << std::endl;
  std::cout << "trans_epsilon: " << trans_eps << std::endl;
  std::cout << "max_iter: " << max_iter << std::endl;
  std::cout << "voxel_leaf_size: " << voxel_leaf_size << std::endl;
  std::cout << "min_scan_range: " << min_scan_range << std::endl;
  std::cout << "max_scan_range: " << max_scan_range << std::endl;
  std::cout << "min_add_scan_shift: " << min_add_scan_shift << std::endl;
}

// TODO: 2 函数传入一个消息,ConfigNdtMappingOutPut,该消息只有三个字段 header,filename, filter_size
// 其中fileter_size指的是体素化栅格的leaf_size
// 即根据传入的消息中所带filter_size的大小,将map进行栅格话处理,并发布栅格后的点云
// 注:不改变map中的点,而是使用map的拷贝(map_ptr)进行操作
static void output_callback(const autoware_config_msgs::ConfigNdtMappingOutput::ConstPtr& input)
{
  double filter_res = input->filter_res;
  std::string filename = input->filename;
  std::cout << "output_callback" << std::endl;
  std::cout << "filter_res: " << filter_res << std::endl;
  std::cout << "filename: " << filename << std::endl;

  // map_ptr是map的一个拷贝
  pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZI>(map));

  pcl::PointCloud<pcl::PointXYZI>::Ptr map_filtered(new pcl::PointCloud<pcl::PointXYZI>());

  map_ptr->header.frame_id = "map";
  map_filtered->header.frame_id = "map";

  // 定义sensor_msgs用以发布点云地图(voxel体素化栅格后的地图)
  sensor_msgs::PointCloud2::Ptr map_msg_ptr(new sensor_msgs::PointCloud2);

  // Apply voxelgrid filter
  if (filter_res == 0.0)
  {
    std::cout << "Original: " << map_ptr->points.size() << " points." << std::endl;
    pcl::toROSMsg(*map_ptr, *map_msg_ptr);
  }
  else
  {
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(filter_res, filter_res, filter_res);
    voxel_grid_filter.setInputCloud(map_ptr);
    voxel_grid_filter.filter(*map_filtered);
    std::cout << "Original: " << map_ptr->points.size() << " points." << std::endl;
    std::cout << "Filtered: " << map_filtered->points.size() << " points." << std::endl;
    pcl::toROSMsg(*map_filtered, *map_msg_ptr);
  }

  ndt_map_pub.publish(*map_msg_ptr);  // 此处发布地图!!! 即讲全局地图进行栅格化降采样后的地图,但是保存的map仍然是稠密点云的map!!!!!!!!

  // Writing Point Cloud data to PCD file  // 保存点云图像到pcd文件!!!
  if (filter_res == 0.0)
  {
    pcl::io::savePCDFileASCII(filename, *map_ptr);
    std::cout << "Saved " << map_ptr->points.size() << " data points to " << filename << "." << std::endl;
  }
  else
  {
    pcl::io::savePCDFileASCII(filename, *map_filtered);
    std::cout << "Saved " << map_filtered->points.size() << " data points to " << filename << "." << std::endl;
  }
}

// TODO: 3-1 imu和odom信息进行融合处理
// 最终的目的是更改guess_pose
static void imu_odom_calc(ros::Time current_time)
{
  static ros::Time previous_time = current_time;
  double diff_time = (current_time - previous_time).toSec();  // static声明的变量只会在第一次使用时被声明,因此不会被覆盖

  // imu信息处理,计算 -- imu只使用陀螺仪,即 只输出转角信息roll,pitch,yaw
  double diff_imu_roll = imu.angular_velocity.x * diff_time;
  double diff_imu_pitch = imu.angular_velocity.y * diff_time;
  double diff_imu_yaw = imu.angular_velocity.z * diff_time;

  current_pose_imu_odom.roll += diff_imu_roll;  // 更新current_pose_imu_odom相关,作为历史记录
  current_pose_imu_odom.pitch += diff_imu_pitch;
  current_pose_imu_odom.yaw += diff_imu_yaw;

  // odom信息处理,计算 -- xyz移动距离的计算,融合odom的速度(位移)信息和imu的转角信息
  double diff_distance = odom.twist.twist.linear.x * diff_time;
  offset_imu_odom_x += diff_distance * cos(-current_pose_imu_odom.pitch) * cos(current_pose_imu_odom.yaw);
  offset_imu_odom_y += diff_distance * cos(-current_pose_imu_odom.pitch) * sin(current_pose_imu_odom.yaw);
  offset_imu_odom_z += diff_distance * sin(-current_pose_imu_odom.pitch);
  
  offset_imu_odom_roll += diff_imu_roll;
  offset_imu_odom_pitch += diff_imu_pitch;
  offset_imu_odom_yaw += diff_imu_yaw;

  // ==> 最终的目的是融合imu和odom输出一个guess_pose
  // 注:guess_pose是在previous_pose基础上叠加一个offset,包括xyz的和rpy的
  // xyz的offset需要融合imu的转角和odom的速度(位移)
  // rpy的offset直接采用imu的rpy偏差值
  guess_pose_imu_odom.x = previous_pose.x + offset_imu_odom_x;
  guess_pose_imu_odom.y = previous_pose.y + offset_imu_odom_y;
  guess_pose_imu_odom.z = previous_pose.z + offset_imu_odom_z;
  guess_pose_imu_odom.roll = previous_pose.roll + offset_imu_odom_roll;
  guess_pose_imu_odom.pitch = previous_pose.pitch + offset_imu_odom_pitch;
  guess_pose_imu_odom.yaw = previous_pose.yaw + offset_imu_odom_yaw;

  previous_time = current_time;
}

// TODO: 3-2 单纯使用odom时的里程计计算
// nav_msgs::Odometry消息包括
//       header
//       geometry_msgs::PoseWithConvariance(Point<xyz>,Quaternion<xyzw>)
//       geometry_msgs::TwiseWithConvariance(linear,angular).   vector格式
static void odom_calc(ros::Time current_time)
{
  static ros::Time previous_time = current_time;
  double diff_time = (current_time - previous_time).toSec();

  double diff_odom_roll = odom.twist.twist.angular.x * diff_time;
  double diff_odom_pitch = odom.twist.twist.angular.y * diff_time;
  double diff_odom_yaw = odom.twist.twist.angular.z * diff_time;

  current_pose_odom.roll += diff_odom_roll;
  current_pose_odom.pitch += diff_odom_pitch;
  current_pose_odom.yaw += diff_odom_yaw;

  double diff_distance = odom.twist.twist.linear.x * diff_time;
  offset_odom_x += diff_distance * cos(-current_pose_odom.pitch) * cos(current_pose_odom.yaw);
  offset_odom_y += diff_distance * cos(-current_pose_odom.pitch) * sin(current_pose_odom.yaw);
  offset_odom_z += diff_distance * sin(-current_pose_odom.pitch);

  offset_odom_roll += diff_odom_roll;
  offset_odom_pitch += diff_odom_pitch;
  offset_odom_yaw += diff_odom_yaw;

  guess_pose_odom.x = previous_pose.x + offset_odom_x;
  guess_pose_odom.y = previous_pose.y + offset_odom_y;
  guess_pose_odom.z = previous_pose.z + offset_odom_z;
  guess_pose_odom.roll = previous_pose.roll + offset_odom_roll;
  guess_pose_odom.pitch = previous_pose.pitch + offset_odom_pitch;
  guess_pose_odom.yaw = previous_pose.yaw + offset_odom_yaw;

  previous_time = current_time;
}

// TODO: 3-3 纯imu消息处理
static void imu_calc(ros::Time current_time)
{
  static ros::Time previous_time = current_time;
  double diff_time = (current_time - previous_time).toSec();

  double diff_imu_roll = imu.angular_velocity.x * diff_time;
  double diff_imu_pitch = imu.angular_velocity.y * diff_time;
  double diff_imu_yaw = imu.angular_velocity.z * diff_time;

  current_pose_imu.roll += diff_imu_roll;
  current_pose_imu.pitch += diff_imu_pitch;
  current_pose_imu.yaw += diff_imu_yaw;

  // 对imu由于不平衡造成的补偿问题,在这里解决
  // start1
  double accX1 = imu.linear_acceleration.x;
  double accY1 = std::cos(current_pose_imu.roll) * imu.linear_acceleration.y -
                 std::sin(current_pose_imu.roll) * imu.linear_acceleration.z;
  double accZ1 = std::sin(current_pose_imu.roll) * imu.linear_acceleration.y +
                 std::cos(current_pose_imu.roll) * imu.linear_acceleration.z;

  double accX2 = std::sin(current_pose_imu.pitch) * accZ1 + std::cos(current_pose_imu.pitch) * accX1;
  double accY2 = accY1;
  double accZ2 = std::cos(current_pose_imu.pitch) * accZ1 - std::sin(current_pose_imu.pitch) * accX1;

  double accX = std::cos(current_pose_imu.yaw) * accX2 - std::sin(current_pose_imu.yaw) * accY2;
  double accY = std::sin(current_pose_imu.yaw) * accX2 + std::cos(current_pose_imu.yaw) * accY2;
  double accZ = accZ2;
  // end1

  offset_imu_x += current_velocity_imu_x * diff_time + accX * diff_time * diff_time / 2.0;
  offset_imu_y += current_velocity_imu_y * diff_time + accY * diff_time * diff_time / 2.0;
  offset_imu_z += current_velocity_imu_z * diff_time + accZ * diff_time * diff_time / 2.0;

  current_velocity_imu_x += accX * diff_time;
  current_velocity_imu_y += accY * diff_time;
  current_velocity_imu_z += accZ * diff_time;

  offset_imu_roll += diff_imu_roll;
  offset_imu_pitch += diff_imu_pitch;
  offset_imu_yaw += diff_imu_yaw;

  guess_pose_imu.x = previous_pose.x + offset_imu_x;
  guess_pose_imu.y = previous_pose.y + offset_imu_y;
  guess_pose_imu.z = previous_pose.z + offset_imu_z;
  guess_pose_imu.roll = previous_pose.roll + offset_imu_roll;
  guess_pose_imu.pitch = previous_pose.pitch + offset_imu_pitch;
  guess_pose_imu.yaw = previous_pose.yaw + offset_imu_yaw;

  previous_time = current_time;
}


// TODO: 4 这两个个是干啥的??? -- 转换弧度值?解决弧度制大于pi的问题?
// start2
static double wrapToPm(double a_num, const double a_max)
{
  if (a_num >= a_max)
  {
    a_num -= 2.0 * a_max;
  }
  return a_num;
}

static double wrapToPmPi(double a_angle_rad)
{
  return wrapToPm(a_angle_rad, M_PI);
}
// end2

// 这个又是干啥的???
static double calcDiffForRadian(const double lhs_rad, const double rhs_rad)
{
  double diff_rad = lhs_rad - rhs_rad;
  if (diff_rad >= M_PI)
    diff_rad = diff_rad - 2 * M_PI;
  else if (diff_rad < -M_PI)
    diff_rad = diff_rad + 2 * M_PI;
  return diff_rad;
}

// odom_callback函数
static void odom_callback(const nav_msgs::Odometry::ConstPtr& input)
{
  // std::cout << __func__ << std::endl;

  odom = *input;
  odom_calc(input->header.stamp);
}

// 对imu传入的数据进行左右变换(正负变换,即解决坐标系方向问题)
static void imuUpsideDown(const sensor_msgs::Imu::Ptr input)
{
  double input_roll, input_pitch, input_yaw;

  tf::Quaternion input_orientation;
  tf::quaternionMsgToTF(input->orientation, input_orientation);
  tf::Matrix3x3(input_orientation).getRPY(input_roll, input_pitch, input_yaw);

  input->angular_velocity.x *= -1;
  input->angular_velocity.y *= -1;
  input->angular_velocity.z *= -1;

  input->linear_acceleration.x *= -1;
  input->linear_acceleration.y *= -1;
  input->linear_acceleration.z *= -1;

  input_roll *= -1;
  input_pitch *= -1;
  input_yaw *= -1;

  input->orientation = tf::createQuaternionMsgFromRollPitchYaw(input_roll, input_pitch, input_yaw);
}

// imu_callback函数
static void imu_callback(const sensor_msgs::Imu::Ptr& input)
{
  // std::cout << __func__ << std::endl;

  if (_imu_upside_down)  // _imu_upside_down指示是否进行imu的正负变换
    imuUpsideDown(input);

  const ros::Time current_time = input->header.stamp;
  static ros::Time previous_time = current_time;
  const double diff_time = (current_time - previous_time).toSec();

      // 解析imu消息,获得rpy
  double imu_roll, imu_pitch, imu_yaw;
  tf::Quaternion imu_orientation;
  tf::quaternionMsgToTF(input->orientation, imu_orientation);
  tf::Matrix3x3(imu_orientation).getRPY(imu_roll, imu_pitch, imu_yaw);

  imu_roll = wrapToPmPi(imu_roll);  // 调整,防止超过PI(180°)  --保持在±180°内
  imu_pitch = wrapToPmPi(imu_pitch);
  imu_yaw = wrapToPmPi(imu_yaw);

  // 难道不是等于0 ?????
  static double previous_imu_roll = imu_roll, previous_imu_pitch = imu_pitch, previous_imu_yaw = imu_yaw; 
  const double diff_imu_roll = calcDiffForRadian(imu_roll, previous_imu_roll);
  const double diff_imu_pitch = calcDiffForRadian(imu_pitch, previous_imu_pitch);
  const double diff_imu_yaw = calcDiffForRadian(imu_yaw, previous_imu_yaw);
  //

  imu.header = input->header;
  imu.linear_acceleration.x = input->linear_acceleration.x;
  // imu.linear_acceleration.y = input->linear_acceleration.y;
  // imu.linear_acceleration.z = input->linear_acceleration.z;
  imu.linear_acceleration.y = 0;
  imu.linear_acceleration.z = 0;

  if (diff_time != 0)
  {
    imu.angular_velocity.x = diff_imu_roll / diff_time;
    imu.angular_velocity.y = diff_imu_pitch / diff_time;
    imu.angular_velocity.z = diff_imu_yaw / diff_time;
  }
  else
  {
    imu.angular_velocity.x = 0;
    imu.angular_velocity.y = 0;
    imu.angular_velocity.z = 0;
  }

  imu_calc(input->header.stamp);

  previous_time = current_time;
  previous_imu_roll = imu_roll;
  previous_imu_pitch = imu_pitch;
  previous_imu_yaw = imu_yaw;
}
// ########################################################################################################
// TODO: 主要的函数--对传入的点云进行处理
static void points_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{
  double r;
  pcl::PointXYZI p;
  pcl::PointCloud<pcl::PointXYZI> tmp, scan;
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
  tf::Quaternion q;

  Eigen::Matrix4f t_localizer(Eigen::Matrix4f::Identity()); // 
  Eigen::Matrix4f t_base_link(Eigen::Matrix4f::Identity()); // 

  static tf::TransformBroadcaster br;  // 
  tf::Transform transform;             //

  current_scan_time = input->header.stamp;

  pcl::fromROSMsg(*input, tmp);

  for (pcl::PointCloud<pcl::PointXYZI>::const_iterator item = tmp.begin(); item != tmp.end(); item++)
  {
    p.x = (double)item->x;
    p.y = (double)item->y;
    p.z = (double)item->z;
    p.intensity = (double)item->intensity;

    r = sqrt(pow(p.x, 2.0) + pow(p.y, 2.0));
    if (min_scan_range < r && r < max_scan_range)
    {
      scan.push_back(p);
    }
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));

  // 如果是全局的第一帧点云,则 Add initial point cloud to velodyne_map
  if (initial_scan_loaded == 0)
  {
    pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, tf_btol);  // 按照tf_btol的规则转换到transformed_scan_ptr // tf_btof会在main函数中定义
    map += *transformed_scan_ptr;
    initial_scan_loaded = 1;
  }

  // Apply voxelgrid filter
  pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
  voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
  voxel_grid_filter.setInputCloud(scan_ptr);
  voxel_grid_filter.filter(*filtered_scan_ptr);

  pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZI>(map));  // 重要作用,用以保存降采样后的全局地图

  if (_method_type == MethodType::PCL_GENERIC)
  {
    ndt.setTransformationEpsilon(trans_eps);
    ndt.setStepSize(step_size);
    ndt.setResolution(ndt_res);
    ndt.setMaximumIterations(max_iter);
    ndt.setInputSource(filtered_scan_ptr);
  }
  else if (_method_type == MethodType::PCL_ANH)
  {
    anh_ndt.setTransformationEpsilon(trans_eps);
    anh_ndt.setStepSize(step_size);
    anh_ndt.setResolution(ndt_res);
    anh_ndt.setMaximumIterations(max_iter);
    anh_ndt.setInputSource(filtered_scan_ptr);
  }
#ifdef CUDA_FOUND
  else if (_method_type == MethodType::PCL_ANH_GPU)
  {
    anh_gpu_ndt.setTransformationEpsilon(trans_eps);
    anh_gpu_ndt.setStepSize(step_size);
    anh_gpu_ndt.setResolution(ndt_res);
    anh_gpu_ndt.setMaximumIterations(max_iter);
    anh_gpu_ndt.setInputSource(filtered_scan_ptr);
  }
#endif
#ifdef USE_PCL_OPENMP
  else if (_method_type == MethodType::PCL_OPENMP)
  {
    omp_ndt.setTransformationEpsilon(trans_eps);
    omp_ndt.setStepSize(step_size);
    omp_ndt.setResolution(ndt_res);
    omp_ndt.setMaximumIterations(max_iter);
    omp_ndt.setInputSource(filtered_scan_ptr);
  }
#endif

  static bool is_first_map = true;  //   --答:static关键字的功能之一:即保证static声明的变量只被初始化一次
  if (is_first_map == true)
  {
    if (_method_type == MethodType::PCL_GENERIC)
      ndt.setInputTarget(map_ptr);
    else if (_method_type == MethodType::PCL_ANH)
      anh_ndt.setInputTarget(map_ptr);
#ifdef CUDA_FOUND
    else if (_method_type == MethodType::PCL_ANH_GPU)
      anh_gpu_ndt.setInputTarget(map_ptr);
#endif
#ifdef USE_PCL_OPENMP
    else if (_method_type == MethodType::PCL_OPENMP)
      omp_ndt.setInputTarget(map_ptr);
#endif
    is_first_map = false;
  }

  guess_pose.x = previous_pose.x + diff_x;  // 初始时diff_x等都为0
  guess_pose.y = previous_pose.y + diff_y;
  guess_pose.z = previous_pose.z + diff_z;
  guess_pose.roll = previous_pose.roll;
  guess_pose.pitch = previous_pose.pitch;
  guess_pose.yaw = previous_pose.yaw + diff_yaw;

  // 根据是否使用imu和odom,按照不同方式更新guess_pose
  if (_use_imu == true && _use_odom == true)
    imu_odom_calc(current_scan_time);
  if (_use_imu == true && _use_odom == false)
    imu_calc(current_scan_time);
  if (_use_imu == false && _use_odom == true)
    odom_calc(current_scan_time);
  
  // start2 只是为了把上面不同方式的guess_pose都标准化成guess_pose_for_ndt,为了后续操作方便 
  pose guess_pose_for_ndt;
  if (_use_imu == true && _use_odom == true)
    guess_pose_for_ndt = guess_pose_imu_odom;
  else if (_use_imu == true && _use_odom == false)
    guess_pose_for_ndt = guess_pose_imu;
  else if (_use_imu == false && _use_odom == true)
    guess_pose_for_ndt = guess_pose_odom;
  else
    guess_pose_for_ndt = guess_pose;
  // end2

  // start3 以下:根据guess_pose_for_ndt 来计算初始变换矩阵guess_init -- 针对TargetSource
  Eigen::AngleAxisf init_rotation_x(guess_pose_for_ndt.roll, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf init_rotation_y(guess_pose_for_ndt.pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf init_rotation_z(guess_pose_for_ndt.yaw, Eigen::Vector3f::UnitZ());

  Eigen::Translation3f init_translation(guess_pose_for_ndt.x, guess_pose_for_ndt.y, guess_pose_for_ndt.z);

  Eigen::Matrix4f init_guess =
      (init_translation * init_rotation_z * init_rotation_y * init_rotation_x).matrix() * tf_btol;  // tf_btof
  // end3

  t3_end = ros::Time::now();
  d3 = t3_end - t3_start;

  t4_start = ros::Time::now();

  pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);

  if (_method_type == MethodType::PCL_GENERIC)
  {
    ndt.align(*output_cloud, init_guess);
    fitness_score = ndt.getFitnessScore();
    t_localizer = ndt.getFinalTransformation();
    has_converged = ndt.hasConverged();
    final_num_iteration = ndt.getFinalNumIteration();
    transformation_probability = ndt.getTransformationProbability();
  }
  else if (_method_type == MethodType::PCL_ANH)
  {
    anh_ndt.align(init_guess);
    fitness_score = anh_ndt.getFitnessScore();
    t_localizer = anh_ndt.getFinalTransformation();
    has_converged = anh_ndt.hasConverged();
    final_num_iteration = anh_ndt.getFinalNumIteration();
  }
#ifdef CUDA_FOUND
  else if (_method_type == MethodType::PCL_ANH_GPU)
  {
    anh_gpu_ndt.align(init_guess);  // ndt_gpu库的align,不传出配准后的点云
    fitness_score = anh_gpu_ndt.getFitnessScore();
    t_localizer = anh_gpu_ndt.getFinalTransformation();
    has_converged = anh_gpu_ndt.hasConverged();
    final_num_iteration = anh_gpu_ndt.getFinalNumIteration();
  }
#endif
#ifdef USE_PCL_OPENMP
  else if (_method_type == MethodType::PCL_OPENMP)
  {
    omp_ndt.align(*output_cloud, init_guess);
    fitness_score = omp_ndt.getFitnessScore();
    t_localizer = omp_ndt.getFinalTransformation();
    has_converged = omp_ndt.hasConverged();
    final_num_iteration = omp_ndt.getFinalNumIteration();
  }
#endif
  // bask_link 需要排除掉全局起始点便宜造成的影响
  // !!!!!!!!!!也即t_localizer对应的是当前局部地图;t_base_link对应的是全局的global地图!!!!!!!!!!!
  t_base_link = t_localizer * tf_ltob;  // t_localizer为每个配准库所计算出的final_transformation !!!!!

  // 配准变换 --注意:
  pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, t_localizer);

  tf::Matrix3x3 mat_l, mat_b;

  mat_l.setValue(static_cast<double>(t_localizer(0, 0)), static_cast<double>(t_localizer(0, 1)),
                 static_cast<double>(t_localizer(0, 2)), static_cast<double>(t_localizer(1, 0)),
                 static_cast<double>(t_localizer(1, 1)), static_cast<double>(t_localizer(1, 2)),
                 static_cast<double>(t_localizer(2, 0)), static_cast<double>(t_localizer(2, 1)),
                 static_cast<double>(t_localizer(2, 2)));

  mat_b.setValue(static_cast<double>(t_base_link(0, 0)), static_cast<double>(t_base_link(0, 1)),
                 static_cast<double>(t_base_link(0, 2)), static_cast<double>(t_base_link(1, 0)),
                 static_cast<double>(t_base_link(1, 1)), static_cast<double>(t_base_link(1, 2)),
                 static_cast<double>(t_base_link(2, 0)), static_cast<double>(t_base_link(2, 1)),
                 static_cast<double>(t_base_link(2, 2)));

  // Update localizer_pose.  // 更新局部下的坐标
  localizer_pose.x = t_localizer(0, 3);
  localizer_pose.y = t_localizer(1, 3);
  localizer_pose.z = t_localizer(2, 3);
  mat_l.getRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw, 1);

  // Update ndt_pose.  // 更新全局下的坐标
  ndt_pose.x = t_base_link(0, 3);
  ndt_pose.y = t_base_link(1, 3);
  ndt_pose.z = t_base_link(2, 3);
  mat_b.getRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw, 1);

  // current_pose对应的当然是全局下的坐标!
  current_pose.x = ndt_pose.x;
  current_pose.y = ndt_pose.y;
  current_pose.z = ndt_pose.z;
  current_pose.roll = ndt_pose.roll;
  current_pose.pitch = ndt_pose.pitch;
  current_pose.yaw = ndt_pose.yaw;

  // 以下:对current_pose做变换, ---------tf变换
  transform.setOrigin(tf::Vector3(current_pose.x, current_pose.y, current_pose.z));
  q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);
  transform.setRotation(q);

  br.sendTransform(tf::StampedTransform(transform, current_scan_time, "map", "base_link"));

  scan_duration = current_scan_time - previous_scan_time;
  double secs = scan_duration.toSec();

  // Calculate the offset (curren_pos - previous_pos)
  diff_x = current_pose.x - previous_pose.x;
  diff_y = current_pose.y - previous_pose.y;
  diff_z = current_pose.z - previous_pose.z;
  diff_yaw = calcDiffForRadian(current_pose.yaw, previous_pose.yaw);
  diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

  current_velocity_x = diff_x / secs;
  current_velocity_y = diff_y / secs;
  current_velocity_z = diff_z / secs;

  current_pose_imu.x = current_pose.x;
  current_pose_imu.y = current_pose.y;
  current_pose_imu.z = current_pose.z;
  current_pose_imu.roll = current_pose.roll;
  current_pose_imu.pitch = current_pose.pitch;
  current_pose_imu.yaw = current_pose.yaw;

  current_pose_odom.x = current_pose.x;
  current_pose_odom.y = current_pose.y;
  current_pose_odom.z = current_pose.z;
  current_pose_odom.roll = current_pose.roll;
  current_pose_odom.pitch = current_pose.pitch;
  current_pose_odom.yaw = current_pose.yaw;

  current_pose_imu_odom.x = current_pose.x;
  current_pose_imu_odom.y = current_pose.y;
  current_pose_imu_odom.z = current_pose.z;
  current_pose_imu_odom.roll = current_pose.roll;
  current_pose_imu_odom.pitch = current_pose.pitch;
  current_pose_imu_odom.yaw = current_pose.yaw;

  current_velocity_imu_x = current_velocity_x;
  current_velocity_imu_y = current_velocity_y;
  current_velocity_imu_z = current_velocity_z;

  // Update position and posture. current_pos -> previous_pos
  previous_pose.x = current_pose.x;
  previous_pose.y = current_pose.y;
  previous_pose.z = current_pose.z;
  previous_pose.roll = current_pose.roll;
  previous_pose.pitch = current_pose.pitch;
  previous_pose.yaw = current_pose.yaw;

  previous_scan_time.sec = current_scan_time.sec;
  previous_scan_time.nsec = current_scan_time.nsec;

  offset_imu_x = 0.0;
  offset_imu_y = 0.0;
  offset_imu_z = 0.0;
  offset_imu_roll = 0.0;
  offset_imu_pitch = 0.0;
  offset_imu_yaw = 0.0;

  offset_odom_x = 0.0;
  offset_odom_y = 0.0;
  offset_odom_z = 0.0;
  offset_odom_roll = 0.0;
  offset_odom_pitch = 0.0;
  offset_odom_yaw = 0.0;

  offset_imu_odom_x = 0.0;
  offset_imu_odom_y = 0.0;
  offset_imu_odom_z = 0.0;
  offset_imu_odom_roll = 0.0;
  offset_imu_odom_pitch = 0.0;
  offset_imu_odom_yaw = 0.0;

  // Calculate the shift between added_pos and current_pos // 以确定是否更新全局地图
  // added_pose将一直定位于localMap的原点
  double shift = sqrt(pow(current_pose.x - added_pose.x, 2.0) + pow(current_pose.y - added_pose.y, 2.0));
  if (shift >= min_add_scan_shift)
  {
    map += *transformed_scan_ptr;
    added_pose.x = current_pose.x;
    added_pose.y = current_pose.y;
    added_pose.z = current_pose.z;
    added_pose.roll = current_pose.roll;
    added_pose.pitch = current_pose.pitch;
    added_pose.yaw = current_pose.yaw;

    if (_method_type == MethodType::PCL_GENERIC)
      ndt.setInputTarget(map_ptr);
    else if (_method_type == MethodType::PCL_ANH)
    {
      if (_incremental_voxel_update == true)
        anh_ndt.updateVoxelGrid(transformed_scan_ptr);
      else
        anh_ndt.setInputTarget(map_ptr);
    }
#ifdef CUDA_FOUND
    else if (_method_type == MethodType::PCL_ANH_GPU)
      anh_gpu_ndt.setInputTarget(map_ptr);
#endif
#ifdef USE_PCL_OPENMP
    else if (_method_type == MethodType::PCL_OPENMP)
      omp_ndt.setInputTarget(map_ptr);
#endif
  }

  // 整理sensor_msg,发布更新后的(且降采样后的)全局地图,以及位置
  // start5
  sensor_msgs::PointCloud2::Ptr map_msg_ptr(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*map_ptr, *map_msg_ptr);
  ndt_map_pub.publish(*map_msg_ptr);  // 发布

  q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);
  current_pose_msg.header.frame_id = "map";
  current_pose_msg.header.stamp = current_scan_time;
  current_pose_msg.pose.position.x = current_pose.x;
  current_pose_msg.pose.position.y = current_pose.y;
  current_pose_msg.pose.position.z = current_pose.z;
  current_pose_msg.pose.orientation.x = q.x();
  current_pose_msg.pose.orientation.y = q.y();
  current_pose_msg.pose.orientation.z = q.z();
  current_pose_msg.pose.orientation.w = q.w();

  current_pose_pub.publish(current_pose_msg);
  // end5

  // Write log // 每次points_callback都会写入log文件  // 正规@!
  if (!ofs)
  {
    std::cerr << "Could not open " << filename << "." << std::endl;
    exit(1);
  }

  ofs << input->header.seq << ","
      << input->header.stamp << ","
      << input->header.frame_id << ","
      << scan_ptr->size() << ","
      << filtered_scan_ptr->size() << ","
      << std::fixed << std::setprecision(5) << current_pose.x << ","
      << std::fixed << std::setprecision(5) << current_pose.y << ","
      << std::fixed << std::setprecision(5) << current_pose.z << ","
      << current_pose.roll << ","
      << current_pose.pitch << ","
      << current_pose.yaw << ","
      << final_num_iteration << ","
      << fitness_score << ","
      << ndt_res << ","
      << step_size << ","
      << trans_eps << ","
      << max_iter << ","
      << voxel_leaf_size << ","
      << min_scan_range << ","
      << max_scan_range << ","
      << min_add_scan_shift << std::endl;

  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "Sequence number: " << input->header.seq << std::endl;
  std::cout << "Number of scan points: " << scan_ptr->size() << " points." << std::endl;
  std::cout << "Number of filtered scan points: " << filtered_scan_ptr->size() << " points." << std::endl;
  std::cout << "transformed_scan_ptr: " << transformed_scan_ptr->points.size() << " points." << std::endl;
  std::cout << "map: " << map.points.size() << " points." << std::endl;
  std::cout << "NDT has converged: " << has_converged << std::endl;
  std::cout << "Fitness score: " << fitness_score << std::endl;
  std::cout << "Number of iteration: " << final_num_iteration << std::endl;
  std::cout << "(x,y,z,roll,pitch,yaw):" << std::endl;
  std::cout << "(" << current_pose.x << ", " << current_pose.y << ", " << current_pose.z << ", " << current_pose.roll
            << ", " << current_pose.pitch << ", " << current_pose.yaw << ")" << std::endl;
  std::cout << "Transformation Matrix:" << std::endl;
  std::cout << t_localizer << std::endl;
  std::cout << "shift: " << shift << std::endl;
  std::cout << "-----------------------------------------------------------------" << std::endl;
}


// TODO: Main函数
int main(int argc, char** argv)
{
  previous_pose.x = 0.0;
  previous_pose.y = 0.0;
  previous_pose.z = 0.0;
  previous_pose.roll = 0.0;
  previous_pose.pitch = 0.0;
  previous_pose.yaw = 0.0;

  ndt_pose.x = 0.0;
  ndt_pose.y = 0.0;
  ndt_pose.z = 0.0;
  ndt_pose.roll = 0.0;
  ndt_pose.pitch = 0.0;
  ndt_pose.yaw = 0.0;

  current_pose.x = 0.0;
  current_pose.y = 0.0;
  current_pose.z = 0.0;
  current_pose.roll = 0.0;
  current_pose.pitch = 0.0;
  current_pose.yaw = 0.0;

  current_pose_imu.x = 0.0;
  current_pose_imu.y = 0.0;
  current_pose_imu.z = 0.0;
  current_pose_imu.roll = 0.0;
  current_pose_imu.pitch = 0.0;
  current_pose_imu.yaw = 0.0;

  guess_pose.x = 0.0;
  guess_pose.y = 0.0;
  guess_pose.z = 0.0;
  guess_pose.roll = 0.0;
  guess_pose.pitch = 0.0;
  guess_pose.yaw = 0.0;

  added_pose.x = 0.0;
  added_pose.y = 0.0;
  added_pose.z = 0.0;
  added_pose.roll = 0.0;
  added_pose.pitch = 0.0;
  added_pose.yaw = 0.0;

  diff_x = 0.0;
  diff_y = 0.0;
  diff_z = 0.0;
  diff_yaw = 0.0;

  offset_imu_x = 0.0;
  offset_imu_y = 0.0;
  offset_imu_z = 0.0;
  offset_imu_roll = 0.0;
  offset_imu_pitch = 0.0;
  offset_imu_yaw = 0.0;

  offset_odom_x = 0.0;
  offset_odom_y = 0.0;
  offset_odom_z = 0.0;
  offset_odom_roll = 0.0;
  offset_odom_pitch = 0.0;
  offset_odom_yaw = 0.0;

  offset_imu_odom_x = 0.0;
  offset_imu_odom_y = 0.0;
  offset_imu_odom_z = 0.0;
  offset_imu_odom_roll = 0.0;
  offset_imu_odom_pitch = 0.0;
  offset_imu_odom_yaw = 0.0;

  ros::init(argc, argv, "ndt_mapping");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  // Set log file name. // 日志文件记录?? ofs
  char buffer[80];
  std::time_t now = std::time(NULL);
  std::tm* pnow = std::localtime(&now);
  std::strftime(buffer, 80, "%Y%m%d_%H%M%S", pnow);
  filename = "ndt_mapping_" + std::string(buffer) + ".csv";
  ofs.open(filename.c_str(), std::ios::app);  // ofs在上面定义upup

  // write header for log file
  // start initial_ofs
  if (!ofs)
  {
    std::cerr << "Could not open " << filename << "." << std::endl;
    exit(1);
  }

  ofs << "input->header.seq" << ","
      << "input->header.stamp" << ","
      << "input->header.frame_id" << ","
      << "scan_ptr->size()" << ","
      << "filtered_scan_ptr->size()" << ","
      << "current_pose.x" << ","
      << "current_pose.y" << ","
      << "current_pose.z" << ","
      << "current_pose.roll" << ","
      << "current_pose.pitch" << ","
      << "current_pose.yaw" << ","
      << "final_num_iteration" << ","
      << "fitness_score" << ","
      << "ndt_res" << ","
      << "step_size" << ","
      << "trans_eps" << ","
      << "max_iter" << ","
      << "voxel_leaf_size" << ","
      << "min_scan_range" << ","
      << "max_scan_range" << ","
      << "min_add_scan_shift" << std::endl;
  // end initial_ofs

  // setting parameters
  int method_type_tmp = 0;
  private_nh.getParam("method_type", method_type_tmp);
  _method_type = static_cast<MethodType>(method_type_tmp);  // 通过获取参数文件,来确定_method_type -- 后面也会根据CUDA等的发现与否选择_method_type
  private_nh.getParam("imu_upside_down", _imu_upside_down);
  private_nh.getParam("imu_topic", _imu_topic);
  private_nh.getParam("incremental_voxel_update", _incremental_voxel_update);

  std::cout << "method_type: " << static_cast<int>(_method_type) << std::endl;
  std::cout << "use_odom: " << _use_odom << std::endl;
  std::cout << "use_imu: " << _use_imu << std::endl;
  std::cout << "imu_upside_down: " << _imu_upside_down << std::endl;
  std::cout << "imu_topic: " << _imu_topic << std::endl;
  std::cout << "incremental_voxel_update: " << _incremental_voxel_update << std::endl;

  if (nh.getParam("tf_x", _tf_x) == false)
  {
    std::cout << "tf_x is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_y", _tf_y) == false)
  {
    std::cout << "tf_y is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_z", _tf_z) == false)
  {
    std::cout << "tf_z is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_roll", _tf_roll) == false)
  {
    std::cout << "tf_roll is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_pitch", _tf_pitch) == false)
  {
    std::cout << "tf_pitch is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_yaw", _tf_yaw) == false)
  {
    std::cout << "tf_yaw is not set." << std::endl;
    return 1;
  }

  std::cout << "(tf_x,tf_y,tf_z,tf_roll,tf_pitch,tf_yaw): (" << _tf_x << ", " << _tf_y << ", " << _tf_z << ", "
            << _tf_roll << ", " << _tf_pitch << ", " << _tf_yaw << ")" << std::endl;

#ifndef CUDA_FOUND
  if (_method_type == MethodType::PCL_ANH_GPU)   // ?????????感觉像是写反了???待查证
  {
    std::cerr << "**************************************************************" << std::endl;
    std::cerr << "[ERROR]PCL_ANH_GPU is not built. Please use other method type." << std::endl;
    std::cerr << "**************************************************************" << std::endl;
    exit(1);
  }
#endif
#ifndef USE_PCL_OPENMP
  if (_method_type == MethodType::PCL_OPENMP)
  {
    std::cerr << "**************************************************************" << std::endl;
    std::cerr << "[ERROR]PCL_OPENMP is not built. Please use other method type." << std::endl;
    std::cerr << "**************************************************************" << std::endl;
    exit(1);
  }
#endif

  // 根据初始设定的_tf,_roll等,初始化tl_btol rot_x_btol等
  Eigen::Translation3f tl_btol(_tf_x, _tf_y, _tf_z);                 // tl: translation  // tl_btol是初始设定的
  Eigen::AngleAxisf rot_x_btol(_tf_roll, Eigen::Vector3f::UnitX());  // rot: rotation
  Eigen::AngleAxisf rot_y_btol(_tf_pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rot_z_btol(_tf_yaw, Eigen::Vector3f::UnitZ());

  // 初始化tf_btol == 根据初始化得到的tl_btol等初始化tf_btol -----以获得初始变换矩阵 ---当车辆初始的起点不在预定义的globalMap原点时,就需要tf_btol了
  tf_btol = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();  
  tf_ltob = tf_btol.inverse();  // 将tf_btol取逆 --因为我们后面配准的时候都是相对于localMap的原点的,因此tf_ltob.inv将作为补偿的矩阵

  map.header.frame_id = "map";

  ndt_map_pub = nh.advertise<sensor_msgs::PointCloud2>("/ndt_map", 1000);
  current_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 1000);

  ros::Subscriber param_sub = nh.subscribe("config/ndt_mapping", 10, param_callback);
  ros::Subscriber output_sub = nh.subscribe("config/ndt_mapping_output", 10, output_callback);
  ros::Subscriber points_sub = nh.subscribe("points_raw", 100000, points_callback);
  ros::Subscriber odom_sub = nh.subscribe("/vehicle/odom", 100000, odom_callback);
  ros::Subscriber imu_sub = nh.subscribe(_imu_topic, 100000, imu_callback);

  ros::spin();

  return 0;
}
