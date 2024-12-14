#include <cmath>
#include <vector>
#include <string>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>

#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

ros::Timer mapRepublishTimer;

// Global variables for map management
pcl::PointCloud<PointType>::Ptr mapCornerCloud(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr mapSurfCloud(new pcl::PointCloud<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfMap(new pcl::KdTreeFLANN<PointType>());
ros::Publisher pubLoadedMap;

// Global variables for feature clouds
pcl::PointCloud<PointType>::Ptr cornerCloud(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfCloud(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr fullCloud(new pcl::PointCloud<PointType>());

// Queue management
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odomBuf;
std::mutex mBuf;

// Parameters for optimization
double parameters[7] = {0, 0, 0, 1, 0, 0, 0};  // [q_x, q_y, q_z, q_w, t_x, t_y, t_z]
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);
Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);

// Publishers
ros::Publisher pubOdomAftMapped;
ros::Publisher pubOdomAftMappedHighFrec;
ros::Publisher pubLaserCloudFullRes;
ros::Publisher pubPath;
nav_msgs::Path globalPath;

// System parameters
double timeLaserCloud = 0;
double timeLaserOdometry = 0;
bool systemInited = false;
pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;


void transformAssociateToMap() {
    q_w_curr = q_wmap_wodom * q_wodom_curr;
    t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

void transformUpdate() {
    q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
    t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

void pointAssociateToMap(PointType const* const pi, PointType* const po) {
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->intensity = pi->intensity;
}

void cornerCloudHandler(const sensor_msgs::PointCloud2ConstPtr& cornerCloudMsg) {
    mBuf.lock();
    cornerBuf.push(cornerCloudMsg);
    mBuf.unlock();
}

void surfCloudHandler(const sensor_msgs::PointCloud2ConstPtr& surfCloudMsg) {
    mBuf.lock();
    surfBuf.push(surfCloudMsg);
    mBuf.unlock();
}

void fullCloudHandler(const sensor_msgs::PointCloud2ConstPtr& fullCloudMsg) {
    mBuf.lock();
    fullBuf.push(fullCloudMsg);
    mBuf.unlock();
}

void odomHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
    mBuf.lock();
    odomBuf.push(odomMsg);
    mBuf.unlock();

    // High frequency pose publish
    Eigen::Quaterniond q_wodom_curr;
    Eigen::Vector3d t_wodom_curr;
    q_wodom_curr.x() = odomMsg->pose.pose.orientation.x;
    q_wodom_curr.y() = odomMsg->pose.pose.orientation.y;
    q_wodom_curr.z() = odomMsg->pose.pose.orientation.z;
    q_wodom_curr.w() = odomMsg->pose.pose.orientation.w;
    t_wodom_curr.x() = odomMsg->pose.pose.position.x;
    t_wodom_curr.y() = odomMsg->pose.pose.position.y;
    t_wodom_curr.z() = odomMsg->pose.pose.position.z;

    Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
    Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;

    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    odomAftMapped.header.stamp = odomMsg->header.stamp;
    odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
    odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
    odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
    odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
    odomAftMapped.pose.pose.position.x = t_w_curr.x();
    odomAftMapped.pose.pose.position.y = t_w_curr.y();
    odomAftMapped.pose.pose.position.z = t_w_curr.z();
    pubOdomAftMappedHighFrec.publish(odomAftMapped);

}

bool loadMap(const std::string& mapDirectory) {
    // Load optimized poses
    std::string posesFile = mapDirectory + "/optimized_poses.txt";
    std::ifstream fin(posesFile);
    if (!fin.is_open()) {
        ROS_ERROR("Cannot open poses file: %s", posesFile.c_str());
        return false;
    }

    std::vector<Eigen::Matrix4f> poses;
    std::string line;
    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                ss >> pose(i,j);
        poses.push_back(pose);
    }
    fin.close();

    // Load and transform scans
    for (size_t i = 0; i < poses.size(); i++) {
        std::stringstream ss;
        ss << mapDirectory << "/Scans/" << std::setw(6) << std::setfill('0') << i << ".pcd";

        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
        if (pcl::io::loadPCDFile<PointType>(ss.str(), *cloud) == -1) {
            ROS_WARN("Cannot open scan file: %s", ss.str().c_str());
            continue;
        }

        // Extract features
        pcl::PointCloud<PointType>::Ptr cornerPoints(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfPoints(new pcl::PointCloud<PointType>());

        for (int j = 5; j < (int)cloud->points.size() - 5; j++) {
            float diffX = cloud->points[j - 5].x + cloud->points[j - 4].x + cloud->points[j - 3].x
                       + cloud->points[j - 2].x + cloud->points[j - 1].x - 10 * cloud->points[j].x
                       + cloud->points[j + 1].x + cloud->points[j + 2].x + cloud->points[j + 3].x
                       + cloud->points[j + 4].x + cloud->points[j + 5].x;
            float diffY = cloud->points[j - 5].y + cloud->points[j - 4].y + cloud->points[j - 3].y
                       + cloud->points[j - 2].y + cloud->points[j - 1].y - 10 * cloud->points[j].y
                       + cloud->points[j + 1].y + cloud->points[j + 2].y + cloud->points[j + 3].y
                       + cloud->points[j + 4].y + cloud->points[j + 5].y;
            float diffZ = cloud->points[j - 5].z + cloud->points[j - 4].z + cloud->points[j - 3].z
                       + cloud->points[j - 2].z + cloud->points[j - 1].z - 10 * cloud->points[j].z
                       + cloud->points[j + 1].z + cloud->points[j + 2].z + cloud->points[j + 3].z
                       + cloud->points[j + 4].z + cloud->points[j + 5].z;

            float curvature = diffX * diffX + diffY * diffY + diffZ * diffZ;

            if (curvature > 0.1) {
                cornerPoints->push_back(cloud->points[j]);
            } else if (curvature < 0.1) {
                surfPoints->push_back(cloud->points[j]);
            }
        }

        // Transform to global frame
        pcl::PointCloud<PointType>::Ptr cornerPointsTransformed(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfPointsTransformed(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*cornerPoints, *cornerPointsTransformed, poses[i]);
        pcl::transformPointCloud(*surfPoints, *surfPointsTransformed, poses[i]);

        *mapCornerCloud += *cornerPointsTransformed;
        *mapSurfCloud += *surfPointsTransformed;
    }

    // Downsample map
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setLeafSize(0.4, 0.4, 0.4);

    pcl::PointCloud<PointType>::Ptr mapCornerDS(new pcl::PointCloud<PointType>());
    downSizeFilter.setInputCloud(mapCornerCloud);
    downSizeFilter.filter(*mapCornerDS);
    *mapCornerCloud = *mapCornerDS;

    downSizeFilter.setLeafSize(0.8, 0.8, 0.8);
    pcl::PointCloud<PointType>::Ptr mapSurfDS(new pcl::PointCloud<PointType>());
    downSizeFilter.setInputCloud(mapSurfCloud);
    downSizeFilter.filter(*mapSurfDS);
    *mapSurfCloud = *mapSurfDS;

    // Build kd-trees
    kdtreeCornerMap->setInputCloud(mapCornerCloud);
    kdtreeSurfMap->setInputCloud(mapSurfCloud);

    ROS_INFO("Map loaded: %lu corner points, %lu surface points",
             mapCornerCloud->points.size(), mapSurfCloud->points.size());
    return true;
}

void process() {
    while (ros::ok()) {
        if (!cornerBuf.empty() && !surfBuf.empty() && !fullBuf.empty() && !odomBuf.empty()) {
            // Pop data
            mBuf.lock();
            while (!odomBuf.empty() && odomBuf.front()->header.stamp.toSec() < cornerBuf.front()->header.stamp.toSec())
                odomBuf.pop();
            if (odomBuf.empty()) {
                mBuf.unlock();
                continue;
            }

            timeLaserCloud = cornerBuf.front()->header.stamp.toSec();
            timeLaserOdometry = odomBuf.front()->header.stamp.toSec();

            cornerCloud->clear();
            surfCloud->clear();
            fullCloud->clear();
            pcl::fromROSMsg(*cornerBuf.front(), *cornerCloud);
            pcl::fromROSMsg(*surfBuf.front(), *surfCloud);
            pcl::fromROSMsg(*fullBuf.front(), *fullCloud);

            q_wodom_curr.x() = odomBuf.front()->pose.pose.orientation.x;
            q_wodom_curr.y() = odomBuf.front()->pose.pose.orientation.y;
            q_wodom_curr.z() = odomBuf.front()->pose.pose.orientation.z;
            q_wodom_curr.w() = odomBuf.front()->pose.pose.orientation.w;
            t_wodom_curr.x() = odomBuf.front()->pose.pose.position.x;
            t_wodom_curr.y() = odomBuf.front()->pose.pose.position.y;
            t_wodom_curr.z() = odomBuf.front()->pose.pose.position.z;

            cornerBuf.pop();
            surfBuf.pop();
            fullBuf.pop();
            odomBuf.pop();
            mBuf.unlock();

            TicToc t_whole;
            transformAssociateToMap();

            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // Optimize pose using corner and surface features
            for (int iterCount = 0; iterCount < 2; iterCount++) {
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::LocalParameterization *q_parameterization =
                    new ceres::EigenQuaternionParameterization();
                ceres::Problem::Options problem_options;

                ceres::Problem problem(problem_options);
                problem.AddParameterBlock(parameters, 4, q_parameterization);
                problem.AddParameterBlock(parameters + 4, 3);

                int corner_num = 0;
                for (int i = 0; i < cornerCloud->points.size(); i++) {
                    PointType pointOri = cornerCloud->points[i];
                    PointType pointSel;
                    pointAssociateToMap(&pointOri, &pointSel);

                    kdtreeCornerMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
                    if (pointSearchSqDis[4] < 1.0) {
                        std::vector<Eigen::Vector3d> nearCorners;
                        Eigen::Vector3d center(0, 0, 0);
                        for (int j = 0; j < 5; j++) {
                            Eigen::Vector3d tmp(mapCornerCloud->points[pointSearchInd[j]].x,
                                              mapCornerCloud->points[pointSearchInd[j]].y,
                                              mapCornerCloud->points[pointSearchInd[j]].z);
                            center = center + tmp;
                            nearCorners.push_back(tmp);
                        }
                        center = center / 5.0;

                        Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                        for (int j = 0; j < 5; j++) {
                            Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                            covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                        }

                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
                        Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                        Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);

                        if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
                            Eigen::Vector3d point_on_line = center;
                            Eigen::Vector3d point_a = 0.1 * unit_direction + point_on_line;
                            Eigen::Vector3d point_b = -0.1 * unit_direction + point_on_line;

                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                            problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                            corner_num++;
                        }
                    }
                }

                int surf_num = 0;
                for (int i = 0; i < surfCloud->points.size(); i++) {
                    PointType pointOri = surfCloud->points[i];
                    PointType pointSel;
                    pointAssociateToMap(&pointOri, &pointSel);

                    kdtreeSurfMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
                    Eigen::Matrix<double, 5, 3> matA0;
                    Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                    if (pointSearchSqDis[4] < 1.0) {
                        for (int j = 0; j < 5; j++) {
                            matA0(j, 0) = mapSurfCloud->points[pointSearchInd[j]].x;
                            matA0(j, 1) = mapSurfCloud->points[pointSearchInd[j]].y;
                            matA0(j, 2) = mapSurfCloud->points[pointSearchInd[j]].z;
                        }

                        Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                        double negative_OA_dot_norm = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++) {
                            if (fabs(norm(0) * mapSurfCloud->points[pointSearchInd[j]].x +
                                    norm(1) * mapSurfCloud->points[pointSearchInd[j]].y +
                                    norm(2) * mapSurfCloud->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2) {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid) {
                            Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                            ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
                            problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                            surf_num++;
                        }
                    }
                }

                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 4;
                options.minimizer_progress_to_stdout = false;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
            }

            transformUpdate();

            // Transform and publish full point cloud
            for (int i = 0; i < fullCloud->points.size(); i++) {
                pointAssociateToMap(&fullCloud->points[i], &fullCloud->points[i]);
            }

            sensor_msgs::PointCloud2 fullCloudMsg;
            pcl::toROSMsg(*fullCloud, fullCloudMsg);
            fullCloudMsg.header.stamp = ros::Time().fromSec(timeLaserCloud);
            fullCloudMsg.header.frame_id = "camera_init";
            pubLaserCloudFullRes.publish(fullCloudMsg);

            // Publish odometry and path
            nav_msgs::Odometry odomAftMapped;
            odomAftMapped.header.frame_id = "camera_init";
            odomAftMapped.child_frame_id = "aft_mapped";
            odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserCloud);
            odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
            odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
            odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
            odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
            odomAftMapped.pose.pose.position.x = t_w_curr.x();
            odomAftMapped.pose.pose.position.y = t_w_curr.y();
            odomAftMapped.pose.pose.position.z = t_w_curr.z();
            pubOdomAftMapped.publish(odomAftMapped);

            geometry_msgs::PoseStamped pose;
            pose.header = odomAftMapped.header;
            pose.pose = odomAftMapped.pose.pose;
            globalPath.header = odomAftMapped.header;
            globalPath.poses.push_back(pose);
            pubPath.publish(globalPath);

            // Broadcast transform
            static tf::TransformBroadcaster br;
            tf::Transform transform;
            tf::Quaternion q;
            transform.setOrigin(tf::Vector3(t_w_curr.x(), t_w_curr.y(), t_w_curr.z()));
            q.setW(q_w_curr.w());
            q.setX(q_w_curr.x());
            q.setY(q_w_curr.y());
            q.setZ(q_w_curr.z());
            transform.setRotation(q);
            br.sendTransform(tf::StampedTransform(transform,
                                                 ros::Time().fromSec(timeLaserCloud),
                                                 "camera_init",
                                                 "aft_mapped"));
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "lidar_localization");
    ros::NodeHandle nh("~");

    // Get parameters
    std::string mapDirectory;
    double mapVizFilterSize;
    std::string defaultPath = ros::package::getPath("aloam_velodyne") + "/output";
    nh.param<std::string>("map_directory", mapDirectory, defaultPath);
    nh.param<double>("map_viz_filter_size", mapVizFilterSize, 0.4);

    // Set up publishers first
    pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);
    pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);
    pubPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);
    pubLoadedMap = nh.advertise<sensor_msgs::PointCloud2>("/loaded_map", 2);

    ros::Duration(0.5).sleep();

    if (mapDirectory.empty()) {
        ROS_ERROR("map_directory parameter must be specified!");
        return -1;
    }

    // Load map
    if (!loadMap(mapDirectory)) {
        ROS_ERROR("Failed to load map from directory: %s", mapDirectory.c_str());
        return -1;
    }

    // Publish initial map after loading
    sensor_msgs::PointCloud2 mapMsg;
    pcl::PointCloud<PointType>::Ptr fullMap(new pcl::PointCloud<PointType>());
    *fullMap = *mapCornerCloud + *mapSurfCloud;
    pcl::toROSMsg(*fullMap, mapMsg);
    mapMsg.header.frame_id = "camera_init";
    mapMsg.header.stamp = ros::Time::now();
    pubLoadedMap.publish(mapMsg);

    ros::Subscriber subCorner = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, cornerCloudHandler);
    ros::Subscriber subSurf = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, surfCloudHandler);
    ros::Subscriber subOdom = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, odomHandler);
    ros::Subscriber subCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, fullCloudHandler);

    // Initialize path message
    globalPath.header.frame_id = "camera_init";

    // Set up downsampling filters
    downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);

    mapRepublishTimer = nh.createTimer(ros::Duration(2.0), [&](const ros::TimerEvent&) {
        if(mapCornerCloud->size() > 0 && mapSurfCloud->size() > 0) {
            sensor_msgs::PointCloud2 mapMsg;
            pcl::PointCloud<PointType>::Ptr fullMap(new pcl::PointCloud<PointType>());
            *fullMap = *mapCornerCloud + *mapSurfCloud;
            pcl::toROSMsg(*fullMap, mapMsg);
            mapMsg.header.frame_id = "camera_init";
            mapMsg.header.stamp = ros::Time::now();
            pubLoadedMap.publish(mapMsg);
        }
    });

    std::thread processThread(process);

    ros::spin();

    return 0;
}