#include <ros/ros.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <yaml-cpp/yaml.h>

class KittiTransformBroadcaster {
public:
    KittiTransformBroadcaster() {
        loadTransforms();
        publishTransforms();
    }

private:
    tf2_ros::StaticTransformBroadcaster static_broadcaster_;
    std::vector<geometry_msgs::TransformStamped> transforms_;

    void loadTransforms() {
        std::string config_file;
        ros::param::param<std::string>("~transform_config", config_file,
            "config/kitti_transform.yaml");

        try {
            YAML::Node config = YAML::LoadFile(config_file);
            auto transform_config = config["transforms"];

            loadTransform(transform_config["base_to_imu"]);

            loadTransform(transform_config["imu_to_velo"]);

            auto cameras = transform_config["cameras"];
            for (const auto& camera : cameras) {
                loadTransform(camera.second);
            }

        } catch (const YAML::Exception& e) {
            ROS_ERROR_STREAM("Error loading transform config: " << e.what());
        }
    }

    void loadTransform(const YAML::Node& transform_node) {
        geometry_msgs::TransformStamped transform;

        transform.header.stamp = ros::Time::now();
        transform.header.frame_id = transform_node["frame_id"].as<std::string>();
        transform.child_frame_id = transform_node["child_frame_id"].as<std::string>();

        const auto& translation = transform_node["translation"];
        transform.transform.translation.x = translation["x"].as<double>();
        transform.transform.translation.y = translation["y"].as<double>();
        transform.transform.translation.z = translation["z"].as<double>();

        const auto& rotation = transform_node["rotation"];
        transform.transform.rotation.x = rotation["x"].as<double>();
        transform.transform.rotation.y = rotation["y"].as<double>();
        transform.transform.rotation.z = rotation["z"].as<double>();
        transform.transform.rotation.w = rotation["w"].as<double>();

        transforms_.push_back(transform);
    }

    void publishTransforms() {
        if (!transforms_.empty()) {
            static_broadcaster_.sendTransform(transforms_);
            ROS_INFO("Published static transforms from config file");
        } else {
            ROS_WARN("No transforms loaded from config file");
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "kittiTransformBroadcaster");
    ros::NodeHandle nh("~");

    KittiTransformBroadcaster broadcaster;
    ros::spin();

    return 0;
}