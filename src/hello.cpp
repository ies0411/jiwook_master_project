#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/String.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;
    ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1);
    ros::Publisher img_pub = n.advertise<sensor_msgs::Image>("img", 10);
    cv::Mat image = cv::imread("/root/img.jpg");

    ros::Rate loop_rate(10);

    sensor_msgs::Image img_msgs;
    cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg(img_msgs);

    // int count = 0;
    while (ros::ok()) {
        std_msgs::String msg;
        // std::stringstream ss;
        // ss << "hello world " << count;
        // msg.data = ss.str();
        msg.data = "jiwook";
        // ROS_INFO("%s", msg.data.c_str());
        chatter_pub.publish(msg);
        img_pub.publish(img_msgs);
        // ros::spinOnce();
        loop_rate.sleep();
        // ++count;
    }

    return 0;
}
