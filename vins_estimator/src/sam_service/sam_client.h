/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Integration with Segment Anything Model
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vins/SAMSegmentation.h>
#include <geometry_msgs/Point.h>
#include <mutex>

class SAMClient
{
public:
    SAMClient();
    ~SAMClient();
    
    /**
     * Get segmentation mask for an image
     * @param image Input image (BGR format)
     * @param mask Output binary mask (255 for segmented regions, 0 otherwise)
     * @return true if successful, false otherwise
     */
    bool getSegmentationMask(const cv::Mat& image, cv::Mat& mask);
    
    /**
     * Get segmentation mask with point prompts
     * @param image Input image (BGR format)
     * @param points Point prompts for segmentation
     * @param mask Output binary mask
     * @return true if successful, false otherwise
     */
    bool getSegmentationMaskWithPoints(const cv::Mat& image, 
                                       const std::vector<cv::Point2f>& points, 
                                       cv::Mat& mask);
    
    /**
     * Check if SAM service is available
     * @return true if service is available
     */
    bool isServiceAvailable();
    
    /**
     * Enable/disable SAM integration
     */
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

private:
    ros::NodeHandle nh_;
    ros::ServiceClient sam_client_;
    bool enabled_;
    std::mutex mutex_;
    bool service_available_;
    
    bool checkServiceAvailability();
};

