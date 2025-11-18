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

#include "sam_client.h"
#include <cv_bridge/cv_bridge.h>

SAMClient::SAMClient() : enabled_(true), service_available_(false)
{
    sam_client_ = nh_.serviceClient<vins::SAMSegmentation>("sam_segmentation");
    
    // Wait for service to become available (with timeout)
    ros::Duration timeout(5.0);
    service_available_ = sam_client_.waitForExistence(timeout);
    
    if (service_available_)
    {
        ROS_INFO("SAM service connected successfully");
    }
    else
    {
        ROS_WARN("SAM service not available. Segmentation will be disabled.");
        enabled_ = false;
    }
}

SAMClient::~SAMClient()
{
}

bool SAMClient::isServiceAvailable()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return service_available_ && enabled_;
}

bool SAMClient::checkServiceAvailability()
{
    std::lock_guard<std::mutex> lock(mutex_);
    service_available_ = sam_client_.exists();
    return service_available_;
}

bool SAMClient::getSegmentationMask(const cv::Mat& image, cv::Mat& mask)
{
    if (!enabled_ || !checkServiceAvailability())
    {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    try
    {
        vins::SAMSegmentation srv;
        
        // Convert OpenCV image to ROS message
        cv_bridge::CvImage cv_image;
        cv_image.encoding = "bgr8";
        cv_image.image = image;
        cv_image.toImageMsg(srv.request.image);
        
        // Call service
        if (sam_client_.call(srv))
        {
            if (srv.response.success)
            {
                // Convert ROS image message to OpenCV
                cv_bridge::CvImagePtr cv_ptr;
                try
                {
                    cv_ptr = cv_bridge::toCvCopy(srv.response.mask, sensor_msgs::image_encodings::MONO8);
                    mask = cv_ptr->image.clone();
                    return true;
                }
                catch (cv_bridge::Exception& e)
                {
                    ROS_ERROR("cv_bridge exception: %s", e.what());
                    return false;
                }
            }
            else
            {
                ROS_WARN("SAM service returned failure");
                return false;
            }
        }
        else
        {
            ROS_WARN("Failed to call SAM service");
            service_available_ = false;
            return false;
        }
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("Exception in getSegmentationMask: %s", e.what());
        return false;
    }
}

bool SAMClient::getSegmentationMaskWithPoints(const cv::Mat& image, 
                                              const std::vector<cv::Point2f>& points, 
                                              cv::Mat& mask)
{
    if (!enabled_ || !checkServiceAvailability())
    {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    try
    {
        vins::SAMSegmentation srv;
        
        // Convert OpenCV image to ROS message
        cv_bridge::CvImage cv_image;
        cv_image.encoding = "bgr8";
        cv_image.image = image;
        cv_image.toImageMsg(srv.request.image);
        
        // Add point prompts
        srv.request.points.clear();
        for (const auto& pt : points)
        {
            geometry_msgs::Point ros_pt;
            ros_pt.x = pt.x;
            ros_pt.y = pt.y;
            ros_pt.z = 0.0;
            srv.request.points.push_back(ros_pt);
        }
        
        // Call service
        if (sam_client_.call(srv))
        {
            if (srv.response.success)
            {
                // Convert ROS image message to OpenCV
                cv_bridge::CvImagePtr cv_ptr;
                try
                {
                    cv_ptr = cv_bridge::toCvCopy(srv.response.mask, sensor_msgs::image_encodings::MONO8);
                    mask = cv_ptr->image.clone();
                    return true;
                }
                catch (cv_bridge::Exception& e)
                {
                    ROS_ERROR("cv_bridge exception: %s", e.what());
                    return false;
                }
            }
            else
            {
                ROS_WARN("SAM service returned failure");
                return false;
            }
        }
        else
        {
            ROS_WARN("Failed to call SAM service");
            service_available_ = false;
            return false;
        }
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("Exception in getSegmentationMaskWithPoints: %s", e.what());
        return false;
    }
}

