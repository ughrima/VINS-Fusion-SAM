#!/usr/bin/env python3
"""
ROS Service for Segment Anything Model (SAM)
This service receives images and returns segmentation masks
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vins.srv import SAMSegmentation, SAMSegmentationResponse
import sys
import os

# Add segment-anything to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../segment-anything'))

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch

class SAMService:
    def __init__(self):
        rospy.init_node('sam_service', anonymous=True)
        
        # Get parameters
        self.model_type = rospy.get_param('~sam_model_type', 'vit_b')
        self.checkpoint_path = rospy.get_param('~sam_checkpoint_path', '')
        self.use_automatic_mask = rospy.get_param('~use_automatic_mask', True)
        self.device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        if not self.checkpoint_path:
            rospy.logerr("SAM checkpoint path not provided! Please set ~sam_checkpoint_path parameter")
            rospy.signal_shutdown("Missing checkpoint path")
            return
        
        # Initialize SAM model
        rospy.loginfo(f"Loading SAM model: {self.model_type} from {self.checkpoint_path}")
        rospy.loginfo(f"Using device: {self.device}")
        
        try:
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            
            if self.use_automatic_mask:
                # Use automatic mask generator for full image segmentation
                self.mask_generator = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,
                )
                self.predictor = None
            else:
                # Use predictor for prompt-based segmentation
                self.predictor = SamPredictor(sam)
                self.mask_generator = None
            
            rospy.loginfo("SAM model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load SAM model: {str(e)}")
            rospy.signal_shutdown("Failed to load SAM model")
            return
        
        self.bridge = CvBridge()
        
        # Create service
        self.service = rospy.Service('sam_segmentation', SAMSegmentation, self.handle_segmentation)
        rospy.loginfo("SAM service ready")
    
    def handle_segmentation(self, req):
        """
        Handle segmentation request
        """
        try:
            # Convert ROS image to OpenCV
            try:
                gray = self.bridge.imgmsg_to_cv2(req.image, "mono8")
                cv_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                rospy.logerr(f"Image conversion failed: {e}")
                return
            
            # Convert BGR to RGB for SAM
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            if self.use_automatic_mask:
                # Generate automatic masks
                masks = self.mask_generator.generate(rgb_image)
                
                # Combine all masks into a single binary mask
                # Option 1: Union of all masks (all segmented regions)
                combined_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
                
                for mask_data in masks:
                    if isinstance(mask_data['segmentation'], np.ndarray):
                        mask = mask_data['segmentation'].astype(np.uint8) * 255
                    else:
                        # Handle RLE format
                        from pycocotools import mask as mask_utils
                        mask = mask_utils.decode(mask_data['segmentation']).astype(np.uint8) * 255
                    combined_mask = np.maximum(combined_mask, mask)
                
                # Option 2: Use only high-quality masks (uncomment to use)
                # combined_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
                # for mask_data in masks:
                #     if mask_data['stability_score'] > 0.9 and mask_data['predicted_iou'] > 0.9:
                #         if isinstance(mask_data['segmentation'], np.ndarray):
                #             mask = mask_data['segmentation'].astype(np.uint8) * 255
                #         else:
                #             from pycocotools import mask as mask_utils
                #             mask = mask_utils.decode(mask_data['segmentation']).astype(np.uint8) * 255
                #         combined_mask = np.maximum(combined_mask, mask)
                
            else:
                # Use predictor with points (if provided)
                if len(req.points) > 0:
                    self.predictor.set_image(rgb_image)
                    points = np.array([[p.x, p.y] for p in req.points])
                    labels = np.ones(len(points))  # All foreground points
                    
                    masks, scores, _ = self.predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        multimask_output=False
                    )
                    combined_mask = masks[0].astype(np.uint8) * 255
                else:
                    # No points provided, return empty mask
                    combined_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
            
            # Convert mask to ROS image message
            mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding="mono8")
            
            response = SAMSegmentationResponse()
            response.mask = mask_msg
            response.success = True
            
            return response
            
        except Exception as e:
            rospy.logerr(f"Error in segmentation: {str(e)}")
            response = SAMSegmentationResponse()
            response.success = False
            return response
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        service = SAMService()
        service.run()
    except rospy.ROSInterruptException:
        pass

