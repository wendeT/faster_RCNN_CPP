#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <assert.h> 

#include "RoiPoolingConv.h"

void RoiPoolingConv::build(std::vector<std::vector<int>> input_shape){
     int nb_channels = input_shape[0][3] ; 
}

std::vector <int> RoiPoolingConv:: compute_output_shape(std::vector<std::vector<int>>  input_shape){
      int pool_size;
      int num_rois;
      int nb_channels;
      std::vector <int> temp {num_rois,pool_size,pool_size, nb_channels};
      return temp;
}
 void RoiPoolingConv:: call(std::vector <cv::Mat> x, bool mask=false){
       /* assert (x.size() == 2);
        // x[0] is image with shape (rows, cols, channels)
        cv::Mat img ;
        cv::Mat rois ;
        img = x[0];
        // x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1];
        // input_shape = K.shape(img)
        std::vector <int> outputs {};
        //for (roi_idx i = 0;roi_idx<num_rois;roi_idx++)
        for(int roi_idx  = 0;roi_idx<num_rois;roi_idx++){
            int x = rois[0, roi_idx, 0];
            int y = rois[0, roi_idx, 1];
            int w = rois[0, roi_idx, 2];
            int h = rois[0, roi_idx, 3];
        } */
    /* void RoiPoolingConv:: get_config(self){
        config = {'pool_size': self.pool_size,'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    } */
              
 }