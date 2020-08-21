#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

class RoiPoolingConv {      
  public: 
        //self.dim_ordering = K.image_dim_ordering()
        int pool_size;
        int num_rois;
        RoiPoolingConv(int pool_size, int num_rois) {}
        void build(std::vector<std::vector<int>>  input_shape);
        std::vector <int> compute_output_shape(std::vector<std::vector<int>>  input_shape);
        void call(std::vector <cv::Mat> x, bool mask=false);
        void get_config();
   };

