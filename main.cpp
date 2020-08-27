#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iterator> 
#include <assert.h>   

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "RoiPoolingConv.h"
#include "Config.h"
#include <algorithm> 
using tensorflow::Tensor;
using namespace tensorflow;
using namespace tensorflow::ops;

 std::vector<std::string> get_data(std::string input_path){
    bool found_bg = false;
    std::map<std::string, std::map<std::string, std::string>> all_imgs {};
    std::map<std::string, int> classes_count {}; 
    std::map<std::string, int> class_mapping {};

    bool visualise = true;
    int i = 1;
    std::ifstream ifs(input_path);
    std::string temp_text( (std::istreambuf_iterator<char>(ifs) ),(std::istreambuf_iterator<char>()) );

    std::string delimiter1 = "\n";
    size_t pos = 0;
    std::string token;
    std::vector<std::string> f {};
    //Read content of the file and add to f
    while ((pos = temp_text.find(delimiter1)) != std::string::npos) {
        token = temp_text.substr(0, pos);
        f.push_back(token);
        std::cout << token << std::endl;
        temp_text.erase(0, pos + delimiter1.length());
    }
    std::cout<< "Parsing annotation files";

    for (auto &line : f)
    {
        // Print process
		std::cout<<" idx= " << std::to_string(i);
		i ++ ;
         //Read content of the line  and add to line_split
        std::vector<std::string> line_split {};
        std::string delimiter2 = ",";
         while ((pos = line.find(delimiter2)) != std::string::npos)
         {
             token = line.substr(0, pos);
             line_split.push_back(token);
             std::cout << token << std::endl;
             line.erase(0, pos + delimiter2.length());
         }

        std::string filename = line_split[0];
        std::string x1 = line_split[1];
        std::string y1 = line_split[2];
        std::string x2 = line_split[3];
        std::string y2 = line_split[4];
        std::string class_name = line_split[5];
        //(filename,x1,y1,x2,y2,class_name) = line_split
        
     
        if (classes_count.find(class_name) == classes_count.end())
            classes_count[class_name] = 1;
        else
            classes_count[class_name] += 1;

        if (class_mapping.find(class_name) == class_mapping.end()){
            if (class_name == "bg" && found_bg == false){
                std::cout<<"Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).";
                found_bg = true;
            }
            class_mapping[class_name] = class_mapping.size();
        }

       //all_imgs ==all_data
         if (all_imgs.find(filename) == all_imgs.end())
        {
            all_imgs[filename] = {};
           
            cv::Mat img;
            img = cv::imread(filename);  

            
            int rows = img.rows;
            int cols = img.cols;

           
            all_imgs[filename]["filepath"] = filename;
            all_imgs[filename]["width"] = cols;
            all_imgs[filename]["height"] = rows;
            all_imgs[filename]["bboxes"] = {};
        }
        //all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})
        std::map<std::string, std::string> temp_mp {};
        temp_mp.insert(std::pair("class", class_name)); 
        temp_mp.insert(std::pair("x1", x1)); 
        temp_mp.insert(std::pair("x2", x2));
        temp_mp.insert(std::pair("y1", y1));
        temp_mp.insert(std::pair("y2", y2));
        //all_imgs[filename]["bboxes"] = temp_mp;  
    }

    std::vector <std::string>all_data {};
   
    for (auto i : all_imgs )
    {
        all_data.push_back(i.first);
      
    }
    // make sure the bg class is last in the list
    std::vector <std::string> key_to_switch_temp {};
    std::string key_to_switch;
    if (found_bg){
        	if (class_mapping["bg"] != class_mapping.size() - 1){
                 for (auto i : class_mapping) {
                       if  (i.second==class_mapping.size()-1)
                                key_to_switch_temp.push_back(i.first);
                                  }
                key_to_switch = key_to_switch_temp[0];
				int val_to_switch = class_mapping["bg"];
				class_mapping["bg"] = class_mapping.size() - 1;
				class_mapping[key_to_switch] = val_to_switch;
            }		
    }

    //std::vector<std::map> temp_all {all_data, classes_count, class_mapping}; //Fix this 
    return all_data;
}
	
	
int get_output_length(int input_length){
     return int (input_length/16);
}
       
std::vector <int> get_img_output_length(int width, int height){
    std::vector <int> temp {get_output_length( width), get_output_length( height)};
    return temp; 
}

Output nn_base( Tensor input_tensor, bool trainable=false){

   //input_shape = (None, None, 3);
    //auto input_shape = {0,0,3};
    Tensor input_shape(DT_FLOAT, TensorShape({3}));
  
   int bn_axis = 3;
   TensorShape sp({3});
   Scope scope = tensorflow::Scope::NewRootScope();
   std::map<std::string, Output> m_vars;
   std::string idx;
   m_vars["W"+idx] = Variable(scope.WithOpName("W"), sp, DT_FLOAT);
   Tensor img_input;

    // Block 1
    // auto conv = Conv2D(scope.WithOpName("Conv"), input, m_vars["W"+idx], {1, 1, 1, 1}, "SAME");- c++
    auto x = Conv2D(scope.WithOpName(64,"block1_conv1"), img_input,m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input) -python 

    // auto conv = Conv2D(scope.WithOpName("Conv"), input, m_vars["W"+idx], {1, 1, 1, 1}, "SAME");- c++
    x = Conv2D(scope.WithOpName(64,"block1_conv2"), x, m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    
    //MaxPool(scope.WithOpName("Pool"), relu, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME"); - c++
    //f - x = MaxPool(scope.WithOpName("block1_pool"), x,{2,2}, {2,2}, "SAME");
    //x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x) -python 
    
    // Block 2
     auto x = Conv2D(scope.WithOpName(128,"block2_conv1"), x,m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)

    auto x = Conv2D(scope.WithOpName(128,"block2_conv2"), x,m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)

      //f - x = MaxPool(scope.WithOpName("block2_pool"), x,{2,2}, {2,2}, "SAME");
    //x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
    // Block 3
     x = Conv2D(scope.WithOpName(256,"block3_conv1"), x, m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)

     x = Conv2D(scope.WithOpName(256,"block3_conv2"), x, m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)

     x = Conv2D(scope.WithOpName(256,"block3_conv3"), x, m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)

     //f - x = MaxPool(scope.WithOpName("block3_pool"), x,{2,2}, {2,2}, "SAME");
    //x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    // Block 4
     x = Conv2D(scope.WithOpName(512,"block4_conv1"), x, m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)

     x = Conv2D(scope.WithOpName(512,"block4_conv2"), x, m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)

    x = Conv2D(scope.WithOpName(512,"block4_conv2"), x, m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)

     //f - x = MaxPool(scope.WithOpName("block4_pool"), x,{2,2}, {2,2}, "SAME");
    //x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    // Block 5
     x = Conv2D(scope.WithOpName(512,"block5_conv1"), x, m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)

     x = Conv2D(scope.WithOpName(512,"block5_conv2"), x, m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)

    x = Conv2D(scope.WithOpName(512,"block5_conv3"), x, m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
  
    return x;
}


std::vector <Tensor> rpn_layer(Tensor base_layers, int num_anchors){
     Scope scope = tensorflow::Scope::NewRootScope();
     std::map<std::string, Output> m_vars;
    std::string idx;
     auto x = Conv2D(scope.WithOpName(512,"rpn_conv1"), base_layers,m_vars["W"+idx], {3,3}, "SAME");
    //x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    auto x_class = Conv2D(scope.WithOpName(num_anchors,"rpn_out_class"), base_layers,m_vars["W"+idx], {1,1}, "SAME");
    //x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    
    auto x_regr = Conv2D(scope.WithOpName(num_anchors,"rpn_out_regress"), base_layers,m_vars["W"+idx], {1,1}, "SAME");
    //x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    std::vector <Tensor> temp {x_class, x_regr, base_layers};
    return temp ;
}  


std::vector <Tensor> classifier_layer(Tensor base_layers, std::vector <int> input_rois, int num_rois, int nb_classes = 4){
    std::vector <int>input_shape {num_rois,7,7,512};
    //input_shape = (num_rois,7,7,512)
     int  pooling_regions = 7;

    // out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois]);

    // Couldn't find c++ equivalent for TimeDistributed 
    //out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    // out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    // out = TimeDistributed(Dropout(0.5))(out)
    // out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    // out = TimeDistributed(Dropout(0.5))(out)
    //out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    // out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    //return [out_class, out_regr];
}  


int union_renamed(std::vector <int> au, std::vector <int> bu, int area_intersection){

    int area_a = (au[2] - au[0]) * (au[3] - au[1]);
	int area_b = (bu[2] - bu[0]) * (bu[3] - bu[1]);
	int area_union = area_a + area_b - area_intersection;
	return area_union;
}
	

int  intersection(std::vector <int> ai, std::vector <int> bi){
    int x = std::max(ai[0], bi[0]);
	int y = std::max(ai[1], bi[1]);
	int w = std::min(ai[2], bi[2]) - x;
	int h = std::min(ai[3], bi[3]) - y;
	if (w < 0 || h < 0)
		return 0;
	return w*h;
}


float iou(std::vector <float> a, std::vector <float> b){
    // a and b should be (x1,y1,x2,y2)
	if ((a[0] >= a[2]) ||(a[1] >= a[3]) || (b[0] >= b[2]) || (b[1] >= b[3]))
		return 0.0;	
    int area_i = intersection(a, b);
	int area_u = union_renamed(a, b, area_i);
	return float(area_i) / float(area_u + 1e-6);
}
	

std::vector <float> calc_rpn(Config C,std::map<std::string, std::map<std::string,int>> img_data, int width, int height, int resized_width, int resized_height){

 //img_data holds key(string) and value (image data ) pair
 //image_data is map with key (string) and value (map)

 float downscale = float(C.rpn_stride) ;
 std::vector<int> anchor_sizes = C.anchor_box_scales;   // 128, 256, 512
 std::vector<std::vector<double>> anchor_ratios = C.anchor_box_ratios;  // 1:1, 1:2*sqrt(2), 2*sqrt(2):1
 int num_anchors = anchor_sizes.size() * anchor_ratios.size(); // 3x3=9
 // calculate the output map size based on the network architecture
 int output_width = get_img_output_length(resized_width,resized_height)[0];
 int output_height =get_img_output_length(resized_width,resized_height)[1];

 int n_anchratios = anchor_ratios.size();    // 3
 
 // initialise empty output objectives
     cv::Mat y_rpn_overlap = cv::Mat::zeros(output_height, output_height, num_anchors);
     cv::Mat y_is_box_valid = cv::Mat::zeros(output_height, output_width, num_anchors);
     cv::Mat y_rpn_regr = cv::Mat::zeros(output_height, output_width, num_anchors * 4);


    int num_bboxes = img_data["bboxes"].size();


    cv::Mat num_anchors_for_bbox = cv::Mat::zeros(num_bboxes,1,1);
    //num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)

    cv::Mat best_anchor_for_bbox = cv::Mat::ones(num_bboxes,4,1);
	//best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)

	 cv::Mat best_iou_for_bbox = cv::Mat::zeros(num_bboxes,1,1.0);
    //best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)

     cv::Mat best_x_for_bbox = cv::Mat::zeros(num_bboxes,4,1);
	//best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)

    cv::Mat best_dx_for_bbox = cv::Mat::zeros(num_bboxes, 4,1.0);
	//best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    // get the GT box coordinates, and resize to account for image resizing
    cv::Mat gta = cv::Mat::zeros(num_bboxes,4,1);

    
    for (auto i : img_data["bboxes"]) {
        gta.at<float>(0, 0) =  i.second["x1"] * (resized_width / float(width));
        gta.at<float>(0, 1) =  i.second["x2"] * (resized_width / float(width));
        gta.at<float>(0, 2) =  i.second["y1"] * (resized_height / float(height));
        gta.at<float>(0, 3) =  i.second["y2"] * (resized_height / float(height));

    }
 
 //std::vector<int> x(anchor_sizes.size());
 //std::iota(std::begin(x), std::end(x), 0); //0 is the starting number
    // rpn ground truth
    for (int anchor_size_idx =0;anchor_size_idx< anchor_sizes.size();anchor_size_idx++){
         for (int anchor_ratio_idx =0;anchor_ratio_idx< n_anchratios;anchor_ratio_idx++){
            int anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0];
		    int anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1];

            for (int ix=0;ix< output_width;ix++){
                // x-coordinates of the current anchor box	
				float x1_anc = downscale * (ix + 0.5) - anchor_x / 2;
				float x2_anc = downscale * (ix + 0.5) + anchor_x / 2;	
                // ignore boxes that go across image boundaries					
				if ((x1_anc < 0) || (x2_anc > resized_width))
					continue;
                
                for (int jy=0; jy<output_height;jy++){
                    // y-coordinates of the current anchor box
					float y1_anc = downscale * (jy + 0.5) - anchor_y / 2;
					float y2_anc = downscale * (jy + 0.5) + anchor_y / 2;

                    // ignore boxes that go across image boundaries
					if ((y1_anc < 0) || (y2_anc > resized_height))
						continue;
                    
                    // bbox_type indicates whether an anchor should be a target
					// Initialize with 'negative'
					std::string bbox_type = "neg";

					// this is the best IOU for the (x,y) coord and the current anchor
					// note that this is different from the best IOU for a GT bbox
					double best_iou_for_loc = 0.0;
                    for (int bbox_num=0;bbox_num< num_bboxes;bbox_num++){
                        // get IOU of the current GT box and the current anchor box
                        std::vector <float> a {gta.at<float>(bbox_num, 0), gta.at<float>(bbox_num, 2), gta.at<float>(bbox_num, 1), gta.at<float>(bbox_num, 3)};
                        std::vector <float> b {x1_anc, y1_anc, x2_anc, y2_anc};
						float curr_iou = iou(a,b);
						// calculate the regression targets if they will be needed
                        float tx, ty, tw, th;
                        // calculate the regression targets if they will be needed
						if (curr_iou > best_iou_for_bbox.at<float>(bbox_num) || curr_iou > C.rpn_max_overlap){

                            float cx = (gta.at<float>(bbox_num, 0) + gta.at<float>(bbox_num,1)) / 2.0;
							float cy = (gta.at<float>(bbox_num, 2) + gta.at<float>(bbox_num, 3)) / 2.0;
							float cxa = (x1_anc + x2_anc)/2.0;
							float cya = (y1_anc + y2_anc)/2.0;
                            // x,y are the center point of ground-truth bbox
							// xa,ya are the center point of anchor bbox (xa=downscale * (ix + 0.5); ya=downscale * (iy+0.5))
							// w,h are the width and height of ground-truth bbox
							// wa,ha are the width and height of anchor bboxe
							// tx = (x - xa) / wa
							// ty = (y - ya) / ha
							// tw = log(w / wa)
							// th = log(h / ha)
                            float tx = (cx - cxa) / (x2_anc - x1_anc);
							float ty = (cy - cya) / (y2_anc - y1_anc);
							float tw = log((gta.at<float>(bbox_num, 1) - gta.at<float>(bbox_num, 0)) / (x2_anc - x1_anc));
							float th = log((gta.at<float>(bbox_num, 3) - gta.at<float>(bbox_num, 2)) / (y2_anc - y1_anc));
                        }
                        //img_data['bboxes'][bbox_num]['class'] != 'bg':
                        //Complete later 

                        if (true){
                            //all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if (curr_iou > best_iou_for_bbox.at<float>(bbox_num)){
                            //best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                            //A.row(1).setTo(Scalar(value));
                                best_anchor_for_bbox.at<float>(bbox_num) = (jy, ix, anchor_ratio_idx, anchor_size_idx);
                                best_iou_for_bbox.at<float>(bbox_num) = (curr_iou);
                                best_x_for_bbox.at<float>(bbox_num) = (x1_anc, x2_anc, y1_anc, y2_anc);
                                best_dx_for_bbox.at<float>(bbox_num) = (tx, ty, tw, th);	
                            }
                            //we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if (curr_iou > C.rpn_max_overlap){
                                bbox_type = "pos";
								num_anchors_for_bbox.at<float>(bbox_num) += 1;
								// we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
								if (curr_iou > best_iou_for_loc){
                                    best_iou_for_loc = curr_iou;
									std::vector <float> best_regr {tx, ty, tw, th};
                                }
									
                            }
                            // if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
							if (C.rpn_min_overlap < curr_iou < C.rpn_max_overlap){
                                // gray zone between neg and pos
								if (bbox_type != "pos")
									bbox_type = "neutral";
                            }
                        }
                    }
                    // turn on or off outputs depending on IOUs
					if(bbox_type == "neg") {
                        y_is_box_valid .at<float>(jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx) = 1;
						y_rpn_overlap.at<float>(jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx) = 0;
                    }
					else if (bbox_type == "neutral"){
                        y_is_box_valid.at<float>(jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx) = 0;
						y_rpn_overlap.at<float>(jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx) = 0;
                    }
						
					else if (bbox_type == "pos"){
                        y_is_box_valid.at<float>(jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx) = 1;
						y_rpn_overlap.at<float>(jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx) = 1;
						float start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx);
                        std::vector <float >best_regr;
						y_rpn_regr.at<float>(jy, ix, start,start+4) = best_regr;
                    }			
                }		
            }					
         }
    }
	
    // we ensure that every bbox has at least one positive RPN region
    for (int idx;idx<num_anchors_for_bbox.rows;idx++){
        if (num_anchors_for_bbox.at<float>(idx) == 0){
            // no box with an IOU greater than zero ...
            if (best_anchor_for_bbox.at<float>(idx, 0)== -1);
				continue;
            y_is_box_valid.at<float>(best_anchor_for_bbox.at<float>(idx,0), best_anchor_for_bbox.at<float>(idx,1), best_anchor_for_bbox.at<float>(idx,2) + n_anchratios *best_anchor_for_bbox.at<float>(idx,3))= 1;
            y_rpn_overlap.at<float>(best_anchor_for_bbox.at<float>(idx,0), best_anchor_for_bbox.at<float>(idx,1), best_anchor_for_bbox.at<float>(idx,2) + n_anchratios *best_anchor_for_bbox.at<float>(idx,3)) = 1;
            float start = 4 * (best_anchor_for_bbox.at<float>(idx,2) + n_anchratios * best_anchor_for_bbox.at<float>(idx,3));
            y_rpn_regr.at<float>(best_anchor_for_bbox.at<float>(idx,0), best_anchor_for_bbox.at<float>(idx,1), start,start+4) = best_dx_for_bbox.at<float>(idx);
        }
    }
    //y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    //y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)
    cv::Mat y_rpn_overlap_temp = y_rpn_overlap;
    cv::transpose(y_rpn_overlap_temp,y_rpn_overlap);

    cv::Mat y_is_box_valid_temp = y_is_box_valid;
    cv::transpose(y_is_box_valid_temp,y_is_box_valid);
	
    cv::Mat y_rpn_regr_temp = y_rpn_regr;
	cv::transpose(y_rpn_regr_temp,y_rpn_regr);
    /* if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1) */

   std::vector <float> temp_final {0.0,0.0,0.0};
	//return np.copy(y_rpn_cls), np.copy(y_rpn_regr), num_pos
    return temp_final;      
}


std::vector <int > get_new_img_size(int width, int height, int img_min_side=300){
    int resized_height,resized_width;
    if (width <= height){
        float f = float(img_min_side) / width;
		 resized_height = int(f * height);
		 resized_width = img_min_side;
    }
	
	else{
        float f = float(img_min_side) / height;
		int resized_width = int(f * width);
		int resized_height = img_min_side;
    }
    std::vector <int> temp {resized_width,resized_height};
	return temp;
}

std::vector <std::string> augment(std::map<std::string, std::map<std::string,int>> img_data, Config config, bool augment=true){
    
    //img_data is a map with key - string and value image data
    assert  (img_data.find("filepath")!=img_data.end());
    assert  (img_data.find("bboxes")!=img_data.end());
    assert  (img_data.find("width")!=img_data.end());
    assert  (img_data.find("height")!=img_data.end());
    
    std::map<std::string, std::map<std::string,int>> img_data_aug;
    img_data_aug.insert(img_data.begin(), img_data.end());

    cv::Mat img;
    std::map<std::string,string> temp_filepath;
    temp_filepath = img_data_aug.at("filepath");
    img = cv::imread(temp_filepath.at("filepath"));  
    if (augment){
        // rows, cols = img.shape[:2]
        int rows = img.rows;
        int cols = img.cols;

        if (config.use_horizontal_flips && rand()%2 == 0){
            //img = cv2.flip(img, 1)
            cv::flip(img, img, +1);
            std::map<std::string,int> temp_bboxes;
            temp_bboxes = img_data_aug.at("bboxes");
            // for (auto& x : temp_bboxes) {
            int x1 = temp_bboxes.at("x1");
            int x2 = temp_bboxes.at("x2");

           temp_bboxes["x2"] = cols - x1;
           temp_bboxes["x1"] = cols - x2;

        }

        if (config.use_vertical_flips && rand()%2 == 0){
                cv::flip(img, img, 0);
                std::map<std::string,int> temp_bboxes;
                temp_bboxes = img_data_aug.at("bboxes");
                // for (auto& x : temp_bboxes) {
                int y1 = temp_bboxes.at("y1");
                int y2 = temp_bboxes.at("y2");
                temp_bboxes["y2"] = cols - y1;
                temp_bboxes["y1"] = cols - y2;
        }
        if (config.rot_90){
            std::vector <int > rand_list {0,90,180,270};
            int angle = rand_list[rand()%4];
            if (angle == 270){
                cv::Mat img_temp = img;
                cv::transpose(img_temp,img);
				cv::flip(img_temp, img_temp, 0);
                img = img_temp;
            }
				
			else if (angle == 180){
                cv::flip(img, img, -1);
            }
			else if (angle == 90){
                cv::Mat img_temp = img;
                cv::transpose(img_temp,img);
				cv::flip(img_temp, img_temp, 1);
                img = img_temp;
            }
			else if (angle == 0){
                std::cout <<"do nothing";
            }	
            //for bbox in img_data_aug['bboxes']{
            std::map<std::string,int> temp_bboxes1;
            temp_bboxes1 = img_data_aug.at("bboxes");
            int x1 =  temp_bboxes1.at("x1");
            int x2 = temp_bboxes1.at("x2");
            int y1 = temp_bboxes1.at("y1");
            int y2 = temp_bboxes1.at("y2");
            if (angle == 270){
                temp_bboxes1["x1"] = y1;
                temp_bboxes1["x2"] = y2;
                temp_bboxes1["y1"] = cols - x2;
                temp_bboxes1["y2"] = cols - x1;
            }
                
            else if (angle == 180){
                temp_bboxes1["x2"] = cols - x1;
                temp_bboxes1["x1"] = cols - x2;
                temp_bboxes1["y2"] = rows - y1;
                temp_bboxes1["y1"] =  rows - y2;
            }
                
            else if (angle == 90){
                temp_bboxes1["x1"] = rows - y2;
                temp_bboxes1["x2"] = rows - y1;
                temp_bboxes1["y1"] = x1;
                temp_bboxes1["y2"] =  x2;
            }
                     
            else if (angle == 0){
                std::cout <<"do nothing";
            }
            		
        }

       
    }
    cv::Size s = img.size();
	img_data_aug.at("width") = s.width;
	img_data_aug.at("height") = s.height;
	//return img_data_aug, img	
    std::vector <std::string> temp {};
    //Check how tp return a container with map and cv 
    return temp;
}

void get_anchor_gt(std::vector <std::string>all_img_data, Config C, std::string mode="train"){
    //std::map<std::string, std::map<std::string,int>> img_data
    
    for (auto &img_data:all_img_data){
        // read in image, and optionally add augmentation
        cv::Mat img_data_aug;
        cv::Mat x_img;
        std::map<std::string, std::map<std::string,int>> img_data1; //Change this later to img_data
        if (mode == "train"){
            std::vector <std::string> temp = augment(img_data1, C, true);
            std::vector<cv::Mat> temp1;
            img_data_aug = temp1[0]; //Change this to temp later 
            x_img = temp1[1];   //Change this to temp later 
        }
        else {
            std::vector <std::string> temp = augment(img_data1, C, false);
            std::vector<cv::Mat> temp1;
            cv::Mat img_data_aug = temp1[0]; //Change this to temp later 
            cv::Mat x_img = temp1[1];   //Change this to temp later 
            //img_data_aug, x_img = augment(img_data, C, augment=False)
        }
        cv::Size s = img_data_aug.size();
        int width = s.width;
        int height =  s.height;
    
        //(rows, cols, _) = x_img.shape
		int rows = 	x_img.rows;
        int cols = x_img.cols;
        assert (cols = width);
        assert (rows = height);

        // get image dimensions for resizing
        int resized_width = get_new_img_size(width, height, C.im_size)[0];
        int resized_height = get_new_img_size(width, height, C.im_size)[1];

        // resize the image so that smalles side is length = 300px
       
        cv::Mat debug_img;
        cv::resize(x_img, debug_img, cv::Size(resized_width,resized_height),0,0);

        //y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
        std::map<std::string,int>> img_data_aug11;
        std::vector <float> temp_calc_rpn = calc_rpn(C, img_data_aug11, width, height, resized_width, resized_height);
        float y_rpn_cls = temp_calc_rpn[0];
        float y_rpn_regr = temp_calc_rpn[1];
        float num_pos = temp_calc_rpn [2];

        // Zero-center by mean pixel, and preprocess image
        //x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB - Revise this
        //x_img = x_img.astype(np.float32)

        x_img.convertTo(x_img, CV_32F);

        x_img(0) -= C.img_channel_mean[0]; //Revise this section
        x_img(1) -= C.img_channel_mean[1];
        x_img(2) -= C.img_channel_mean[2];
        x_img /= C.img_scaling_factor;

        cv::Mat img_temp;
        //x_img = np.transpose(x_img, (2, 0, 1))
         cv::transpose(img_temp,x_img);

        //x_img = np.expand_dims(x_img, axis=0) //Revise this section
        //y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling
        //x_img = np.transpose(x_img, (0, 2, 3, 1))
        //y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
        //y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))
        //yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug, debug_i
          
    }
}

float rpn_loss_regr(int num_anchors,std::vector <float>y_true, std::vector <float>y_pred){
    //def rpn_loss_regr_fixed_num(y_true, y_pred):
    // Merged with rpn_loss_regr 
    // x is the difference between true value and predicted vaue
    //Get back to this after choosing data type for y_true

}


float rpn_loss_cls(int num_anchors){
     //Get back to this after choosing data type for y_true

}

float class_loss_regr (int num_classes){
//Get back to this after choosing data type for y_true


}

float class_loss_cls(std::vector <float>y_true, std::vector <float>y_pred){
//Get back to this after choosing data type for y_true

}



int main(int argc, char** argv){
    float lambda_rpn_regr = 1.0;
    float lambda_rpn_class = 1.0;

    float lambda_cls_regr = 1.0;
    float lambda_cls_class = 1.0;

    float epsilon = 1e-4;
    return 0;

}