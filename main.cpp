#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iterator> 
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
    std::vector <int> temp {get_output_length( width), get_output_length( height)}
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
   Input img_input;

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


     

   

int main(int argc, char** argv){

    return 0;

}