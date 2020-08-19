#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iterator> 



 std::vector<std::map> get_data(std::string input_path){
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
        all_imgs[filename]["bboxes"] = temp_mp;  
    }

    std::vector <std::string>all_data {};
   
    for (auto const& [key, val] : all_imgs )
    {
        all_data.push_back(val);
      
    }
    // make sure the bg class is last in the list
    if (found_bg){
        	if (class_mapping["bg"] != class_mapping.size() - 1){
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch
            }		
    }
    std::vector<std::map> temp_all {all_data, classes_count, class_mapping};
   
    return temp_all;


}
	
	


int main(int argc, char** argv){

    return 0;

}