/*
  author : Ahmad Jan
  year   : 2022
  email  : ahmad.jan@noaa.gov
  Description: the code runs Lumped Aric/semi-arid model (LASAM) through a BMI
  Input: Precipitation, Potential ET, soil types, initial conditions
  Output: see the list below (also the code writes the variables and state of the wetting fronts
          to two separate files for analysis and visualization
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "fstream"
#include <iomanip>
#include <limits>
#include <random>
#include <thread>
#include <filesystem>
#include <mutex>
#include <unistd.h>
#include <limits.h>
#include <unordered_map>
#include <atomic>
#include <unordered_set>
#include <algorithm>

#include "bmi/bmi.hxx"
#include "include/all.hxx"
#include "include/bmi_lgar.hxx"
#include "include/SettingInfo.hxx"



// module finds forcing file name in the config file
// std::string GetForcingFile(std::string config_file);

// module read forcings (precipitation and PET)

void ReadSettingsFromFile(const std::string& settings_path, setting_info& Info);
vector<double> GetSlyRlySM(struct wetting_front* head);

// site processing function
void process_site(
    const std::vector<double>& Prec, 
    const std::vector<double>& PET,
    setting_info& Info);

void print_first_24(const std::vector<double>& vec, const std::string& name);


#define SUCCESS 0

int main(int argc, char *argv[])
{

    std::string filename_prec, filename_pet, settings_path;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--filename_prec" && i + 1 < argc) {
            filename_prec = argv[++i];
        } else if (arg == "--filename_pet" && i + 1 < argc) {
            filename_pet = argv[++i];
        } else if (arg == "--settings_path" && i + 1 < argc) {
            settings_path = argv[++i];
        }
    }
    
    if (filename_prec.empty() || filename_pet.empty() || settings_path.empty()) {
        std::cerr << "Usage: " << argv[0] << "--filename_prec <path> --filename_pet <path> --settings_path <path>" << std::endl;
        return 1;
    }

    setting_info Info;
    ReadSettingsFromFile(settings_path, Info);

    // read driving force data
    size_t n = Info.num_hours; 
    std::vector<double> Prec(n);
    std::vector<double> PET(n);
    std::ifstream Prec_infile(filename_prec, std::ios::binary);
    if (Prec_infile) {
        Prec_infile.read(reinterpret_cast<char*>(Prec.data()), n * sizeof(double));
    }else{
        std::cerr << "no precipitation file" << std::endl;
        return 1;
    }

    std::ifstream PET_infile(filename_pet, std::ios::binary);
    if (PET_infile)
    {
        PET_infile.read(reinterpret_cast<char*>(PET.data()), n * sizeof(double));
    }else{
        std::cerr << "no PET file" << std::endl;
        return 1;
    }

    std::vector<std::string> vG_default_params_HYDRUS_names = {
        "sand", "loamy_sand", "sandy_loam", "loam", "silt",
        "silt_loam", "sandy_clay_loam", "clay_loam", "silty_clay_loam",
        "sandy_clay", "silty_clay", "clay"
    };
    std::vector<std::vector<double>> vG_default_params_HYDRUS_values = {
        {0.045, 0.43, 0.145, 2.68, 29.7},
        {0.057, 0.41, 0.124, 2.28, 14.5917},
        {0.065, 0.41, 0.075, 1.89, 4.42083},
        {0.078, 0.43, 0.036, 1.56, 1.04},
        {0.034, 0.46, 0.016, 1.37, 0.25},
        {0.067, 0.45, 0.02, 1.41, 0.45},
        {0.1,   0.39, 0.059, 1.48, 1.31},
        {0.095, 0.41, 0.019, 1.31, 0.26},
        {0.089, 0.43, 0.01,  1.23, 0.07},
        {0.1,   0.38, 0.027, 1.23, 0.12},
        {0.07,  0.36, 0.005, 1.09, 0.02},
        {0.068, 0.38, 0.008, 1.09, 0.2}
    };
    Info.vG_default_params_HYDRUS_names = vG_default_params_HYDRUS_names;
    Info.vG_default_params_HYDRUS_values = vG_default_params_HYDRUS_values;

    process_site(Prec, PET, Info);

  return SUCCESS;
}


// process site
void process_site(
    const std::vector<double>& sitePrec, 
    const std::vector<double>& sitePET,
    setting_info& Info){
        std::cout<<"Site ID: " << Info.site_ID << "\n";
        
        // for each site, run create a new object and run the model.
        FILE *out_fptr = NULL;
        std::ostringstream filename_stream;
        filename_stream << "Original_LGAR_" << Info.site_ID << ".csv";
        std::string data_variables_filename = filename_stream.str();
        out_fptr = fopen(data_variables_filename.c_str(), "w");
        
        BmiLGAR LGAR_model;
        LGAR_model.Initialize(Info);
        std::vector<double> LGAR_state;
        LGAR_state = GetSlyRlySM(LGAR_model.get_model()->head);

        // get time steps
        double endtime = LGAR_model.GetEndTime();
        double timestep = LGAR_model.GetTimeStep();
        int nsteps = int(endtime/timestep); // total number of time steps
        
        std::string var_name_precip = "precipitation_rate";
        std::string var_name_pet    = "potential_evapotranspiration_rate";
        std::string var_name_wf     = "soil_moisture_wetting_fronts";
        std::string var_name_thickness_wf = "soil_depth_wetting_fronts";

        
        // ****** update models
        for (int j = 0; j < nsteps; j++) {
            // if (verbosity.compare("none") != 0) {
            //     std::cout<<"===============================================================\n";
            //     std::cout<<"Real time order | "<<j<<"\n";
            //     std::cout<<"Rainfall [mm/h], PET [mm/h] = "<<sitePrec[j]<<" , "<<sitePET[j]<<"\n";
            // }

            LGAR_model.SetValue(var_name_precip, const_cast<double*>(&sitePrec[j]));
            LGAR_model.SetValue(var_name_pet, const_cast<double*>(&sitePET[j]));
            LGAR_model.Update(); // Update model
            LGAR_state = GetSlyRlySM(LGAR_model.get_model()->head);

            // write file - time, surface layer soil moisture, rootzone soil moisture
            fprintf(out_fptr,"%d,",j);
            fprintf(out_fptr,"%lf,%lf\n", LGAR_state[0], LGAR_state[1]);
        }
        
        // close filestream
        if (out_fptr) {
          fclose(out_fptr);
        }

        LGAR_model.Finalize();
    }

void print_first_24(const std::vector<double>& vec, const std::string& name) {
    std::cout << name << " first 24 elements:" << std::endl;
    size_t limit = vec.size() < 24 ? vec.size() : 24;
    for (size_t i = 0; i < limit; ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}



// Function to parse list of numbers from a string
std::vector<double> parseList(std::string input) {
    std::vector<double> values;
     // Extract content between brackets
     std::string content = input.substr(1, input.size() - 2);
    
     std::stringstream ss(content);
     std::string item;
     
     while (std::getline(ss, item, ',')) {
         // Trim whitespace from the item
         item.erase(std::remove_if(item.begin(), item.end(), 
                    [](unsigned char c) { return std::isspace(c); }), 
                    item.end());
         
         if (!item.empty()) {
             try {
                 values.push_back(std::stod(item));
             } catch (const std::exception& e) {
                 throw std::invalid_argument("Invalid number format: " + item);
             }
         }
     }
    return values;
}

std::vector<int> ConvertToIntVector(const std::vector<double>& input) {
    std::vector<int> result;
    result.reserve(input.size());  // Reserve space for efficiency

    for (double val : input) {
        result.push_back(static_cast<int>(val));  // Truncate decimals
    }

    return result;
}

// Function to convert string to bool
bool toBool(std::string value) {
    return (value == "true");
}


void ReadSettingsFromFile(const std::string& settings_path, setting_info& Info) {
    std::ifstream file(settings_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << settings_path << std::endl;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string key, value;
        if (!std::getline(iss, key, ':') || !std::getline(iss, value)) continue;

        // Trim spaces
        key.erase(key.find_last_not_of(" \t\r\n") + 1);
        value.erase(0, value.find_first_not_of(" \t\r\n"));

        // Assign values to the structure
        if (key == "verbosity") Info.verbosity = value;
        else if (key == "site_ID") Info.site_ID = std::stoi(value);
        else if (key == "num_hours") Info.num_hours = std::stoi(value);
        else if (key == "layer_thickness_cm") Info.layer_thickness_cm = parseList(value);
        else if (key == "giuh_ordinates") Info.giuh_ordinates = parseList(value);
        else if (key == "layer_soil_type") Info.layer_soil_type = ConvertToIntVector(parseList(value));
        else if (key == "timestep_h") Info.timestep_h = std::stod(value);
        else if (key == "endtime_s") Info.endtime_s = std::stod(value);
        else if (key == "field_capacity_psi") Info.field_capacity_psi = std::stod(value);
        else if (key == "wilting_point_psi") Info.wilting_point_psi = std::stod(value);
        else if (key == "initial_psi") Info.initial_psi = std::stod(value);
        else if (key == "initial_sm_perturb") Info.initial_sm_perturb = std::stod(value);
        else if (key == "soil_params_file") Info.soil_params_file = value;
        else if (key == "TO_enabled") Info.TO_enabled = toBool(value);
        else if (key == "adaptive_timestep") Info.adaptive_timestep = toBool(value);
        else if (key == "free_drainage_enabled") Info.free_drainage_enabled = toBool(value);
        else if (key == "use_closed_form_G") Info.use_closed_form_G = toBool(value);
        else if (key == "root_zone_depth_cm") Info.root_zone_depth_cm = std::stod(value);
        else if (key == "mbal_tol") Info.mbal_tol = std::stod(value);
        else if (key == "ponded_depth_max_cm") Info.ponded_depth_max_cm = std::stod(value);
        else if (key == "forcing_resolution_h") Info.forcing_resolution_h = std::stod(value);
        else if (key == "max_valid_soil_types") Info.max_valid_soil_types = std::stoi(value);
    }

    file.close();
}

extern void write_state(FILE *out, struct wetting_front* head){

  struct wetting_front *current = head;

  fprintf(out, "[");
  while(current != NULL)
  {
    if (current == head)
      fprintf(out,"(%lf,%lf,%d,%d,%lf)",current->depth_cm*10., current->theta, current->layer_num, current->front_num, current->psi_cm*10.);
    else
      fprintf(out,"|(%lf,%lf,%d,%d,%lf)",current->depth_cm*10., current->theta, current->layer_num, current->front_num, current->psi_cm*10.);
  current = current->next;
  }
  fprintf(out, "]\n");

}

extern void write_state_only_sm(FILE *out, struct wetting_front* head){   // soil moisture for each layer
    
    struct wetting_front* current;
    // struct wetting_front* next;
    current=head;
    
    std::vector<double> total_sm_mass(5, 0.0);
    double base_depth = 0;
    int layer;
    int front;

    do
    {
        layer = current->layer_num;   // layer number
        front = current->front_num;   // front number
        
        total_sm_mass[layer-1] += (current->depth_cm - base_depth) * current->theta;
        
        base_depth = current->depth_cm;
        current=current->next;
        
    } while(current != NULL );
    
    std::vector<double> soil_layer_depth = {5, 10, 15, 30, 40};
    
    for (int i=0;i<total_sm_mass.size();i++){
        fprintf(out,"%lf,",total_sm_mass[i]/soil_layer_depth[i]);
    }
    fprintf(out, "\n");
    
}

vector<double> GetSlyRlySM(struct wetting_front* head){
    struct wetting_front* current;
    // struct wetting_front* next;
    current=head;
    
    std::vector<double> total_sm_mass(5, 0.0);
    double base_depth = 0;
    int layer;
    int front;

    do{
        layer = current->layer_num;   // layer number
        front = current->front_num;   // front number
        total_sm_mass[layer-1] += (current->depth_cm - base_depth) * current->theta;
        base_depth = current->depth_cm;
        current=current->next;
    } while(current != NULL );
    
    std::vector<double> soil_layer_depth = {5, 10, 15, 30, 40};
    
    // based on the wetting front, get the corresponding soil moisture for surface layer (0-5 cm) and rootzone (0-100 cm)
    std::vector<double> sm = {total_sm_mass[0]/soil_layer_depth[0],
        std::accumulate(total_sm_mass.begin(), total_sm_mass.end(), 0.0)/std::accumulate(soil_layer_depth.begin(), soil_layer_depth.end(), 0.0)};
    
    
    return sm;
}

