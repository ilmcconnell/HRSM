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
void ReadForcingData(const std::string& filename_prec, const std::string& filename_pet, std::vector<double>& Prec, std::vector<double>& PET);
void ReadSoilNames(const std::string& filename_soilname, std::vector<std::string>& Soilnames);
void readSoilParamsCSV(const std::string& filename, 
    std::vector<std::string>& vG_default_params_HYDRUS_names,
    std::vector<std::vector<double>>& vG_default_params_HYDRUS_values);
void ReadMLResults(const std::string& filename_sm_sly, const std::string& filename_sm_rly,
    const std::string& filename_sd_sly, const std::string& filename_sd_rly,
    std::vector<double>& sm_sly_arr, std::vector<double>& sm_rly_arr,
    std::vector<double>& sd_sly_arr, std::vector<double>& sd_rly_arr);
vector<double> GetSlyRlySM(struct wetting_front* head);
// matrix operation function:
std::vector<std::vector<double>> matrix_transpose(const std::vector<std::vector<double>>& matrix);
std::vector<std::vector<double>> matrix_multiply(const std::vector<std::vector<double>>& A,
                                                 const std::vector<std::vector<double>>& B);
std::vector<std::vector<double>> matrix_sum(const std::vector<std::vector<double>>& A,
                                            const std::vector<std::vector<double>>& B);
std::vector<std::vector<double>> matrix_inverse2D(const std::vector<std::vector<double>>& matrix);
// site processing function
void process_site(
    const std::vector<double>& Prec, 
    const std::vector<double>& PET,
    const std::vector<double>& sm_sly_arr, const std::vector<double>& sm_rly_arr,
    const std::vector<double>& sd_sly_arr, const std::vector<double>& sd_rly_arr,
    setting_info& Info);



// __global__ void process_site_kernel(int site_order, 
//     const std::vector<double>& Prec, 
//     const std::vector<double>& PET,
//     const std::vector<double>& sm_sly_arr, const std::vector<double>& sm_rly_arr,
//     const std::vector<double>& sd_sly_arr, const std::vector<double>& sd_rly_arr,
//     const std::vector<std::string>& Soilnames, const std::vector<std::string>& time);


#define SUCCESS 0

int main(int argc, char *argv[])
{


    std::string filename_prec, filename_pet, filename_sm_sly, filename_sm_rly, filename_sd_sly, filename_sd_rly, settings_path;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--filename_prec" && i + 1 < argc) {
            filename_prec = argv[++i];
        } else if (arg == "--filename_pet" && i + 1 < argc) {
            filename_pet = argv[++i];
        } else if (arg == "--filename_sm_sly" && i + 1 < argc) {
            filename_sm_sly = argv[++i];
        } else if (arg == "--filename_sm_rly" && i + 1 < argc) {
            filename_sm_rly = argv[++i];
        } else if (arg == "--filename_sd_sly" && i + 1 < argc) {
            filename_sd_sly = argv[++i];
        } else if (arg == "--filename_sd_rly" && i + 1 < argc) {
            filename_sd_rly = argv[++i];
        } else if (arg == "--settings_path" && i + 1 < argc) {
            settings_path = argv[++i];
        }
    }
    
    if (filename_sm_sly.empty() ||filename_sm_rly.empty() ||filename_sd_sly.empty() ||filename_sd_rly.empty() || filename_prec.empty() || filename_pet.empty() || settings_path.empty()) {
        std::cerr << "Usage: " << argv[0] << "--filename_prec <path> --filename_pet <path> --filename_sm_sly <path> --filename_sm_rly <path> --filename_sd_sly <path> --filename_sd_rly <path> --settings_path <path>" << std::endl;
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

    // read ML
    std::vector<double> sm_sly(n);
    std::vector<double> sm_rly(n);
    std::vector<double> sd_sly(n);
    std::vector<double> sd_rly(n);

    std::ifstream sm_sly_infile(filename_sm_sly, std::ios::binary);
    if (sm_sly_infile)
    {
        sm_sly_infile.read(reinterpret_cast<char*>(sm_sly.data()), n * sizeof(double));
    }else{
        std::cerr << "no sm_sly file" << std::endl;
        return 1;
    }
    std::ifstream sm_rly_infile(filename_sm_rly, std::ios::binary);
    if (sm_rly_infile)
    {
        sm_rly_infile.read(reinterpret_cast<char*>(sm_rly.data()), n * sizeof(double));
    }else{
        std::cerr << "no sm_rly file" << std::endl;
        return 1;
    }
    std::ifstream sd_sly_infile(filename_sd_sly, std::ios::binary);
    if (sd_sly_infile)
    {
        sd_sly_infile.read(reinterpret_cast<char*>(sd_sly.data()), n * sizeof(double));
    }else{
        std::cerr << "no sd_sly file" << std::endl;
        return 1;
    }
    std::ifstream sd_rly_infile(filename_sd_rly, std::ios::binary);
    if (sd_rly_infile)
    {
        sd_rly_infile.read(reinterpret_cast<char*>(sd_rly.data()), n * sizeof(double));
    }else{
        std::cerr << "no sd_rly file" << std::endl;
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

    process_site(Prec, PET, sm_sly, sm_rly, sd_sly, sd_rly, Info);

  return SUCCESS;
}


// process site
void process_site(
    const std::vector<double>& sitePrec, 
    const std::vector<double>& sitePET,
    const std::vector<double>& site_sm_sly_arr, const std::vector<double>& site_sm_rly_arr,
    const std::vector<double>& site_sd_sly_arr, const std::vector<double>& site_sd_rly_arr,
    setting_info& Info){
        std::cout<<"is running site: " << Info.site_ID << "\n";
        
        // set ensemble size of 20?
        int N_ensemble = 100;
        std::vector<BmiLGAR> ensemble_models(N_ensemble);
        std::vector<std::vector<double>> ensemble_states(N_ensemble, std::vector<double>(2)); // Two depths: 0-5 cm, 0-100 cm
        
        FILE *out_fptr = NULL;
        std::ostringstream filename_stream;
        filename_stream << "EnKF_LGAR_" << Info.site_ID << ".csv";
        std::string data_variables_filename = filename_stream.str();
        out_fptr = fopen(data_variables_filename.c_str(), "w");
        fprintf(out_fptr,"time,forcast_sd_sly,forcast_sd_rly,analysis_sd_sly,analysis_sd_rly,forcast_sm_sly,forcast_sm_rly,analysis_sm_sly,analysis_sm_rly,ml_sd_sly,ml_sd_rly,ml_sm_sly,ml_sm_rly\n");
        
        // ****** initilize models
        for (int i = 0; i < N_ensemble; ++i) {
            ensemble_models[i].Initialize(Info);
            ensemble_states[i] = GetSlyRlySM(ensemble_models[i].get_model()->head);  // get surface layer (0-5 cm) and rootzone (0-100 cm) soil moisture
        }
        
        // get time steps
        double endtime = ensemble_models[0].GetEndTime();
        double timestep = ensemble_models[0].GetTimeStep();
        int nsteps = int(endtime/timestep); // total number of time steps
        
        std::string var_name_precip = "precipitation_rate";
        std::string var_name_pet    = "potential_evapotranspiration_rate";
        std::string var_name_wf     = "soil_moisture_wetting_fronts";
        std::string var_name_thickness_wf = "soil_depth_wetting_fronts";

        // Define Gaussian noise generators
        random_device rd;
        default_random_engine generator(rd());

        double sdlog_prec = 0.5;
        double sdlog_pet = 0.4;
        std::lognormal_distribution<double> prec_distribution_sly(0.0, sdlog_prec);
        std::lognormal_distribution<double> pet_distribution_rly(0.0, sdlog_pet);
        double mean_factor_precip = exp(sdlog_prec * sdlog_prec / 2.0);
        double mean_factor_pet    = exp(sdlog_pet * sdlog_pet / 2.0);

        // add model structure error
        std::normal_distribution<double> model_error_sly(0.0, 0.03);
        std::normal_distribution<double> model_error_rly(0.0, 0.02);
        std::vector<double> model_error_term_sly(N_ensemble, 0.0);
        std::vector<double> model_error_term_rly(N_ensemble, 0.0);
        // double rho = 0.95;
        
        // ****** update models
        for (int j = 0; j < nsteps; j++) {
            
            for (int i = 0; i < N_ensemble; ++i) {
                double noise_factor_precip = prec_distribution_sly(generator)/mean_factor_precip;  
                double noise_factor_pet = pet_distribution_rly(generator)/mean_factor_pet;
                double perturbed_precip = std::max(0.0, sitePrec[j] * noise_factor_precip);
                double perturbed_pet = std::max(0.0, sitePET[j] * noise_factor_pet);

                ensemble_models[i].SetValue(var_name_precip, &perturbed_precip);
                ensemble_models[i].SetValue(var_name_pet, &perturbed_pet);
                ensemble_models[i].Update(); // Update model
                ensemble_states[i] = GetSlyRlySM(ensemble_models[i].get_model()->head);  // model forecasting: ensemble_states[i][0] is surface layer soil moisture, ensemble_states[i][1] is rootzone soil moisture
                
            } 
            
            double model_noise_sly;
            double model_noise_rly;
            for (int i = 0; i < N_ensemble; ++i) {
                // model_error_term_sly[i] = rho * model_error_term_sly[i] + model_error_sly(generator);
                // model_error_term_rly[i] = rho * model_error_term_rly[i] + model_error_rly(generator);
                model_error_term_sly[i] = model_error_sly(generator);
                model_error_term_rly[i] = model_error_rly(generator);
                ensemble_states[i][0] += model_error_term_sly[i];
                ensemble_states[i][1] += model_error_term_rly[i];
            }

            // calculate the mean of ensemble states
            std::vector<double> mean_state(2, 0.0); // Mean soil moisture for 0-5 cm and 0-100 cm
            for (int i = 0; i < N_ensemble; ++i) {
                mean_state[0] += ensemble_states[i][0];
                mean_state[1] += ensemble_states[i][1];
            }
            mean_state[0] /= N_ensemble;
            mean_state[1] /= N_ensemble;
            
            if ((!std::isnan(site_sm_sly_arr[j]))&&(!std::isnan(site_sm_rly_arr[j]))&&(!std::isnan(site_sd_sly_arr[j]))&&(!std::isnan(site_sd_rly_arr[j]))) {
                // do kalman filter
                double ml_site_sm_sly = site_sm_sly_arr[j]; // ML results - soil moisture for surface layer (regarded as observation)
                double ml_site_sm_rly = site_sm_rly_arr[j]; // ML results - soil moisture for rootzone (regarded as observation)
                double ml_site_sd_sly = site_sd_sly_arr[j]; // ML results - standard deviation for surface layer
                double ml_site_sd_rly = site_sd_rly_arr[j]; // ML results - standard deviation for rootzone
                
                // calculate covariance matrix of forecast
                std::vector<std::vector<double>> P(2, std::vector<double>(2, 0.0)); // 2x2 matrix
                for (size_t i = 0; i < N_ensemble; ++i) {
                    // double epsilon = 1e-6;
                    double sd_sly = ensemble_states[i][0] - mean_state[0];
                    double sd_rly = ensemble_states[i][1] - mean_state[1];
                    P[0][0] += sd_sly * sd_sly ; // + epsilon Variance for surface
                    P[0][1] += sd_sly * sd_rly; // Covariance
                    P[1][0] += sd_rly * sd_sly; // Covariance
                    P[1][1] += sd_rly * sd_rly ; // + epsilon Variance for root zone
                }
                for (size_t row = 0; row < 2; ++row) {
                    for (size_t col = 0; col < 2; ++col) {
                        P[row][col] /= (N_ensemble - 1);
                    }
                }

                // write to outputfile
                // write file - time, surface layer soil moisture, rootzone soil moisture
                fprintf(out_fptr,"%d,",j);
                fprintf(out_fptr,"%lf,%lf,", sqrt(P[0][0]), sqrt(P[1][1]));
                
                // define observation error covariance
                std::vector<std::vector<double>> R = {
                    {ml_site_sd_sly * ml_site_sd_sly, 0.0},
                    {0.0, ml_site_sd_rly * ml_site_sd_rly}
                };
                
                // compute the kalman gain:
                std::vector<std::vector<double>> K(2, std::vector<double>(2));
                K = matrix_multiply(P, matrix_inverse2D(matrix_sum(P, R)));
                
                // update ensemble states
                std::normal_distribution<double> ml_distribution_sly(0.0, ml_site_sd_sly);
                std::normal_distribution<double> ml_distribution_rly(0.0, ml_site_sd_rly);
                double ml_noise_sly;
                double ml_noise_rly;
                double ml_site_sm_sly_perturbed;
                double ml_site_sm_rly_perturbed;
                for (size_t i = 0; i < N_ensemble; ++i) {
                    ml_noise_sly = ml_distribution_sly(generator);
                    ml_noise_rly = ml_distribution_rly(generator);
                    ml_site_sm_sly_perturbed = ml_site_sm_sly + ml_noise_sly;
                    ml_site_sm_rly_perturbed = ml_site_sm_rly + ml_noise_rly;

                    double delta_sly = K[0][0] * (ml_site_sm_sly_perturbed - ensemble_states[i][0]) + K[0][1] * (ml_site_sm_rly_perturbed - ensemble_states[i][1]);
                    double delta_rly = K[1][0] * (ml_site_sm_sly_perturbed - ensemble_states[i][0]) + K[1][1] * (ml_site_sm_rly_perturbed - ensemble_states[i][1]);
                    
                    ensemble_states[i][0] += delta_sly;
                    ensemble_states[i][1] += delta_rly;
                }

                // output results
                std::vector<double> mean_state_analysis(2, 0.0); // Mean soil moisture for 0-5 cm and 0-100 cm
                for (int i = 0; i < N_ensemble; ++i) {
                    mean_state_analysis[0] += ensemble_states[i][0];
                    mean_state_analysis[1] += ensemble_states[i][1];
                }
                mean_state_analysis[0] /= N_ensemble;
                mean_state_analysis[1] /= N_ensemble;


                // calculate covariance matrix of analysis
                std::vector<std::vector<double>> PA(2, std::vector<double>(2, 0.0)); // 2x2 matrix
                for (size_t i = 0; i < N_ensemble; ++i) {
                    double sd_sly = ensemble_states[i][0] - mean_state_analysis[0];
                    double sd_rly = ensemble_states[i][1] - mean_state_analysis[1];
                    PA[0][0] += sd_sly * sd_sly; // Variance for surface
                    PA[0][1] += sd_sly * sd_rly; // Covariance
                    PA[1][0] += sd_rly * sd_sly; // Covariance
                    PA[1][1] += sd_rly * sd_rly; // Variance for root zone
                }
                for (size_t row = 0; row < 2; ++row) {
                    for (size_t col = 0; col < 2; ++col) {
                        PA[row][col] /= (N_ensemble - 1);
                    }
                }
                // write to outputfile
                fprintf(out_fptr,"%lf,%lf,", sqrt(PA[0][0]), sqrt(PA[1][1]));

                
                // write updated states (analysis) back to model for next step : key process
                for (size_t i = 0; i < N_ensemble; ++i) {
                    // this one?
                    ensemble_models[i].Update_wetting_fronts(ensemble_states[i][0], ensemble_states[i][1]);
                }
                
                fprintf(out_fptr,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", mean_state[0], mean_state[1], mean_state_analysis[0], mean_state_analysis[1], ml_site_sd_sly, ml_site_sd_rly,ml_site_sm_sly,ml_site_sm_rly);

                if (std::isnan(mean_state[0]))
                {
                    break;
                }
                
            } else {
                // calculate covariance matrix of measurements
                std::vector<std::vector<double>> P(2, std::vector<double>(2, 0.0)); // 2x2 matrix
                for (size_t i = 0; i < N_ensemble; ++i) {
                    double sd_sly = ensemble_states[i][0] - mean_state[0];
                    double sd_rly = ensemble_states[i][1] - mean_state[1];
                    P[0][0] += sd_sly * sd_sly; // Variance for surface
                    P[0][1] += sd_sly * sd_rly; // Covariance
                    P[1][0] += sd_rly * sd_sly; // Covariance
                    P[1][1] += sd_rly * sd_rly; // Variance for root zone
                }
                for (size_t row = 0; row < 2; ++row) {
                    for (size_t col = 0; col < 2; ++col) {
                        P[row][col] /= (N_ensemble - 1);
                    }
                }

                // write to outputfile
                // write file - time, surface layer soil moisture, rootzone soil moisture
                fprintf(out_fptr,"%d,",j);
                fprintf(out_fptr,"%lf,%lf,%lf,%lf,", sqrt(P[0][0]), sqrt(P[1][1]), sqrt(P[0][0]), sqrt(P[1][1]));
                
                // write file - time, surface layer soil moisture, rootzone soil moisture
                fprintf(out_fptr,"%lf,%lf,%lf,%lf,,,,\n", mean_state[0], mean_state[1], mean_state[0], mean_state[1]);
            }
            
        }
        
        // close filestream
        if (out_fptr) {
          fclose(out_fptr);
        }
        
        // ****** Finalize models
        for (int i = 0; i < N_ensemble; ++i) {
            ensemble_models[i].Finalize();
        }
    }


std::vector<int> ConvertToIntVector(const std::vector<double>& input) {
    std::vector<int> result;
    result.reserve(input.size());  // Reserve space for efficiency

    for (double val : input) {
        result.push_back(static_cast<int>(val));  // Truncate decimals
    }

    return result;
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

std::vector<std::vector<double>> matrix_transpose(const std::vector<std::vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

std::vector<std::vector<double>> matrix_multiply(const std::vector<std::vector<double>>& A,
                                          const std::vector<std::vector<double>>& B) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t colsB = B[0].size();
    
    // Initialize result matrix with zeros
    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));

    // Perform multiplication
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

std::vector<std::vector<double>> matrix_sum(const std::vector<std::vector<double>>& A,
                                     const std::vector<std::vector<double>>& B) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}


std::vector<std::vector<double>> matrix_inverse2D(const std::vector<std::vector<double>>& matrix) {
    double a = matrix[0][0], b = matrix[0][1];
    double c = matrix[1][0], d = matrix[1][1];

    // Calculate determinant
    double det = a * d - b * c;
    if (det == 0) {
        throw std::runtime_error("Matrix is singular and cannot be inverted.");
    }

    // Compute the inverse
    double inv_det = 1.0 / det;
    std::vector<std::vector<double>> result = {
        { d * inv_det, -b * inv_det },
        { -c * inv_det, a * inv_det }
    };

    return result;
}

