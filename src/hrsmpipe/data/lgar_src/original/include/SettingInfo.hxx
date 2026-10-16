// SettingInfo.h
#ifndef SETTINGINFO_H
#define SETTINGINFO_H

#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <stddef.h>
#include <assert.h>
#include <stdbool.h>
#include <vector>
#include <time.h>
#include <sstream>

struct setting_info {
    std::string verbosity;
    int site_ID;
    int num_hours;
    std::vector<double> layer_thickness_cm;
    std::vector<double> giuh_ordinates;
    std::vector<int> layer_soil_type;
    // std::vector<double> soil_z;
    double initial_psi;
    double initial_sm_perturb;
    int max_valid_soil_types;
    std::string soil_params_file;
    double wilting_point_psi;
    double field_capacity_psi;
    double root_zone_depth_cm;
    bool use_closed_form_G=false;
    bool adaptive_timestep=false;
    bool free_drainage_enabled=true;
    double mbal_tol;
    bool TO_enabled=false;
    double timestep_h;
    double endtime_s;
    double forcing_resolution_h;
    int    sft_coupled=0;
    double ponded_depth_max_cm;
    int  calib_params_flag=0;
    std::vector<std::string> vG_default_params_HYDRUS_names;
    std::vector<std::vector<double>> vG_default_params_HYDRUS_values;
};

#endif // SETTINGINFO_H

