import data_utils

projectFolder = 'D:/Data Science/NYU Data Science/DS-GA 1003 Project'
data_dir_path = projectFolder + '/Data'

### --- Create Training / Validation and Testing Samples
sample_fractions = [0.6, 0.3, 0.1] # training / validation / testing
input_size = (71,71)
output_size = 1

handler = data_utils.data_handler(data_dir_path, sample_fractions=sample_fractions, 
                              input_size=input_size, labels_type='classes', 
                              output_size=output_size, normalize_input=False, 
                              create_samples_bool=True, preprocess_bool=True, 
                              crp_factor=2, ds_factor=3)

