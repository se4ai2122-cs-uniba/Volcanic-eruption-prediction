"""Data validation with Great Expectations""" 
import datetime
from pathlib import Path
import great_expectations as ge
import pandas as pd
import glob
import os

#files paths
separator = os.sep
cur_dir= os.getcwd()  #path assoluto della directory del progetto
dataset_path= os.path.join(cur_dir, 'data', 'raw', 'predict-volcanic-eruptions_dataset')
processed_dataset_path = os.path.join(cur_dir, 'data', 'processed') #non mettere il separator nel join se no il path che lo precede è transformato in C:
processed_training_set_path = processed_dataset_path + separator +'processed_training_set.csv'
processed_val_set_path = processed_dataset_path + separator + 'processed_validation_set.csv'
proc_test_set_path = processed_dataset_path + separator + 'processed_test_set.csv'
sample_submission_path= os.path.join(dataset_path, 'sample_submission.csv') #contiene i nomi dei csv di test. Usato per costruire il test set
train_path= os.path.join(dataset_path, 'train.csv')       #csv con colonne "segment_id" e "time_to_eruption". 'Segment_id' denota un csv di training nella cartella 'train'
train_folder_path= os.path.join(dataset_path, 'train')
test_folder_path= os.path.join(dataset_path, 'test')
train_files_path= os.path.join(train_folder_path, '*')
test_files_path= os.path.join(test_folder_path, '*')
y_train_path = processed_dataset_path + separator + 'y_train.csv'
y_val_path = processed_dataset_path + separator + 'y_validation.csv'

#expected columns for the various csv
expected_columns_train_test = ['sensor_1','sensor_2','sensor_3','sensor_4','sensor_5','sensor_6',
                    'sensor_7','sensor_8','sensor_9','sensor_10']
expected_columns_tr_samp_sub = ['segment_id','time_to_eruption']
expected_columns_y_train_val = ['time_to_eruption']
all_columns_proc_train_val_test = ['sensor_1_sum','sensor_1_mean','sensor_1_std','sensor_1_var','sensor_1_max','sensor_1_min','sensor_1_skew','sensor_1_mad','sensor_1_kurtosis','sensor_1_quantile99','sensor_1_quantile95','sensor_1_quantile85','sensor_1_quantile75','sensor_1_quantile55','sensor_1_quantile45','sensor_1_quantile25','sensor_1_quantile15',	
'sensor_1_quantile05','sensor_1_quantile01','sensor_1_fft_real_mean','sensor_1_fft_real_std','sensor_1_fft_real_max','sensor_1_fft_real_min','sensor_2_sum','sensor_2_mean','sensor_2_std','sensor_2_var','sensor_2_max','sensor_2_min','sensor_2_skew','sensor_2_mad','sensor_2_kurtosis','sensor_2_quantile99','sensor_2_quantile95','sensor_2_quantile85',	
'sensor_2_quantile75','sensor_2_quantile55','sensor_2_quantile45','sensor_2_quantile25','sensor_2_quantile15','sensor_2_quantile05','sensor_2_quantile01','sensor_2_fft_real_mean','sensor_2_fft_real_std','sensor_2_fft_real_max','sensor_2_fft_real_min','sensor_3_sum','sensor_3_mean','sensor_3_std','sensor_3_var','sensor_3_max','sensor_3_min','sensor_3_skew',	
'sensor_3_mad','sensor_3_kurtosis','sensor_3_quantile99','sensor_3_quantile95','sensor_3_quantile85','sensor_3_quantile75','sensor_3_quantile55','sensor_3_quantile45','sensor_3_quantile25','sensor_3_quantile15','sensor_3_quantile05','sensor_3_quantile01','sensor_3_fft_real_mean','sensor_3_fft_real_std','sensor_3_fft_real_max','sensor_3_fft_real_min','sensor_4_sum',	
'sensor_4_mean','sensor_4_std','sensor_4_var','sensor_4_max','sensor_4_min','sensor_4_skew','sensor_4_mad','sensor_4_kurtosis','sensor_4_quantile99','sensor_4_quantile95','sensor_4_quantile85','sensor_4_quantile75','sensor_4_quantile55','sensor_4_quantile45','sensor_4_quantile25','sensor_4_quantile15','sensor_4_quantile05','sensor_4_quantile01','sensor_4_fft_real_mean',	
'sensor_4_fft_real_std','sensor_4_fft_real_max','sensor_4_fft_real_min','sensor_5_sum','sensor_5_mean','sensor_5_std','sensor_5_var','sensor_5_max','sensor_5_min','sensor_5_skew','sensor_5_mad','sensor_5_kurtosis','sensor_5_quantile99','sensor_5_quantile95','sensor_5_quantile85','sensor_5_quantile75','sensor_5_quantile55','sensor_5_quantile45','sensor_5_quantile25',	
'sensor_5_quantile15','sensor_5_quantile05','sensor_5_quantile01','sensor_5_fft_real_mean','sensor_5_fft_real_std','sensor_5_fft_real_max','sensor_5_fft_real_min','sensor_6_sum','sensor_6_mean','sensor_6_std','sensor_6_var','sensor_6_max','sensor_6_min','sensor_6_skew','sensor_6_mad','sensor_6_kurtosis','sensor_6_quantile99','sensor_6_quantile95','sensor_6_quantile85',	
'sensor_6_quantile75','sensor_6_quantile55','sensor_6_quantile45','sensor_6_quantile25','sensor_6_quantile15','sensor_6_quantile05','sensor_6_quantile01','sensor_6_fft_real_mean','sensor_6_fft_real_std','sensor_6_fft_real_max','sensor_6_fft_real_min','sensor_7_sum','sensor_7_mean','sensor_7_std','sensor_7_var','sensor_7_max','sensor_7_min','sensor_7_skew','sensor_7_mad',	
'sensor_7_kurtosis','sensor_7_quantile99','sensor_7_quantile95','sensor_7_quantile85','sensor_7_quantile75','sensor_7_quantile55','sensor_7_quantile45','sensor_7_quantile25','sensor_7_quantile15','sensor_7_quantile05','sensor_7_quantile01','sensor_7_fft_real_mean','sensor_7_fft_real_std','sensor_7_fft_real_max','sensor_7_fft_real_min','sensor_8_sum','sensor_8_mean','sensor_8_std',	
'sensor_8_var','sensor_8_max','sensor_8_min','sensor_8_skew','sensor_8_mad','sensor_8_kurtosis','sensor_8_quantile99','sensor_8_quantile95','sensor_8_quantile85','sensor_8_quantile75','sensor_8_quantile55','sensor_8_quantile45','sensor_8_quantile25','sensor_8_quantile15','sensor_8_quantile05','sensor_8_quantile01','sensor_8_fft_real_mean','sensor_8_fft_real_std','sensor_8_fft_real_max',	
'sensor_8_fft_real_min','sensor_9_sum','sensor_9_mean','sensor_9_std','sensor_9_var','sensor_9_max','sensor_9_min','sensor_9_skew','sensor_9_mad','sensor_9_kurtosis','sensor_9_quantile99','sensor_9_quantile95','sensor_9_quantile85','sensor_9_quantile75','sensor_9_quantile55','sensor_9_quantile45','sensor_9_quantile25','sensor_9_quantile15','sensor_9_quantile05','sensor_9_quantile01',	
'sensor_9_fft_real_mean','sensor_9_fft_real_std','sensor_9_fft_real_max','sensor_9_fft_real_min','sensor_10_sum','sensor_10_mean','sensor_10_std','sensor_10_var','sensor_10_max','sensor_10_min','sensor_10_skew','sensor_10_mad','sensor_10_kurtosis','sensor_10_quantile99','sensor_10_quantile95','sensor_10_quantile85','sensor_10_quantile75','sensor_10_quantile55','sensor_10_quantile45','sensor_10_quantile25',
'sensor_10_quantile15','sensor_10_quantile05','sensor_10_quantile01','sensor_10_fft_real_mean','sensor_10_fft_real_std','sensor_10_fft_real_max','sensor_10_fft_real_min','has_missed_sensors','missed_percent_sensor1','missed_percent_sensor2','missed_percent_sensor3','missed_percent_sensor4','missed_percent_sensor5','missed_percent_sensor6','missed_percent_sensor7','missed_percent_sensor8',
'missed_percent_sensor9','missed_percent_sensor10']

reducted_features = pd.read_csv(Path("data/processed") / "processed_validation_set.csv", nrows=1).columns.tolist()  #output of the feature selection

#data validation using great expectation, write all results on a file
def validate_data(file_path, expected_columns, columns_types, processed_dataset = False, file_write_mode = 'a'):
    results = {}
    for filename in glob.glob(file_path):               #glob return a list of paths 
        frame = pd.read_csv(filename, index_col=None, header=0)       #header=0 so the first row can be assigned as the column names
        df = ge.dataset.PandasDataset(frame)
        filename = 'filename: ' + filename.split(separator)[-1]
        results[filename]= []                                    #to each file is associated a list containing expectations results
        if processed_dataset:
           results[filename].append('selected columns are a valid subset:'+ str(set(expected_columns).issubset(set(all_columns_proc_train_val_test))))
        else:    
          results[filename].append(df.expect_table_columns_to_match_ordered_list(column_list=expected_columns))
        
        results[filename].append(df.expect_table_row_count_to_be_between(min_value=1, max_value=None))       #not empty csv
        for col in expected_columns:
            results[filename].append(df.expect_column_values_to_not_be_null(column=col))
            results[filename].append(df.expect_column_values_to_be_in_type_list(column=col, type_list= columns_types))

    with open(Path('tests/data_validation')/'validation_output.txt', file_write_mode) as f:
        f.write('\n\n'+ 'Analysis date: '+ str(datetime.datetime.now())+'\n\n'+ str(results))
       
    return print('results written in the file: tests/data_validation/validation_output.txt' )

#validate all 4k csv in train e test folder
validate_data(train_files_path, expected_columns_train_test, ['float'], file_write_mode = 'w')    #8 minutes to validate all the 4k csv
validate_data(test_files_path, expected_columns_train_test, ['float'])

#validate the csv which maps each training csv to its label
validate_data(train_path, expected_columns_tr_samp_sub, ['int'])
validate_data(sample_submission_path, expected_columns_tr_samp_sub, ['int'])

#validate the processed csv which will used to train the model
validate_data(processed_training_set_path, reducted_features, ['float','int'], processed_dataset = True)
validate_data(processed_val_set_path, reducted_features, ['float', 'int'], processed_dataset = True)
validate_data(proc_test_set_path, reducted_features, ['int'], processed_dataset = True)

#validate the processed csv which contains the labels
validate_data(y_train_path, expected_columns_y_train_val, ['int'])
validate_data(y_val_path, expected_columns_y_train_val, ['int'])




