import sys
import numpy as np
import pandas as pd
import glob
import os
import yaml
from sklearn.model_selection import train_test_split

def process_dataset():
            separator = os.sep
            cur_dir= os.getcwd()  #path assoluto della directory del progetto
            dataset_path= os.path.join(cur_dir, 'data', 'raw', 'predict-volcanic-eruptions_dataset')
            train_path= os.path.join(dataset_path, 'train.csv')       #csv con colonne "segment_id" e "time_to_eruption". 'Segment_id' denota un csv di training nella cartella 'train'
            sample_submission_path= os.path.join(dataset_path, 'sample_submission.csv') #contiene i nomi dei csv di test. Usato per costruire il test set
            train_folder_path= os.path.join(dataset_path, 'train')
            test_folder_path= os.path.join(dataset_path, 'test')
            train_files_path= os.path.join(train_folder_path, '*')
            test_files_path= os.path.join(test_folder_path, '*')
            processed_dataset_path = os.path.join(cur_dir, 'data', 'processed') #non mettere il separator nel join se no il path che lo precede è transformato in C:
            processed_training_set_path = processed_dataset_path + separator +'processed_training_set.csv'
            processed_val_set_path = processed_dataset_path + separator + 'processed_validation_set.csv'
            y_train_path = processed_dataset_path + separator + 'y_train.csv'
            y_val_path = processed_dataset_path + separator + 'y_validation.csv'
            proc_test_set_path = processed_dataset_path + separator + 'processed_test_set.csv'
            params_path = os.path.join(cur_dir, 'params.yaml')

            if not os.path.isdir(processed_dataset_path):
                    os.makedirs(processed_dataset_path)

            # Read data preparation parameters
            with open(params_path, "r") as params_file:
                try:
                    params = yaml.safe_load(params_file)
                    params = params["prepare"]
                except yaml.YAMLError as exc:
                    print(exc)

            train = pd.read_csv(train_path)
            sample_submission = pd.read_csv(sample_submission_path)

            train_frags = glob.glob(train_files_path)   #crea un array dove in ogni posizione c'è il percorso di un .csv della cartella 'train'
            print("num train csv: ", len(train_frags))

            #Let's check number of observations and number of sensors for every sample in train directory.
            sensors = set()
            observations = set()
            nan_columns = list()
            missed_groups = list()
            missing_sensors_train = list()      #prima chiamato for_df

            for item in train_frags:
                name = int(item.split('.')[-2].split(separator)[-1]) #da path/nome.csv prende il nome (numerico) dei file di traning
                at_least_one_missed = 0
                frag = pd.read_csv(item)
                missed_group = list()
                missed_percents = list()
                for col in frag.columns:
                    missed_percents.append(frag[col].isnull().sum() / len(frag))
                    if pd.isnull(frag[col]).all() == True:
                        at_least_one_missed = 1
                        nan_columns.append(col)
                        missed_group.append(col)
                if len(missed_group) > 0:
                    missed_groups.append(missed_group)
                sensors.add(len(frag.columns))
                observations.add(len(frag))
                missing_sensors_train.append([name, at_least_one_missed] + missed_percents)

            missing_sensors_train = pd.DataFrame(
                missing_sensors_train, 
                columns=[
                    'segment_id', 'has_missed_sensors', 'missed_percent_sensor1', 
                    'missed_percent_sensor2', 'missed_percent_sensor3', 'missed_percent_sensor4', 
                    'missed_percent_sensor5', 'missed_percent_sensor6', 'missed_percent_sensor7', 
                    'missed_percent_sensor8', 'missed_percent_sensor9', 'missed_percent_sensor10'
                ]
            )

            train = pd.merge(train, missing_sensors_train)   #colonne di 'missing_sensors_train' + 'time_to_eruption'
            train_set = list()
            j=0
            for seg in train.segment_id:         #trasforma ogni csv(60k righe, 10 colonne) in una riga con 243 features, calcolando delle statistiche su ogni colonna del csv
                signals = pd.read_csv(train_folder_path +separator+ f'{seg}.csv')
                train_row = []
                if j%500 == 0:
                    print('training files processed:' + f'{j}')                   #ogni 500 csv processati
                for i in range(0, 10):
                    sensor_id = f'sensor_{i+1}'
                    train_row.append(build_features(signals[sensor_id].fillna(0), seg, sensor_id))
                train_row = pd.concat(train_row, axis=1)
                train_set.append(train_row)
                j+=1

            train_set = pd.concat(train_set)
            train_set = train_set.reset_index() #reset index of the dataframe object to default indexing (0 to number of rows minus 1), the original index gets converted to a column
            train_set = train_set.rename(columns={'index': 'segment_id'})
            train_set = pd.merge(train_set, train, on='segment_id')


            """crea dataset con features più significative, 75 invece di 243"""
            drop_cols = list()
            for col in train_set.columns:     #scarta le colonne con poca correlazione con la colonna target
                if col == 'segment_id':
                    continue
                if abs(train_set[col].corr(train_set['time_to_eruption'])) < 0.01:   #corr() compute pairwise correlation(default Pearson) of columns, excluding NA/null values
                    drop_cols.append(col)

            not_to_drop_cols = list()

            for col1 in train_set.columns:
                for col2 in train_set.columns:
                    if col1 == col2:
                        continue
                    if col1 == 'segment_id' or col2 == 'segment_id': 
                        continue
                    if col1 == 'time_to_eruption' or col2 == 'time_to_eruption':
                        continue
                    if abs(train_set[col1].corr(train_set[col2])) > 0.98:          #scarta una tra due colonne correlate tra loro più del 98%
                        if col2 not in drop_cols and col1 not in not_to_drop_cols:
                            drop_cols.append(col2)
                            not_to_drop_cols.append(col1)

            train = train_set.drop(['segment_id', 'time_to_eruption'], axis=1)
            y_train = train_set['time_to_eruption']

            reduced_y = y_train.copy()
            reduced_train = train.copy()
            reduced_train = reduced_train.drop(drop_cols, axis=1)

            #splitta il traning set in train e validation sets
            reduced_train, reduced_val, reduced_y, reduced_y_val = train_test_split(reduced_train, reduced_y, train_size=params["train_size"], test_size=params["test_size"], random_state=params["random_state"]) #random_state sets a seed to the random generator, so that train-test splits are always deterministic.It's for reproducibility. If you don't set a seed, it is different each time

            print("Writing file {} to disk.".format(processed_training_set_path))
            reduced_train.to_csv(processed_training_set_path, index=False)

            print("Writing file {} to disk.".format(processed_val_set_path))
            reduced_val.to_csv(processed_val_set_path, index=False)

            print("Writing file {} to disk.".format(y_train_path))
            reduced_y.to_csv(y_train_path, index=False)

            print("Writing file {} to disk.".format(y_val_path))
            reduced_y_val.to_csv(y_val_path, index=False)

            if (len(sys.argv) > 1 and sys.argv[1].lower() == 'true'):        #process the test set
                test_frags = glob.glob(test_files_path)
                print("num test csv: ", len(test_frags))
                sensors = set()
                observations = set()
                nan_columns = list()
                missed_groups = list()
                missing_sensors_test = list()

                for item in test_frags:
                    name = int(item.split('.')[-2].split(separator)[-1])
                    at_least_one_missed = 0
                    frag = pd.read_csv(item)
                    missed_group = list()
                    missed_percents = list()
                    for col in frag.columns:
                        missed_percents.append(frag[col].isnull().sum() / len(frag))
                        if pd.isnull(frag[col]).all() == True:
                            at_least_one_missed = 1
                            nan_columns.append(col)
                            missed_group.append(col)
                    if len(missed_group) > 0:
                        missed_groups.append(missed_group)
                    sensors.add(len(frag.columns))
                    observations.add(len(frag))
                    missing_sensors_test.append([name, at_least_one_missed] + missed_percents)

                missing_sensors_test = pd.DataFrame(
                    missing_sensors_test, 
                    columns=[
                        'segment_id', 'has_missed_sensors', 'missed_percent_sensor1', 'missed_percent_sensor2', 'missed_percent_sensor3', 
                        'missed_percent_sensor4', 'missed_percent_sensor5', 'missed_percent_sensor6', 'missed_percent_sensor7', 
                        'missed_percent_sensor8', 'missed_percent_sensor9', 'missed_percent_sensor10'
                    ]
                )

                test_set = list()
                j=0
                for seg in sample_submission.segment_id:
                    signals = pd.read_csv(test_folder_path +separator+ f'{seg}.csv')
                    test_row = []
                    if j%500 == 0:
                        print('test files processed:' + f'{j}')
                    for i in range(0, 10):
                        sensor_id = f'sensor_{i+1}'
                        test_row.append(build_features(signals[sensor_id].fillna(0), seg, sensor_id))
                    test_row = pd.concat(test_row, axis=1)
                    test_set.append(test_row)
                    j+=1
                test_set = pd.concat(test_set)
                test_set = test_set.reset_index()
                test_set = test_set.rename(columns={'index': 'segment_id'})
                test_set = pd.merge(test_set, missing_sensors_test, on='segment_id')
                test = test_set.drop(['segment_id'], axis=1)
                reduced_test = test.copy()
                reduced_test = reduced_test.drop(drop_cols, axis=1)

                print("Writing file {} to disk.".format(proc_test_set_path))
                reduced_test.to_csv(proc_test_set_path, index=False)      #non possiamo calcolare l'errore sul test set perchè mancano i valori della variabile target dato che il codice è preso da una challenge su kaggle



#create the training set csv
def build_features(signal, ts, sensor_id):       #signal=colonna di un csv, ts=nome csv. Calcola delle statistiche per ogni colonna di ogni csv, cioè per le rilevazioni di ogni sensore
    X = pd.DataFrame()
    f = np.fft.fft(signal)  #compute the Discrete Fourier Transform of a sequence. It converts a space or time signal to signal of the frequency domain
    f_real = np.real(f)
    X.loc[ts, f'{sensor_id}_sum']       = signal.sum()
    X.loc[ts, f'{sensor_id}_mean']      = signal.mean()
    X.loc[ts, f'{sensor_id}_std']       = signal.std()
    X.loc[ts, f'{sensor_id}_var']       = signal.var() 
    X.loc[ts, f'{sensor_id}_max']       = signal.max()
    X.loc[ts, f'{sensor_id}_min']       = signal.min()
    X.loc[ts, f'{sensor_id}_skew']      = signal.skew()
    X.loc[ts, f'{sensor_id}_mad']       = signal.mad()  # compute the Mean (or median) Absolute Deviation (average distance between each data point and the mean(or median). Gives an idea about the variability in a dataset)
    X.loc[ts, f'{sensor_id}_kurtosis']  = signal.kurtosis()
    X.loc[ts, f'{sensor_id}_quantile99']= np.quantile(signal, 0.99)
    X.loc[ts, f'{sensor_id}_quantile95']= np.quantile(signal, 0.95)
    X.loc[ts, f'{sensor_id}_quantile85']= np.quantile(signal, 0.85)
    X.loc[ts, f'{sensor_id}_quantile75']= np.quantile(signal, 0.75)
    X.loc[ts, f'{sensor_id}_quantile55']= np.quantile(signal, 0.55)
    X.loc[ts, f'{sensor_id}_quantile45']= np.quantile(signal, 0.45) 
    X.loc[ts, f'{sensor_id}_quantile25']= np.quantile(signal, 0.25) 
    X.loc[ts, f'{sensor_id}_quantile15']= np.quantile(signal, 0.15) 
    X.loc[ts, f'{sensor_id}_quantile05']= np.quantile(signal, 0.05)
    X.loc[ts, f'{sensor_id}_quantile01']= np.quantile(signal, 0.01)
    X.loc[ts, f'{sensor_id}_fft_real_mean']= f_real.mean()
    X.loc[ts, f'{sensor_id}_fft_real_std'] = f_real.std()
    X.loc[ts, f'{sensor_id}_fft_real_max'] = f_real.max()
    X.loc[ts, f'{sensor_id}_fft_real_min'] = f_real.min()
    return X


if __name__ == '__main__':   #execute the code only if the file was run directly, and not imported.
    process_dataset()        # If the python interpreter is running a file as the main program, it sets the variable __name__  to “__main__”. If the file is being imported from another module, __name__ will be set to the module’s name. 
    