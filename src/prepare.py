import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

dataset_path="C:/Users/super/OneDrive/Desktop/materiale università/Magistrale/2° anno/Software Engineering for AI Systems/progetto-vep-su-github/Volcanic-eruption-prediction/data/predict-volcanic-eruptions_dataset"
train_path= dataset_path + "/train.csv"       #csv con colonne "segment_id" e "time_to_eruption". 'Segment_id' denota un csv di training nella cartella 'train'
sample_submission_path= dataset_path + "/sample_submission.csv" #contiene i nomi dei csv di test. Usato per costruire il test set
train_folder_path= dataset_path + "/train"
test_folder_path= dataset_path + "/test"
train_files_path= train_folder_path + "/*"
test_files_path= test_folder_path + "/*"
processed_dataset_path = "C:/Users/super/OneDrive/Desktop/prova_processed_dataset/"

train = pd.read_csv(train_path)
sample_submission = pd.read_csv(sample_submission_path)

train_frags = glob.glob(train_files_path)   #crea un array dove in ogni posizione c'è il percorso di un .csv della cartella 'train'
len(train_frags)

test_frags = glob.glob(test_files_path)
len(test_frags)

#Let's check number of observations and number of sensors for every sample in train directory.
sensors = set()
observations = set()
nan_columns = list()
missed_groups = list()
missing_sensors_train = list()      #prima chiamato for_df

for item in train_frags:
    name = int(item.split('.')[-2].split('/')[-1])
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

#Let's do the same for test set.
sensors = set()
observations = set()
nan_columns = list()
missed_groups = list()
missing_sensors_test = list()

for item in test_frags:
    name = int(item.split('.')[-2].split('/')[-1])
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


#create the training set csv
 
def build_features(signal, ts, sensor_id):       #signal=colonna di un csv, ts= nome csv. Calcola delle statistiche per ogni colonna di ogni csv, cioè per le rilevazioni di ogni sensore
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

train_set = list()
j=0
for seg in train.segment_id:         #trasforma ogni csv(60k righe, 10 colonne) in una riga con 243 features, calcolando delle statistiche su ogni colonna del csv
    signals = pd.read_csv(train_folder_path + f'/{seg}.csv')
    train_row = []
    if j%500 == 0:
        print(j)                   #ogni 500 csv processati
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

"""crea dataset con meno features, 75 su 243"""
# drop_cols = list()
# for col in train_set.columns:     #scarta le colonne con poca correlazione con la colonna target
#     if col == 'segment_id':
#         continue
#     if abs(train_set[col].corr(train_set['time_to_eruption'])) < 0.01:   #corr() compute pairwise correlation(default Pearson) of columns, excluding NA/null values
#         drop_cols.append(col)

# not_to_drop_cols = list()

# for col1 in train_set.columns:
    # for col2 in train_set.columns:
    #     if col1 == col2:
    #         continue
    #     if col1 == 'segment_id' or col2 == 'segment_id': 
    #         continue
    #     if col1 == 'time_to_eruption' or col2 == 'time_to_eruption':
    #         continue
    #     if abs(train_set[col1].corr(train_set[col2])) > 0.98:          #scarta una tra due colonne correllate tra loro più del 98%
    #         if col2 not in drop_cols and col1 not in not_to_drop_cols:
    #             drop_cols.append(col2)
    #             not_to_drop_cols.append(col1)


train = train_set.drop(['segment_id', 'time_to_eruption'], axis=1)
y_train = train_set['time_to_eruption']

# reduced_y = y_train.copy()
# reduced_train = train.copy()
# reduced_train = reduced_train.drop(drop_cols, axis=1)

#splitta il trining set in train e validetion sets
train, val, y_train, y_val = train_test_split(train, y_train, random_state=666, test_size=0.2, shuffle=True) #random_state sets a seed to the random generator, so that train-test splits are always deterministic.It's for reproducibility. If you don't set a seed, it is different each time

train.to_csv(processed_dataset_path + 'processed_training_set.csv', index=False)
val.to_csv(processed_dataset_path + 'processed_validation_set.csv', index=False)
y_train.to_csv(processed_dataset_path + 'y_train.csv', index=False)
y_val.to_csv(processed_dataset_path + 'y_validation.csv', index=False)


#reduced_train, reduced_val, reduced_y, reduced_y_val = train_test_split(reduced_train, reduced_y, random_state=666, test_size=0.2, shuffle=True)

test_set = list()
j=0
for seg in sample_submission.segment_id:
    signals = pd.read_csv(test_folder_path + f'/{seg}.csv')
    test_row = []
    if j%500 == 0:
        print(j)
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
test.to_csv(processed_dataset_path +'processed_test_set.csv', index=False)      #non possiamo calcolare l'errore sul test set perchè mancano i valori della variabile target dato che il codice è preso da una challenge su kaggle

# reduced_test = test.copy()
# reduced_test = reduced_test.drop(drop_cols, axis=1)