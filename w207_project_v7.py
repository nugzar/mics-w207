import numpy as np
import pandas as pd
import os, sys

from sklearn.experimental import enable_hist_gradient_boosting
import sklearn.ensemble as ske

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# Starting code here

if len(sys.argv) != 3:
    print("program csvfile numberofrows")
    exit()

csvfilename = sys.argv[1]
totalrows = int(sys.argv[2])

datatypes = {
    'ProductName': np.int64,
    'IsBeta': np.int64,
    'RtpStateBitfield': np.float64,
    'IsSxsPassiveMode': np.int64,
    'AVProductStatesIdentifier': np.float64,
    'AVProductsInstalled': np.float64,
    'AVProductsEnabled': np.float64,
    'HasTpm': np.int64,
    'CountryIdentifier': np.int64,
    'CityIdentifier': np.int64,
    'OrganizationIdentifier': np.int64,
    'GeoNameIdentifier': np.float64,
    'LocaleEnglishNameIdentifier': np.int64,
    'Platform': np.int64,
    'Processor': np.int64,
    'OsBuild': np.int64,
    'OsSuite': np.int64,
    'OsPlatformSubRelease': np.int64,
    'SkuEdition': np.int64,
    'IsProtected': np.float64,
    'AutoSampleOptIn': np.int64,
    'SMode': np.int64,
    'IeVerIdentifier': np.float64,
    'SmartScreen': np.int64,
    'Firewall': np.float64,
    'UacLuaenable': np.float64,
    'Census_MDC2FormFactor': np.int64,
    'Census_DeviceFamily': np.int64,
    'Census_OEMNameIdentifier': np.float64,
    'Census_OEMModelIdentifier': np.float64,
    'Census_ProcessorCoreCount': np.float64,
    'Census_ProcessorManufacturerIdentifier': np.float64,
    'Census_ProcessorModelIdentifier': np.float64,
    'Census_PrimaryDiskTotalCapacity': np.float64,
    'Census_PrimaryDiskTypeName': np.int64,
    'Census_SystemVolumeTotalCapacity': np.float64,
    'Census_HasOpticalDiskDrive': np.int64,
    'Census_TotalPhysicalRAM': np.float64,
    'Census_ChassisTypeName': np.int64,
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': np.float64,
    'Census_InternalPrimaryDisplayResolutionHorizontal': np.float64,
    'Census_InternalPrimaryDisplayResolutionVertical': np.float64,
    'Census_PowerPlatformRoleName': np.int64,
    'Census_InternalBatteryNumberOfCharges': np.float64,
    'Census_OSArchitecture': np.int64,
    'Census_OSBranch': np.int64,
    'Census_OSBuildNumber': np.int64,
    'Census_OSBuildRevision': np.int64,
    'Census_OSEdition': np.int64,
    'Census_OSSkuName': np.int64,
    'Census_OSInstallTypeName': np.int64,
    'Census_OSInstallLanguageIdentifier': np.float64,
    'Census_OSUILocaleIdentifier': np.int64,
    'Census_OSWUAutoUpdateOptionsName': np.int64,
    'Census_IsPortableOperatingSystem': np.int64,
    'Census_GenuineStateName': np.int64,
    'Census_ActivationChannel': np.int64,
    'Census_IsFlightsDisabled': np.float64,
    'Census_FlightRing': np.int64,
    'Census_ThresholdOptIn': np.float64,
    'Census_FirmwareManufacturerIdentifier': np.float64,
    'Census_FirmwareVersionIdentifier': np.float64,
    'Census_IsSecureBootEnabled': np.int64,
    'Census_IsWIMBootEnabled': np.float64,
    'Census_IsVirtualDevice': np.float64,
    'Census_IsTouchEnabled': np.int64,
    'Census_IsPenCapable': np.int64,
    'Census_IsAlwaysOnAlwaysConnectedCapable': np.float64,
    'Wdft_IsGamer': np.int64,
    'Wdft_RegionIdentifier': np.int64,
    'HasDetections': np.int64,
    'EngineVersion_1': np.int64,
    'EngineVersion_2': np.int64,
    'EngineVersion_3': np.int64,
    'EngineVersion_4': np.int64,
    'AppVersion_1': np.int64,
    'AppVersion_2': np.int64,
    'AppVersion_3': np.int64,
    'AppVersion_4': np.int64,
    'AvSigVersion_1': np.int64,
    'AvSigVersion_2': np.float64,
    'AvSigVersion_3': np.int64,
    'AvSigVersion_4': np.int64,
    'OsVer_1': np.int64,
    'OsVer_2': np.int64,
    'OsVer_3': np.int64,
    'OsVer_4': np.int64,
    'OsBuildLab_1': np.float64,
    'OsBuildLab_2': np.float64,
    'OsBuildLab_3': np.int64,
    'OsBuildLab_4': np.int64,
    'OsBuildLab_5': np.float64,
    'OsBuildLab_6': np.float64,
    'Census_OSVersion_1': np.int64,
    'Census_OSVersion_2': np.int64,
    'Census_OSVersion_3': np.int64,
    'Census_OSVersion_4': np.int64
}

print ('Loading csv', file=open("w207_project_v7.log", "a"))
full_features = pd.read_csv(csvfilename, dtype=datatypes, index_col="MachineIdentifier")

# Shuffle the data
#np.random.seed(0)

print ('Shuffling', file=open("w207_project_v7.log", "a"))
shuffle = np.random.permutation(np.arange(full_features.shape[0]))[:totalrows]
indexes = full_features.index[shuffle]
full_features = full_features.loc[indexes,:]
full_labels = full_features["HasDetections"]
full_features = full_features.drop(["HasDetections"], axis=1)

print (full_features.shape, file=open("w207_project_v7.log", "a"))


train_count = int(totalrows * 0.8)

train_features = full_features.values[:train_count]
test_features  = full_features.values[train_count:]

train_labels = full_labels.values[:train_count]
test_labels = full_labels.values[train_count:]

print (train_labels.shape, test_labels.shape, file=open("w207_project_v7.log", "a"))

clf = ske.HistGradientBoostingClassifier(random_state=123)
clf.fit(train_features, train_labels)
all_columns_score = clf.score(test_features, test_labels)
print ("All columns (original)", train_features.shape, "HistGradientBoostingClassifier", all_columns_score*100, file=open("w207_project_v7.log", "a"))

def optimize_score(all_features, labels, current_score, trn_count, tst_count, level):

    print ('Score for level', level, 'is', current_score*100, 'columns', all_features.columns, file=open("w207_project_v7.log", "a"))

    for c in all_features.columns:
        df_features = all_features.drop(c, axis=1)

        train_features = df_features.values[:trn_count]
        test_features  = df_features.values[trn_count:trn_count+tst_count]

        train_labels = labels.values[:trn_count]
        test_labels = labels.values[trn_count:trn_count+tst_count]

        clf = ske.HistGradientBoostingClassifier(random_state=123)
        clf.fit(train_features, train_labels)
        score = clf.score(test_features, test_labels)

        print ('Level', level,': Dropping', c,
               train_features.shape, test_features.shape, "HistGradientBoosting",
               current_score*100, score*100, score >= current_score, score > current_score,
               file=open("w207_project_v7.log", "a"))

        if score > current_score:
            optimize_score(df_features, labels, score, trn_count, tst_count, level + 1)

# Let's try good old brute force ;)
optimize_score(full_features, full_labels, all_columns_score, train_count, totalrows - train_count, 1)

