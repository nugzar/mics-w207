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

if len(sys.argv) != 2:
    print("program csvfile")
    exit()

csvfilename = sys.argv[1]

datatypes = {
    'ProductName': np.int8,
    'IsBeta': np.int8,
    'RtpStateBitfield': np.int64,
    'IsSxsPassiveMode': np.int8,
    'AVProductStatesIdentifier': np.int64,
    'AVProductsInstalled': np.int64,
    'AVProductsEnabled': np.int64,
    'CountryIdentifier': np.int64,
    'CityIdentifier': np.int64,
    'OrganizationIdentifier': np.int64,
    'GeoNameIdentifier': np.int64,
    'LocaleEnglishNameIdentifier': np.int64,
    'Platform': np.int16,
    'Processor': np.int8,
    'OsSuite': np.int64,
    'OsPlatformSubRelease': np.int16,
    'SkuEdition': np.int8,
    'IsProtected': np.int64,
    'AutoSampleOptIn': np.int8,
    'SMode': np.int64,
    'IeVerIdentifier': np.int64,
    'SmartScreen': np.int8,
    'Firewall': np.int64,
    'UacLuaenable': np.int64,
    'Census_MDC2FormFactor': np.int8,
    'Census_DeviceFamily': np.int8,
    'Census_OEMNameIdentifier': np.int64,
    'Census_ProcessorManufacturerIdentifier': np.int64,
    'Census_ProcessorModelIdentifier': np.int64,
    'Census_PrimaryDiskTotalCapacity': np.float64,
    'Census_PrimaryDiskTypeName': np.int8,
    'Census_SystemVolumeTotalCapacity': np.float64,
    'Census_HasOpticalDiskDrive': np.int8,
    'Census_TotalPhysicalRAM': np.float64,
    'Census_ChassisTypeName': np.int8,
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': np.float64,
    'Census_InternalPrimaryDisplayResolutionHorizontal': np.int64,
    'Census_InternalPrimaryDisplayResolutionVertical': np.int64,
    'Census_PowerPlatformRoleName': np.int8,
    'Census_InternalBatteryNumberOfCharges': np.float64,
    'Census_OSArchitecture': np.int8,
    'Census_OSBranch': np.int8,
    'Census_OSBuildNumber': np.int64,
    'Census_OSBuildRevision': np.int64,
    'Census_OSEdition': np.int8,
    'Census_OSInstallTypeName': np.int8,
    'Census_OSInstallLanguageIdentifier': np.int64,
    'Census_OSUILocaleIdentifier': np.int64,
    'Census_OSWUAutoUpdateOptionsName': np.int8,
    'Census_IsPortableOperatingSystem': np.int8,
    'Census_GenuineStateName': np.int8,
    'Census_ActivationChannel': np.int8,
    'Census_IsFlightsDisabled': np.int64,
    'Census_FlightRing': np.int8,
    'Census_ThresholdOptIn': np.int64,
    'Census_FirmwareManufacturerIdentifier': np.int64,
    'Census_FirmwareVersionIdentifier': np.int64,
    'Census_IsSecureBootEnabled': np.int8,
    'Census_IsWIMBootEnabled': np.int64,
    'Census_IsVirtualDevice': np.int64,
    'Census_IsTouchEnabled': np.int8,
    'Census_IsPenCapable': np.int8,
    'Census_IsAlwaysOnAlwaysConnectedCapable': np.int64,
    'Wdft_IsGamer': np.int64,
    'Wdft_RegionIdentifier': np.int64,
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
    'OsBuildLab_1': np.int64,
    'OsBuildLab_2': np.int64,
    'OsBuildLab_3': np.int8,
    'OsBuildLab_4': np.int8,
    'OsBuildLab_5': np.int64,
    'OsBuildLab_6': np.int64,
    'Census_OSVersion_1': np.int64,
    'Census_OSVersion_2': np.int64,
    'Census_OSVersion_3': np.int64,
    'Census_OSVersion_4': np.int64,
    'CORE': np.int64,
    'EDUCATION': np.int64,
    'PRO': np.int64,
    'ENTERPRISE': np.int64,
    'CLOUD': np.int64,
    'SERVER': np.int64,
    'EVALUATION': np.int64,
    'ScreenProportion': np.float64,
    'ScreenDimensions': np.int64,
    'CapacityDifference': np.float64,
    'CapacityRatio': np.float64,
    'RAMByCores': np.float64,
    'HasDetections': np.int8
}

print ('Loading csv', file=open("w207_project_bruteforce.log", "a"))
full_features = pd.read_csv(csvfilename, dtype=datatypes, index_col="MachineIdentifier")
totalrows = len(full_features)

full_labels = full_features["HasDetections"]
full_features = full_features.drop(["HasDetections"], axis=1)

print (full_features.shape, file=open("w207_project_bruteforce.log", "a"))

train_count = int(totalrows * 0.8)

train_features = full_features.values[:train_count]
test_features  = full_features.values[train_count:]

train_labels = full_labels.values[:train_count]
test_labels = full_labels.values[train_count:]

print (train_labels.shape, test_labels.shape, file=open("w207_project_bruteforce.log", "a"))

clf = ske.HistGradientBoostingClassifier(random_state=123)
clf.fit(train_features, train_labels)
all_columns_score = clf.score(test_features, test_labels)
print ("All columns (original)", train_features.shape, "HistGradientBoostingClassifier", all_columns_score*100, file=open("w207_project_bruteforce.log", "a"))

def optimize_score(all_features, labels, current_score, trn_count, tst_count, level, excluded_columns):

    print ('Score for level', level, 'is', current_score*100, 'columns', all_features.columns, file=open("w207_project_bruteforce.log", "a"))
    processed_columns = []
    processed_columns.extend(excluded_columns)

    for c in all_features.columns:
        if c in processed_columns:
            continue

        processed_columns.append(c)
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
               file=open("w207_project_bruteforce.log", "a"))

        if score >= current_score:
            optimize_score(df_features, labels, score, trn_count, tst_count, level + 1, processed_columns)

# Let's try good old brute force ;)
optimize_score(full_features, full_labels, all_columns_score, train_count, totalrows - train_count, 1, [])
