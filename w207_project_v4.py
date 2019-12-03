import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import sklearn.ensemble as ske
from sklearn.model_selection import train_test_split
from sklearn import tree, linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


datatypes = {'ProductName': str, 'EngineVersion': str, 'AppVersion': str, 'AvSigVersion': str, 'IsBeta': np.int64,
    'RtpStateBitfield': np.float64, 'IsSxsPassiveMode': np.int64, 'DefaultBrowsersIdentifier': np.float64,
    'AVProductStatesIdentifier': np.float64, 'AVProductsInstalled': np.float64, 'AVProductsEnabled': np.float64,
    'HasTpm': np.int64, 'CountryIdentifier': np.int64, 'CityIdentifier': np.float64, 'OrganizationIdentifier': np.float64,
    'GeoNameIdentifier': np.float64, 'LocaleEnglishNameIdentifier': np.int64, 'Platform': str, 'Processor': str,
    'OsVer': str, 'OsBuild': np.int64, 'OsSuite': np.int64, 'OsPlatformSubRelease': str, 'OsBuildLab': str,
    'SkuEdition': str, 'IsProtected': np.float64, 'AutoSampleOptIn': np.int64, 'PuaMode': str, 'SMode': np.float64,
    'IeVerIdentifier': np.float64, 'SmartScreen': str, 'Firewall': np.float64, 'UacLuaenable': np.float64,
    'Census_MDC2FormFactor': str, 'Census_DeviceFamily': str, 'Census_OEMNameIdentifier': np.float64,
    'Census_OEMModelIdentifier': np.float64, 'Census_ProcessorCoreCount': np.float64, 
    'Census_ProcessorManufacturerIdentifier': np.float64, 'Census_ProcessorModelIdentifier': np.float64,
    'Census_ProcessorClass': str, 'Census_PrimaryDiskTotalCapacity': np.float64, 
    'Census_PrimaryDiskTypeName': str, 'Census_SystemVolumeTotalCapacity': np.float64,
    'Census_HasOpticalDiskDrive': np.int64, 'Census_TotalPhysicalRAM': np.float64, 'Census_ChassisTypeName': str,
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': np.float64, 
    'Census_InternalPrimaryDisplayResolutionHorizontal': np.float64, 
    'Census_InternalPrimaryDisplayResolutionVertical': np.float64, 'Census_PowerPlatformRoleName': str,
    'Census_InternalBatteryType': str, 'Census_InternalBatteryNumberOfCharges': np.float64, 
    'Census_OSVersion': str, 'Census_OSArchitecture': str, 'Census_OSBranch': str, 'Census_OSBuildNumber': np.int64,
    'Census_OSBuildRevision': np.int64, 'Census_OSEdition': str, 'Census_OSSkuName': str, 
    'Census_OSInstallTypeName': str, 'Census_OSInstallLanguageIdentifier': np.float64, 
    'Census_OSUILocaleIdentifier': np.int64, 'Census_OSWUAutoUpdateOptionsName': str, 
    'Census_IsPortableOperatingSystem': np.int64, 'Census_GenuineStateName': str, 'Census_ActivationChannel': str,
    'Census_IsFlightingInternal': np.float64, 'Census_IsFlightsDisabled': np.float64, 'Census_FlightRing': str,
    'Census_ThresholdOptIn': np.float64, 'Census_FirmwareManufacturerIdentifier': np.float64, 
    'Census_FirmwareVersionIdentifier': np.float64, 'Census_IsSecureBootEnabled': np.int64, 
    'Census_IsWIMBootEnabled': np.float64, 'Census_IsVirtualDevice': np.float64, 'Census_IsTouchEnabled': np.int64,
    'Census_IsPenCapable': np.int64, 'Census_IsAlwaysOnAlwaysConnectedCapable': np.float64, 'Wdft_IsGamer': np.float64,
    'Wdft_RegionIdentifier': np.float64, 
    'HasDetections': np.int64}

full_features = pd.read_csv("./csv/train.csv", dtype=datatypes, index_col="MachineIdentifier")
#full_features = pd.read_csv("./csv/train.csv", dtype=datatypes, nrows=200000, index_col="MachineIdentifier")
full_labels = full_features["HasDetections"]

# Dropping labels ["HasDetections"] from training dataset
full_features = full_features.drop(["HasDetections"], axis=1)

print (full_features.shape)

# Checking the columns with the most NULL values
print((full_features.isnull().sum()).sort_values(ascending=False).head(10))

full_features = full_features.drop(['PuaMode','Census_ProcessorClass','DefaultBrowsersIdentifier','Census_InternalBatteryType','Census_OSEdition','Census_IsFlightingInternal'], axis=1)

string_columns = []

for colname in full_features.dtypes.keys():
    if full_features[colname].dtypes.name == "object":
        string_columns.append(colname)

print (string_columns)

full_features[string_columns].head(10)

def df_replacevalues(df, colname, oldvalues, newvalues):
    # First, we need to get the most frequent value of the column
    topvalue = df[colname].value_counts().idxmax()

    # Replace NaN values with the popular value
    df[colname].fillna(topvalue, inplace=True)

    # We need to make sure no other value than oldvalues exists
    indexes = df[~df[colname].isin(oldvalues)].index

    # If the "Garbage" values are more than 1%, then raise an error
    if len(indexes) > len(df) / 100:
        raise Exception("Not all neccessary values are present in oldvalues array")

    # Replace "Garbage" with the top value
    df.loc[indexes,[colname]] = topvalue

    print ("Previous values", df[colname].unique())
    df[colname] = pd.to_numeric(df[colname].replace(oldvalues, newvalues), errors='raise', downcast='integer')
    print ("New values", df[colname].unique())


colname = "ProductName"
oldvalues = ['win8defender','mse','mseprerelease','windowsintune','fep','scep']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Platform"
oldvalues = ['windows10','windows7','windows8','windows2016']
newvalues = [10,7,8,2016]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Processor"
oldvalues = ['x64','arm64','x86']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "OsPlatformSubRelease"
oldvalues = ['rs4','rs1','rs3','windows7','windows8.1','th1','rs2','th2','prers5']
newvalues = [504,501,503,507,508,201,502,202,405]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "SkuEdition"
oldvalues = ['Pro','Home','Invalid','Enterprise LTSB','Enterprise','Education','Cloud','Server']
newvalues = [55,52,0,71,70,20,90,80]
df_replacevalues(full_features, colname, oldvalues, newvalues)

print(full_features["SmartScreen"].value_counts())

print(full_features["SmartScreen"].unique())

colname = "SmartScreen"
oldvalues = ['Off','off','OFF','On','on','Warn','Prompt','ExistsNotSet','Block','RequireAdmin']
newvalues = [0,0,0,1,1,2,3,4,5,6]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_MDC2FormFactor"
oldvalues = ['Desktop','Notebook','Detachable','PCOther','AllInOne','Convertible','SmallTablet','LargeTablet','SmallServer','LargeServer','MediumServer','ServerOther','IoTOther']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_DeviceFamily"
oldvalues = ['Windows.Desktop','Windows.Server','Windows']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_PrimaryDiskTypeName"
oldvalues = ['HDD','SSD','UNKNOWN','Unspecified']
newvalues = [1,2,3,3]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_ChassisTypeName"
oldvalues = ['Notebook', 'Desktop', 'Laptop', 'Portable', 'AllinOne', 'MiniTower', 'Convertible', 'Other', 'UNKNOWN', 'Detachable', 
             'LowProfileDesktop', 'HandHeld', 'SpaceSaving', 'Tablet', 'Tower', 'Unknown', 'MainServerChassis', 'MiniPC', 'LunchBox', 
             'RackMountChassis', 'SubNotebook', 'BusExpansionChassis']
newvalues = [1,2,1,1,3,4,5,6,-1,7,8,9,10,11,12,-1,13,2,14,15,1,16]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_PowerPlatformRoleName"
oldvalues = ['Mobile', 'Desktop', 'Slate', 'Workstation', 'SOHOServer', 'UNKNOWN', 'EnterpriseServer', 'AppliancePC', 'PerformanceServer', 'Unspecified']
newvalues = [1,2,3,2,4,-1,5,6,7,-1]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_OSArchitecture"
oldvalues = ['amd64', 'x86', 'arm64']
newvalues = [1,3,2]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_OSBranch"
oldvalues = ['rs4_release', 'rs3_release', 'rs3_release_svc_escrow', 'rs2_release', 'rs1_release', 'th2_release', 'th2_release_sec', 'th1_st1', 'th1', 'rs5_release', 'rs3_release_svc_escrow_im', 'rs_prerelease', 'rs_prerelease_flt', 'rs5_release_sigma']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_OSSkuName"
oldvalues = ['CORE', 'PROFESSIONAL', 'CORE_SINGLELANGUAGE', 'CORE_COUNTRYSPECIFIC', 'EDUCATION', 'ENTERPRISE', 'PROFESSIONAL_N', 'ENTERPRISE_S', 'STANDARD_SERVER', 'CLOUD', 'CORE_N', 'STANDARD_EVALUATION_SERVER', 'EDUCATION_N', 'ENTERPRISE_S_N', 'DATACENTER_EVALUATION_SERVER', 'SB_SOLUTION_SERVER', 'ENTERPRISE_N', 'PRO_WORKSTATION', 'UNLICENSED']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_OSInstallTypeName"
oldvalues = ['UUPUpgrade', 'IBSClean', 'Update', 'Upgrade', 'Other', 'Reset', 'Refresh', 'Clean', 'CleanPCRefresh']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_OSWUAutoUpdateOptionsName"
oldvalues = ['FullAuto', 'UNKNOWN', 'Notify', 'AutoInstallAndRebootAtMaintenanceTime', 'Off', 'DownloadNotify']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_GenuineStateName"
oldvalues = ['IS_GENUINE', 'INVALID_LICENSE', 'OFFLINE', 'UNKNOWN', 'TAMPERED']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_ActivationChannel"
oldvalues = ['Retail', 'OEM:DM', 'Volume:GVLK', 'OEM:NONSLP', 'Volume:MAK', 'Retail:TB:Eval']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_FlightRing"
oldvalues = ['Retail', 'NOT_SET', 'Unknown', 'WIS', 'WIF', 'RP', 'Disabled', 'OSG', 'Canary', 'Invalid']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "Census_FlightRing"
print (full_features[colname].value_counts())
print (colname, full_features[colname].value_counts().keys())

print (full_features[["ProductName","EngineVersion","AppVersion","AvSigVersion","Platform","Processor","OsVer","OsPlatformSubRelease","OsBuildLab","SkuEdition","SmartScreen","Census_MDC2FormFactor","Census_DeviceFamily","Census_PrimaryDiskTypeName","Census_ChassisTypeName","Census_PowerPlatformRoleName", "Census_OSVersion","Census_OSArchitecture","Census_OSBranch","Census_OSSkuName","Census_OSInstallTypeName","Census_OSWUAutoUpdateOptionsName","Census_GenuineStateName", "Census_ActivationChannel","Census_FlightRing"]].head(10))

versions = ['EngineVersion','AppVersion','AvSigVersion','OsVer','OsBuildLab','Census_OSVersion']
newcolumnnames = []

for colname in versions:
    data = full_features[colname].str.split(r"\.|-",expand=True) # Split if '.' or '-'
    for i in range(data.shape[1]):
        newcolumnname = "%s_%d" % (colname, i+1)
        newcolumnnames.append(newcolumnname)
        full_features[newcolumnname] = data[i]

print(full_features[newcolumnnames].head(10))

colname = "OsBuildLab_3"
oldvalues = ['amd64fre', 'x86fre', 'arm64fre']
newvalues = [1,3,2]
df_replacevalues(full_features, colname, oldvalues, newvalues)

colname = "OsBuildLab_4"
oldvalues = ['rs4_release', 'rs3_release_svc_escrow', 'rs3_release', 'rs2_release', 'rs1_release', 'th2_release_sec', 'th1', 'winblue_ltsb_escrow', 'th2_release', 'rs1_release_inmarket', 'winblue_ltsb', 'win7sp1_ldr', 'rs3_release_svc', 'rs1_release_1', 'win7sp1_ldr_escrow', 'rs1_release_sec', 'th1_st1', 'rs5_release', 'rs1_release_inmarket_aim', 'rs3_release_svc_escrow_im', 'th2_release_inmarket', 'rs_prerelease', 'rs_prerelease_flt', 'win7sp1_gdr', 'winblue_gdr', 'th1_escrow', 'win7_gdr', 'winblue_r4', 'rs1_release_inmarket_rim', 'rs1_release_d', 'winblue_r9', 'winblue_r5', 'win7_rtm', 'win7sp1_rtm', 'winblue_r7', 'winblue_r3', 'winblue_r8', 'rs5_release_sigma', 'win7_ldr', 'rs5_release_sigma_dev', 'rs_xbox', 'rs5_release_edge', 'winblue_rtm', 'win7sp1_rc', 'rs3_release_svc_sec', 'rs_onecore_base_cobalt', 'rs6_prerelease', 'rs_onecore_sigma_grfx_dev', 'rs_onecore_stack_per1', 'rs5_release_sign', 'rs_shell']
newvalues = [i+1 for i in range(len(oldvalues))]
df_replacevalues(full_features, colname, oldvalues, newvalues)

versions = ['EngineVersion','AppVersion','AvSigVersion','OsVer','OsBuildLab','Census_OSVersion']
full_features = full_features.drop(versions, axis=1)

for colname in full_features.columns:
    full_features[colname] = pd.to_numeric(full_features[colname], errors='coerce')
    topvalue = full_features[colname].value_counts().idxmax()
    full_features[colname].fillna(topvalue, inplace=True)

print(full_features.head(10))

# Let's see some details of the loaded data
print(full_features.describe())

# Shuffle the data
#np.random.seed(0)
shuffle = np.random.permutation(np.arange(full_features.shape[0]))
train_features, test_features, train_labels, test_labels = train_test_split(full_features.values[shuffle], full_labels.values[shuffle], train_size=0.90)
print (train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

scaler = StandardScaler()
scaler.fit(train_features)

normalized_train_features = scaler.transform(train_features)
normalized_test_features = scaler.transform(test_features)

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


h = .02  # step size in the mesh

names = [ #"Nearest Neighbors",
          #"Linear SVM",
          #"RBF SVM",
          #"Gaussian Process",
         "Decision Tree",
         "Random Forest",
         "Neural Net",
         "AdaBoost",
         "Naive Bayes",
         "QDA"]

classifiers = [
    #KNeighborsClassifier(3),
    #SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=20, max_features=5),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

results = {}
print('Testing algorithms using normalized original dataset...\n')

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(normalized_train_features, train_labels)
    score = clf.score(normalized_test_features, test_labels)
    print("%s : %f %%" % (name, score*100))
    results[name] = score

winner = max(results, key=results.get)
print()
print(f'Winning algorithm is {winner} with a {results[winner]*100}% accuracy')

