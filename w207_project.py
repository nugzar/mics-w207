import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA

# set up display area to show dataframe in jupyter qtconsole
# pd.set_option('display.height', 1000)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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

full_features = pd.read_csv("./csv/train.csv", dtype=datatypes, nrows=10000)

print (full_features.shape)

# Let's see some details of the loaded data
#print (full_features.describe())

# We have 82 columns. It is interetsing to see the unique values by them (k-Diversity)
#for col in full_features.columns:
#    print (col, full_features[col].unique())

# Sanitizing the data. This probably can be done by PCA, but lets do some quick manual check before stressing the system
# Checking the columns with the most number of NULL values and displaying top 5 of them
print ((full_features.isnull().sum()).sort_values(ascending=False).head(5))

# Checking the columns with the most number of NA values and displaying top 5 of them
#(full_features.isna().sum()).sort_values(ascending=False).head(5)

# As we see, PuaMode and Census_ProcessorClass are the columns that are almost useless.
# I think we can drop them

full_features = full_features.drop(['PuaMode', 'Census_ProcessorClass', 'MachineIdentifier'], axis=1)

# As seen i data analysis above, there are also many NaN values in the dataset
# Let's replace them either by -1 it the data type is numeric or by 'N/A' if
# the data type is string

nanvalues = {
    'ProductName': 'N/A', 'EngineVersion': 'N/A', 'AppVersion': 'N/A', 'AvSigVersion': 'N/A', 'IsBeta': -1,
    'RtpStateBitfield': -1., 'IsSxsPassiveMode': -1, 'DefaultBrowsersIdentifier': -1.,
    'AVProductStatesIdentifier': -1., 'AVProductsInstalled': -1., 'AVProductsEnabled': -1.,
    'HasTpm': -1, 'CountryIdentifier': -1, 'CityIdentifier': -1., 'OrganizationIdentifier': -1.,
    'GeoNameIdentifier': -1, 'LocaleEnglishNameIdentifier': -1, 'Platform': 'N/A', 'Processor': 'N/A',
    'OsVer': 'N/A', 'OsBuild': -1, 'OsSuite': -1, 'OsPlatformSubRelease': 'N/A', 'OsBuildLab': 'N/A',
    'SkuEdition': 'N/A', 'IsProtected': -1., 'AutoSampleOptIn': -1, 'SMode': -1.,
    'IeVerIdentifier': -1., 'SmartScreen': 'N/A', 'Firewall': -1., 'UacLuaenable': -1.,
    'Census_MDC2FormFactor': 'N/A', 'Census_DeviceFamily': 'N/A', 'Census_OEMNameIdentifier': -1.,
    'Census_OEMModelIdentifier': -1., 'Census_ProcessorCoreCount': -1., 
    'Census_ProcessorManufacturerIdentifier': -1., 'Census_ProcessorModelIdentifier': -1.,
    'Census_PrimaryDiskTotalCapacity': -1., 
    'Census_PrimaryDiskTypeName': 'N/A', 'Census_SystemVolumeTotalCapacity': -1.,
    'Census_HasOpticalDiskDrive': -1, 'Census_TotalPhysicalRAM': -1., 'Census_ChassisTypeName': 'N/A',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': -1., 
    'Census_InternalPrimaryDisplayResolutionHorizontal': -1., 
    'Census_InternalPrimaryDisplayResolutionVertical': -1., 'Census_PowerPlatformRoleName': 'N/A',
    'Census_InternalBatteryType': 'N/A', 'Census_InternalBatteryNumberOfCharges': -1., 
    'Census_OSVersion': 'N/A', 'Census_OSArchitecture': 'N/A', 'Census_OSBranch': 'N/A', 'Census_OSBuildNumber': -1,
    'Census_OSBuildRevision': -1, 'Census_OSEdition': 'N/A', 'Census_OSSkuName': 'N/A', 
    'Census_OSInstallTypeName': 'N/A', 'Census_OSInstallLanguageIdentifier': -1., 
    'Census_OSUILocaleIdentifier': -1, 'Census_OSWUAutoUpdateOptionsName': 'N/A', 
    'Census_IsPortableOperatingSystem': -1, 'Census_GenuineStateName': 'N/A', 'Census_ActivationChannel': 'N/A',
    'Census_IsFlightingInternal': -1., 'Census_IsFlightsDisabled': -1., 'Census_FlightRing': 'N/A',
    'Census_ThresholdOptIn': -1., 'Census_FirmwareManufacturerIdentifier': -1., 
    'Census_FirmwareVersionIdentifier': -1., 'Census_IsSecureBootEnabled': -1, 
    'Census_IsWIMBootEnabled': -1., 'Census_IsVirtualDevice': -1., 'Census_IsTouchEnabled': -1,
    'Census_IsPenCapable': -1, 'Census_IsAlwaysOnAlwaysConnectedCapable': -1., 'Wdft_IsGamer': -1.,
    'Wdft_RegionIdentifier': -1.
}

full_features = full_features.fillna(value=nanvalues)

print (full_features)

full_labels = full_features[["HasDetections"]]

# Shuffle the data
np.random.seed(0)
shuffle = np.random.permutation(np.arange(full_features.shape[0]))

train_features, test_features, train_labels, test_labels = \
    train_test_split(full_features.values[shuffle], full_labels.values[shuffle], train_size=0.90)

print (train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

enc = OrdinalEncoder() #OneHotEncoder() #(sparse=False)
X = enc.fit_transform(train_features)

print(X.shape)

model = PCA()
model.fit_transform(X)

# We need cumulative sums by components
variances = model.explained_variance_ratio_.cumsum()

for k in range(len(variances)):
    print ("k =", k + 1, " Variance =", variances[k])

