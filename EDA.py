import pandas as pd
import pyarrow.parquet as pq
import os
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from scipy.signal import welch


dir_train = 'assets/original_kaggle/train'
n_train_samples = 10000
train = pq.read_pandas(dir_train + '/train.parquet', columns=[str(i) for i in range(n_train_samples)]).to_pandas()

### metadata
train_metadata = pd.DataFrame.from_csv(dir_train + '/metadata_train.csv')

# smaple metadata
train_metadata = train_metadata.loc[:n_train_samples]

"""
# check null
print(train_metadata.isnull().sum())

# check target
print('Target fraction:', np.round(train_metadata.target.value_counts()[1] / train_metadata.target.value_counts()[0] * 100, 2), '%')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sns.countplot(x="target", data=train_metadata, ax=ax1)
sns.countplot(x="target", data=train_metadata, hue="phase", ax=ax2)

print("id_measurement have {} uniques in train".format(train_metadata.id_measurement.nunique()))

### signal data (waves)
train.info()
nan = 0
for col in range(len(train.columns)):
    nan += np.count_nonzero(train.loc[col, :].isnull())
print("train.parquet have {} cols with nulls".format(nan))
print("train.parquet shape is {}".format(train.shape))

# check waves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
for i in range(3):
    sns.lineplot(x=train.index.values, y=train.iloc[:, i], ax=ax1, label=["phase:"+str(train_metadata.iloc[i, :].phase)])
ax1.set_xlabel("example of undamaged signal", fontsize=18)
ax1.set_ylabel("amp", fontsize=18)
ax1.patch.set_facecolor('blue')
ax1.patch.set_alpha(0.2)
for i in range(3, 6):
    sns.lineplot(x=train.index.values, y=train.iloc[:, i], ax=ax2, label=["phase:"+str(train_metadata.iloc[i, :].phase)])
ax2.set_xlabel("example of damaged signal", fontsize=18)
ax2.set_ylabel("amp", fontsize=18)
ax2.patch.set_facecolor('red')
ax2.patch.set_alpha(0.2)

# sample frequency
period = 0.02
time_step = 0.02 / 800000.
time_vec = np.arange(0, 0.02, time_step)
f_sampling = 1 / time_step
print('Sampling Frequency = {} MHz'.format(f_sampling / 1e6))

# wave frequency
from scipy import fftpack, signal
sig = train['1']
sig_fft = fftpack.fft(sig)
# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)
# The corresponding frequencies
sample_freq = fftpack.fftfreq(sig.size, d=time_step)


def bandpassfilter(spec, sample_freq, lowcut, highcut):
    # a digital bandpass filter with a infinite roll off.
    # Note that we will keep the frequency point right at low cut-off and high cut-off frequencies.
    spec1 = spec.copy()
    spec1[np.abs(sample_freq) < lowcut] = 0
    spec1[np.abs(sample_freq) > highcut] = 0
    filtered_sig = fftpack.ifft(spec1)
    return filtered_sig

# plot some filtered signals
lowcut, highcut = 10, 100
filtered_sig0 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 10, 300
filtered_sig1 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 10, 1000
filtered_sig2 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig0, linewidth=3, label='10-100 Hz')
plt.plot(time_vec, filtered_sig1, linewidth=3, label='10-300 Hz')
plt.plot(time_vec, filtered_sig2, linewidth=3, label='10-1000 Hz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')

# We also demonstrate a band-pass filtered and a high-pass filtered signals.
lowcut, highcut = 1000, 1e6
filtered_sig3 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)
lowcut, highcut = 1000, 40e6
filtered_sig4 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig4, linewidth=3, label='Above 1 kHz')
plt.plot(time_vec, filtered_sig3, linewidth=3, label='1 kHz-1 MHz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')

# generate and stack filtered signals
signals = np.zeros((20, 7, 800000))
# mux = pd.MultiIndex.from_product([train.columns, ['orig','sig0', 'sig1', 'sig2']])
# df = pd.DataFrame(signals.reshape(80, -1), columns=mux)

for col in train.columns[:1]:

    sig = train[col]

    lowcut, highcut = 10, 100
    filtered_sig0 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)

    lowcut, highcut = 10, 300
    filtered_sig1 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)

    lowcut, highcut = 10, 1000
    filtered_sig2 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)

    sig_deviation_sig2 = abs(sig - filtered_sig2)

    lowcut, highcut = 1000, 1e6
    filtered_sig3 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)

    lowcut, highcut = 1000, 40e6
    filtered_sig4 = bandpassfilter(sig_fft,sample_freq, lowcut, highcut)

    signals[int(col)] = np.stack([sig.values, filtered_sig0, filtered_sig1, filtered_sig2, sig_deviation_sig2, filtered_sig3, filtered_sig4], axis =0)
    # df[col] = [[s0, s1, s2, s3] for s0, s1, s2, s3 in zip(sig.values, filtered_sig0, filtered_sig1, filtered_sig2)]
    # df[col] = [sig.values, filtered_sig0, filtered_sig1, filtered_sig2]

"""


### extract extra features
train_extra_features = pd.DataFrame()
for col in train.columns:
    train_col = train[col]
    train_extra_features.loc[int(col), 'signal_id'] = int(col)
    train_extra_features.loc[int(col), 'wave_min'] = train_col.min()
    train_extra_features.loc[int(col), 'wave_max'] = train_col.max()
    train_extra_features.loc[int(col), 'wave_mean'] = train_col.mean()
    train_extra_features.loc[int(col), 'wave_sum'] = train_col.sum()
    train_extra_features.loc[int(col), 'wave_std'] = train_col.std()
del col, train_col

# train_extra_features['signal_id'] = train.columns
# train_extra_features["signal_max"] = train.agg(np.min).values
# train_extra_features["signal_min"] = train.agg(np.max).values
# train_extra_features["signal_mean"] = train.agg(np.mean).values
# train_extra_features["signal_sum"] = train.agg(np.sum).values
# train_extra_features["signal_std"] = train.agg(np.std).values

train_metadata = pd.merge(train_metadata, train_extra_features, on='signal_id')

def welch_max_power_and_frequency(sig):
    f, pxx = welch(sig)
    ix = np.argmax(pxx)
    strong_count = np.sum(pxx>2.5)
    avg_amp = np.mean(pxx)
    sum_amp = np.sum(pxx)
    std_amp = np.std(pxx)
    median_amp = np.median(pxx)
    return [pxx[ix], f[ix], strong_count, avg_amp, sum_amp, std_amp, median_amp]

power_spectrum_summary_features = train.apply(lambda x: welch_max_power_and_frequency(x), result_type="expand")
power_spectrum_summary_features = power_spectrum_summary_features.T.rename(columns={0:"max_amp", 1:"max_freq", 2:"strong_amp_count",
                                                                  3:"avg_amp", 4:"sum_amp", 5:"std_amp", 6:"median_amp"})
power_spectrum_summary_features.index = power_spectrum_summary_features.index.astype(int)

train_metadata = train_metadata.merge(power_spectrum_summary_features, left_on="signal_id", right_index=True)
# train_metadata = pd.concat([train_metadata, power_spectrum_summary_features], axis=1, join_axes=[power_spectrum_summary_features.index])

feature_names = ["phase"] + train_metadata.columns[4:].tolist()
feature_names

train_metadata.to_csv(dir_train + '/metadata_train_incl_extraFeatures.csv', sep=';')
# train_metadata = pd.DataFrame.from_csv(dir_train + '/metadata_train_incl_extraFeatures.csv', sep=';')
