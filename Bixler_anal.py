# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import plots as acmu_plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools import *
# %matplotlib qt

#wing_left = "/home/dteubl/repos/SEMA/ACMU/logs/bixler/04_13/acmuv1100/activitylog199.csv"
#wing_right = "/home/dteubl/repos/SEMA/ACMU/logs/bixler/04_13/acmuv1101/activitylog175.csv"
#wing_left = "/home/dteubl/repos/SEMA/ACMU/logs/bixler/04_28/acmuv1100/activitylog201.csv"
#wing_right = "/home/dteubl/repos/SEMA/ACMU/logs/bixler/04_28/acmuv1101/activitylog177.csv"
wing_left = "/home/dteubl/repos/SEMA/ACMU/logs/bixler/04_29/acmuv1100/activitylog205.csv"
wing_right = "/home/dteubl/repos/SEMA/ACMU/logs/bixler/04_29/acmuv1101/activitylog183.csv"
header_size = 25
temp = pd.read_csv(wing_left, sep=';', header=header_size)
print(temp)

# +


data_left = acmu_plots.ACMU(wing_left,header_size)
data_right = acmu_plots.ACMU(wing_right,header_size)

def convert_raw_data_to_relative_angles(raw_data, offset):
    return (360/2**12)*(raw_data - offset)

def calc_offset_from_beginning(raw_data, window):
    return (sum(raw_data[0:window])/window)


print(data_left)
print(data_right)

# +


# blue 73 --  flap_a --- Innerside
# red 65 --- flap-b -- outter size
import plotly




# -

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# +
# offset time

t0 = data_left.time_us()[0]
T = data_left.time_us() - data_left.time_us()[0]

# get time as number of samples
tn = np.arange(0,len(T))


# +
servo_pos = 
flap_root = despike_vector(data_left.flap_root())
flap_tip = despike_vector(data_left.flap_tip())




# +

servo_pos = despike_vector(data_left.servo_pos())
flap_root = despike_vector(data_left.flap_root())
flap_tip = despike_vector(data_left.flap_tip())
# -

plt.figure()
plt.plot(tn,data_left.servo_pos(),  label = 'data - dirty')
# plt.plot(tn,servo_pos,  label = 'data - clean')
plt.title("data clean with custom function")
plt.ylabel("amplitude")
plt.xlabel("sample")
plt.legend()
plt.show()

# +
vecc = data_left.servo_pos()[39270:39310]
print(vecc)
vc2 = copy.deepcopy(vecc)

print(vc2-vecc)

# +
servo_pos_r = despike_vector(data_right.servo_pos()[:])
flap_root_r = despike_vector(data_right.flap_root()[:])
flap_tip_r = despike_vector(data_right.flap_tip()[:])
nr =  np.arange(0,len(data_right.time_us()))

vec = flap_tip_r[52250:52300]

mi = 0
ma = len(flap_tip_r)
#mi = 52270
#ma = 52280

plt.figure()
plt.plot(nr[mi:ma],data_right.servo_pos()[mi:ma],  label = 'data - dirty')
plt.plot(nr[mi:ma],servo_pos_r[mi:ma],  label = 'data - clean')
#plt.plot(nr[mi:ma], flap_tip_r[mi:ma]-data_right.flap_tip()[mi:ma], label= "diff")
plt.title("data clean with custom function")
plt.ylabel("amplitude")
plt.xlabel("sample")
plt.grid("on")
plt.legend()
plt.show()





# +
### Close sample on real data...

vec1 = data_right.flap_tip()[52270:52280]
vec2 = despike_vector(vec1[:])
tn = np.arange(0,len(vec2))

plt.figure()
plt.plot(tn,vec1,  label = 'dirty')
plt.plot(tn,vec2,  label = 'clean')
plt.title("data clean with custom function")
plt.ylabel("amplitude")
plt.xlabel("sample")
plt.legend()
plt.show()
print(vec1)
# -

v = [12.23,  11.26,  10.21, -16.08,   8.45,   7.4,    7.13,   6.69,   6.25,   5.99]
vec1 = np.array(v)
d = abs(data_right.servo_pos()[0:-1] - data_right.servo_pos()[1:])
print(d)
tn = np.arange(0,len(d))
plt.figure()
plt.plot(tn,d,  label = 'dirty')
plt.title("derivative of signal")
plt.ylabel("amplitude")
plt.xlabel("sample")
plt.legend()
plt.show()


# +
### Close sample on real data...

vec1 = [12.23,  11.26,  10.21, -16.08,   8.45,   7.4,    7.13,   6.69,   6.25,   5.99]
d = vec1[0:-1]

vec2 = despike_vector(vec1[:])
tn = np.arange(0,len(vec2))

plt.figure()
plt.plot(tn,vec1,  label = 'dirty')
plt.plot(tn,vec2,  label = 'clean')
plt.title("data clean with custom function")
plt.ylabel("amplitude")
plt.xlabel("sample")
plt.legend()
plt.show()
# -

plt.figure()
plt.plot(data_left.time_min(),data_left.servo_pos(),  label = 'Servo')
plt.plot(data_left.time_min(),data_left.flap_tip(),  label = 'Outter side')
plt.plot(data_left.time_min(),data_left.flap_root(), label = 'Inner side')
plt.title("Aileron deflection measurement of HK-Bixler left wing")
plt.ylabel("Defelections [deg]")
plt.xlabel("sample")
plt.legend()
plt.show()

plt.figure()
plt.plot(data_right.time_min(),data_right.servo_pos(),  label = 'Servo')
plt.plot(data_right.time_min(),data_right.flap_tip(),  label = 'Outter side')
plt.plot(data_right.time_min(),data_right.flap_root(), label = 'Inner side')
plt.title("Aileron deflection measurement of HK-Bixler right wing")
plt.ylabel("Defelections [deg]")
plt.xlabel("Time [ms]")
plt.legend()
plt.show()

# +
plt.figure()

pos = data_left.servo_pos()
for i in np.arange(0,len(pos)):
    if (abs(pos[i]) - abs(pos[i-1])) > 5.0:
        pos[i] = pos[i-1]
    

ref_zero = -1*(data_left.ref()-1500)

ref_max = max(abs(ref_zero))
pos_max = max(abs(pos))

ref = (ref_zero/ref_max)* pos_max
pos = (pos/pos_max)* pos_max

plt.plot(data_left.time_s(),ref,  label = 'pwm reference')
plt.plot(data_left.time_s(),data_left.servo_pos(),  label = 'Servo')
plt.plot(data_left.time_s(),data_left.flap_tip(),  label = 'Outter side')
plt.plot(data_left.time_s(),data_left.flap_root(), label = 'Inner side')
plt.title("Aileron deflection measurement of HK-Bixler left wing")
plt.show()
plt.ylabel("Position [deg]")
plt.xlabel("Time [s]")
plt.legend()
plt.show()



# +
plt.figure()

plt.plot(data_left.time_min(), data_left.voltage(), label = 'Input Voltage')
plt.title("Input voltage ripple of HK-Bixler aileron")
plt.ylabel("Voltage [mV]")
plt.xlabel("Time [min]")
plt.legend()
plt.show()


# +
plt.figure()

current = data_left.current() 
plt.plot(data_left.time_min(), current, label = 'consumed current')
plt.title("Input voltage ripple of HK-Bixler aileron")
plt.ylabel("Current [mV]")
plt.xlabel("Time [min]")
plt.legend()
plt.show()


# +
def calc_resistor(measured_value):
    Vin = 5
    R2 = 2000
    resistor = R2*Vin*( 1 - measured_value/Vin )/measured_value


plt.figure()

resistor = calc_resistor(data_left.air_temp())

#plt.plot(data.time_s,resistor , label = 'ambient temperature')
plt.plot(data_left.time_s(),data_left.servo_temp() , label = 'servot temperature')
plt.plot(data_left.time_s(),data_left.air_temp() , label = 'air temperature')
plt.title("temperature changes of HK-Bixler aileron")
plt.ylabel("temperature [raw]")
plt.xlabel("Time [s]")
plt.legend()
plt.show()
print(data_left.air_temp)

# +
# convert the data to degrees.
max_res = 2**12
degree_90 = (360/max_res)*1024
print(degree_90)



# +


pos = data.servo_pos
for i in np.arange(0,len(pos)):
    if (abs(pos[i]) - abs(pos[i-1])) > 5.0:
        pos[i] = pos[i-1]
        
        
# -

gain = 10/12
print(gain)


# +
def calc_resistor(measured_value):
    Vin = 5000 # mV
    R2 = 2000 # ohm
    R = R2*(Vin -measured_value)/measured_value
    return R

def inv_calc(resistor):
    Vin = 5000 # mV
    R2 = 2000 # ohm
    return  Vin * (R2/(resistor + R2))


R = calc_resistor(733)
print(R)
print(inv_calc(11642))




# +
# temperature table

NTC_R = [32554, 25339, 19872, 15872, 12488, 10000, 8059, 6535, 5330, 4372, 3605, 2989, 2490, 2084, 1753, 1481, 1256,1070, 915,786, 677]    
NTC_T = np.arange(0,105,5)

plt.figure()

plt.plot(NTC_T, NTC_R, label = 'Resistor')
plt.title("NTC characteristic")
plt.ylabel("Resistance [Ohm]")
plt.xlabel("Temperature [Celsius]")
plt.legend()
plt.show()




# +
air_t = data.air_temp

#for i in np.arange(0,len(air_t)):
r = (calc_resistor(air_t[1000]))

idxs = NTC_R<=r
print(idxs)
print(np.array(NTC_R)[~np.array(idxs)][-1])
print(r)
val  = np.array(NTC_R)[idxs][0]
print(val)
idx = NTC_R>=val
print(idx)
arr = np.array(NTC_R)[idx][-2:]
print(arr)

def find_boundarie_for_interpolation(vec, point):
    arr = []
    idx = vec<=point
    arr.append(np.array(vec)[idx][0])
    arr.append(np.array(NTC_R)[~np.array(idx)][-1])
    return arr

print(find_boundarie_for_interpolation(NTC_R, r))


# +
N = 1000
Nominal = 8000

print(Nominal-sum(data.voltage[N:N+N])/N)

