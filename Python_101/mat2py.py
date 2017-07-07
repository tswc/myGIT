import scipy.io as sio
import numpy as np
import TJ_pack as tj

filepath = r'C:\Users\TSWC\Google Drive\FA\data_for_python\FQ101\\'

filename = [None]*23
filename[0] = 'OM2017_05_18_M12_HF110_NF5_angle_offset_cal.mat'
filename[1] = 'OM2017_05_18_M12_HF110_NF5_cu_circle_cal.mat'
filename[2] = 'OM2017_05_18_M12_HF110_NF5_cu_circle_flush_cal.mat'
filename[3] = 'OM2017_05_18_M12_HF110_NF5_emb_angle_2p5_cal.mat'
filename[4] = 'OM2017_05_18_M12_HF110_NF5_emb_angle_4p5_cal.mat'
filename[5] = 'OM2017_05_18_M12_HF110_NF5_emb_angle_6p3_cal.mat'
filename[6] = 'OM2017_05_18_M12_HF110_NF5_emb_angle_8p0_cal.mat'
filename[7] = 'OM2017_05_18_M12_HF110_NF5_emb_angle_9p2_cal.mat'
filename[8] = 'OM2017_05_18_M12_HF110_NF5_emb_angle_flush_cal.mat'
filename[9] = 'OM2017_05_18_M12_HF110_NF5_not_emb0_cal.mat'
filename[10] = 'OM2017_05_18_M12_HF110_NF5_not_emb1_cal.mat'
filename[11] = 'OM2017_06_01_M12_HF125_NF5_cal.mat'
filename[12] = 'OM2017_06_01_M12_HF125_NF5_emb_winkel_x0_cal.mat'
filename[13] = 'OM2017_06_01_M12_HF125_NF5_emb_winkel_x1_cal.mat'
filename[14] = 'OM2017_06_01_M12_HF125_NF5_emb_winkel_x2_cal.mat'
filename[15] = 'OM2017_06_01_M12_HF125_NF5_emb_winkel_x3_cal.mat'
filename[16] = 'OM2017_06_01_M12_HF125_NF5_emb_winkel_x4_cal.mat'
filename[17] = 'OM2017_06_01_M12_HF125_NF5_emb_winkel_x5_cal.mat'
filename[18] = 'OM2017_06_01_M12_HF125_NF5_emb_winkel_x6_cal.mat'
filename[19] = 'OM2017_06_01_M12_HF125_NF5_emb_winkel_x7_cal.mat'
filename[20] = 'OM2017_06_01_M12_HF125_NF5_emb_winkel_x8_cal.mat'
filename[21] = 'OM2017_06_01_M12_HF125_NF5_emb_winkel_x9_cal.mat'
filename[22] = 'OM2017_06_01_M12_HF125_NF5_emb_winkel_x10_cal.mat'


filepath2 = r'C:\Users\TSWC\Google Drive\FA\data_for_python\6.23\\'
filename2 = [None]*55
for i in range(55):
    filename2[i] = 'OM'+ str(i+1)



FQ101_data = tj.mat2py(filepath, filename)

FQ101_data2 = tj.mat2py(filepath2,filename2)


data = np.row_stack((FQ101_data,FQ101_data2))
print data
print FQ101_data.shape
print FQ101_data2.shape
print data.shape
np.save("FQ101_data", data)
