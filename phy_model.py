import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin,cos
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


root = '/Users/guoqiushi/Documents/thesis/indoor_final/'


class fileObject:

    def __init__(self, root, index):
        self.root = root
        self.index = index
        self.path = os.path.join(root, str(index))
        for files in os.listdir(self.path):
            if files.startswith('IMU'):
                self.imu_path = os.path.join(self.path, files)
            elif files.startswith('Mag'):
                self.mag_path = os.path.join(self.path, files)
            elif files.startswith('ultra'):
                self.ultra_path = os.path.join(self.path, files)

        self.height_dict = {
            '1': 0.736,
            '2': 0.782,
            '3': 0.827,
            '4': 0.877,
            '5': 0.936,
            '6': 0.981,
            '7': 1.043,
            '8': 1.102,
            '9': 1.10
        }
    '''
    get the ultra after the interpolation
    '''
    def get_ultra(self):
        Dataframe = pd.read_csv(self.ultra_path, sep=" ",
                                names=["ID", "P_x", "P_y", "P_z", "unknow", "unknow_1", "Time_Stamp"])
        Dataframe = Dataframe.drop(["ID", "unknow", "unknow_1"], axis=1)
        rows, columns = Dataframe.shape
        serial_num = np.arange(0, rows, 1).tolist()
        serial_num_inte = np.arange(0, rows, 0.1).tolist()
        Px_inte = np.interp(serial_num_inte, serial_num, Dataframe["P_x"])
        Py_inte = np.interp(serial_num_inte, serial_num, Dataframe["P_y"])
        Pz_inte = np.interp(serial_num_inte, serial_num, Dataframe["P_z"])
        Pt_inte = np.interp(serial_num_inte, serial_num, Dataframe["Time_Stamp"])
        Dataframe_inte = pd.DataFrame(columns=["Serial_Num", 'P_X', 'P_Y', 'P_Z', "Time_Stamp"])
        Dataframe_inte["Serial_Num"] = np.arange(0, len(serial_num_inte), 1).tolist()
        Dataframe_inte["P_X"] = Px_inte
        Dataframe_inte["P_Y"] = Py_inte
        Dataframe_inte['P_Z'] = self.height_dict[str(self.index)]
        # Dataframe_inte["P_Z"] = Pz_inte
        Dataframe_inte["Time_Stamp"] = Pt_inte
        return Dataframe_inte

    def get_mag(self):
        df_tem = pd.read_csv(self.mag_path, sep=" ", names=["Serial_Num", "R_x", "R_y", "R_z", "Time_Stamp"],
                             index_col="Serial_Num")
        df_tem["Magnitude"] = np.sqrt(
            np.power(df_tem["R_x"], 2) + np.power(df_tem["R_y"], 2) + np.power(df_tem["R_z"], 2))
        return df_tem

    def get_imu(self):
        df_tem = pd.read_csv(self.imu_path, sep=" ",
                             names=['a_x', 'a_y', 'a_z', 'G_x', 'G_y', 'G_z', 'M_x', 'M_y', 'M_z', 'alpha', 'beta',
                                    'gama', 'mill', 'Time_Stamp', 'index'],
                             index_col="index")
        return df_tem

    '''
    return the acceleration and euler angle of the time
    '''
    def get_IMU_info(self, time_mag, df_imu):
        time_imu = df_imu['Time_Stamp'].to_numpy()
        sub_array = abs(time_imu - time_mag)
        index = np.argmin(sub_array)
        alpha, beta, gama = df_imu.iloc[index, 9], df_imu.iloc[index, 10], df_imu.iloc[index, 11]
        a_x, a_y, a_z = df_imu.iloc[index, 0], df_imu.iloc[index, 1], df_imu.iloc[index, 2]
        return a_x, a_y, a_z, alpha, beta, gama

    def get_position(self, time_mag, df_ultra):
        time_ultra = df_ultra['Time_Stamp'].to_numpy()
        sub_array = abs(time_ultra - time_mag)
        index = np.argmin(sub_array)
        p_x, p_y, p_z = df_ultra.iloc[index, 1], df_ultra.iloc[index, 2], df_ultra.iloc[index, 3]
        # print(p_x,p_y,p_z)
        return p_x, p_y, p_z
    '''
    merge the IMU, ultrasound and mag data into a single dataframe
    '''
    def merge_all(self):
        df_mag = self.get_mag()
        df_ultra = self.get_ultra()
        df_imu = self.get_imu()
        position = []
        imu_data = []
        for i in range(df_mag.shape[0]):
            position.append(self.get_position(df_mag.iloc[i, 3], df_ultra))
            imu_data.append(self.get_IMU_info(df_mag.iloc[i, 3], df_imu))
        position_array = np.array(position)
        imu_data_array = np.array(imu_data)
        df_mag['p_x'] = position_array[:, 0]
        df_mag['p_y'] = position_array[:, 1]
        df_mag['p_z'] = position_array[:, 2]
        df_mag['d_t1'] = np.sqrt(
            np.power(position_array[:, 0] - 1.90, 2) + np.power(position_array[:, 1] - 0.17, 2) + np.power(
                position_array[:, 2] - 1.24, 2))
        df_mag['d_t2'] = np.sqrt(
            np.power(position_array[:, 0] - 5.00, 2) + np.power(position_array[:, 1] - 0.33, 2) + np.power(
                position_array[:, 2] - 1.24, 2))
        df_mag['d_t3'] = np.sqrt(
            np.power(position_array[:, 0] - 3.23, 2) + np.power(position_array[:, 1] + 2.94, 2) + np.power(
                position_array[:, 2] - 1.24, 2))
        df_mag['a_x'] = imu_data_array[:, 0]
        df_mag['a_y'] = imu_data_array[:, 1]
        df_mag['a_z'] = imu_data_array[:, 2]
        df_mag['alpha'] = imu_data_array[:, 3]
        df_mag['beta'] = imu_data_array[:, 4]
        df_mag['gama'] = imu_data_array[:, 5]
        return df_mag
    '''
    get the start index of t1,t2,t3
    '''
    def get_T_index(self, noise_threshould):
        df_mag_merge = self.merge_all()
        mag_array = np.array(df_mag_merge['Magnitude'].tolist())
        mag_index = []
        for rows in range(10, df_mag_merge.shape[0] - 100):
            if np.sum(mag_array[rows - 1:rows + 4]) < np.sum(mag_array[rows:rows + 5]) and np.sum(
                    mag_array[rows + 1:rows + 6]) < np.sum(mag_array[rows:rows + 5]) and np.min(
                mag_array[rows:rows + 5]) > noise_threshould:  #### outdoor: 6800000
                mag_index.append(rows)
        T1_start = []
        T2_start = []
        T3_start = []

        for index in range(len(mag_index) - 3):
            if 7 < mag_index[index + 2] - mag_index[index + 1] < 11 and 7 < mag_index[index + 1] - mag_index[
                index] < 11:
                T1_start.append(mag_index[index])


            elif 10 < mag_index[index + 2] - mag_index[index + 1] < 14 and 7 < mag_index[index + 1] - mag_index[
                index] < 11:
                T2_start.append(mag_index[index])

            elif 7 < mag_index[index + 2] - mag_index[index + 1] < 11 and 10 < mag_index[index + 1] - mag_index[
                index] < 14:
                T3_start.append(mag_index[index])

        return T1_start, T2_start, T3_start

class phy_model:

    def __init__(self,filter_1,filter_2,filter_3):
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = filter_3

    def PolynomialRegression(self,degree):
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree)),
            ("std_scaler", StandardScaler()),
            ("lin_reg", LinearRegression())
        ])

    def get_reg(self,filter_data):
        reg = self.PolynomialRegression(degree=3)
        reg.fit(filter_data[:, 0].reshape(-1, 1), filter_data[:, 1])
        return reg

    def cal_dis(self,x, y, z):
        d_t1 = float(np.sqrt(np.power(x - 1.90, 2) + np.power(y - 0.17, 2) + np.power(z - 1.24, 2)))
        d_t2 = float(np.sqrt(np.power(x - 5.00, 2) + np.power(y - 0.33, 2) + np.power(z - 1.24, 2)))
        d_t3 = float(np.sqrt(np.power(x - 3.23, 2) + np.power(y + 2.94, 2) + np.power(z - 1.24, 2)))
        return d_t1, d_t2, d_t3


    def predict_phy(self,mag_x, mag_y, mag_z,step):
        reg_1 = self.get_reg(filter_t1)
        reg_2 = self.get_reg(filter_t2)
        reg_3 = self.get_reg(filter_t3)
        D_t1 = reg_1.predict([[mag_x]])
        D_t2 = reg_2.predict([[mag_y]])
        D_t3 = reg_3.predict([[mag_z]])
        x_min, x_max = 1.698, 5.709
        y_min, y_max = -2.957,0.553
        z_min, z_max =  0.736,1.10
        x_list = list(np.arange(x_min, x_max, step))
        y_list = list(np.arange(y_min, y_max, step))
        z_list = list(np.arange(z_min, z_max, step))
        result = []
        for x_i in x_list:
            for y_i in y_list:
                for z_i in z_list:
                    d_t1, d_t2, d_t3 = self.cal_dis(x_i, y_i, z_i)
                    if abs(d_t1 - D_t1) < 0.5 and abs(d_t2 - D_t2) < 0.5 and abs(d_t3 - D_t3) < 0.5:
                        sum_dis = abs(d_t1 - D_t1) + abs(d_t2 - D_t2) + abs(d_t3 - D_t3)
                        result.append((x_i, y_i, z_i, sum_dis))
        result = np.array(result)

        p_x = result[:, 0]
        p_y = result[:, 1]
        p_z = result[:, 2]
        distance = result[:, 3]
        data = {'x': p_x,
                'y': p_y,
                'z': p_z,
                'distance': distance
                }
        df_result = pd.DataFrame(data)
        df_result = df_result.sort_values(by="distance", ascending=True)

        return df_result.iloc[0, 0], df_result.iloc[0, 1], df_result.iloc[0, 2]

if __name__ == '__main__':

    model_1 = fileObject(root, 1)
    model_2 = fileObject(root, 2)
    model_3 = fileObject(root, 3)

    model1_t1, model1_t2, model1_t3 = model_1.get_T_index(8000000)
    model2_t1, model2_t2, model2_t3 = model_2.get_T_index(8000000)
    model3_t1, model3_t2, model3_t3 = model_3.get_T_index(8000000)

    df_merge_1 = model_1.merge_all()
    df_merge_2 = model_2.merge_all()
    df_merge_3 = model_3.merge_all()

    mag_1_T1 = [max(df_merge_1.iloc[x:x + 5, 4]) for x in model1_t1]
    dis_1_T1 = [df_merge_1.iloc[x, 8] for x in model1_t1]
    mag_2_T1 = [max(df_merge_2.iloc[x:x + 5, 4]) for x in model2_t1]
    dis_2_T1 = [df_merge_2.iloc[x, 8] for x in model2_t1]
    mag_3_T1 = [max(df_merge_3.iloc[x:x + 5, 4]) for x in model3_t1]
    dis_3_T1 = [df_merge_3.iloc[x, 8] for x in model3_t1]
    dis_1 = dis_1_T1 + dis_2_T1 + dis_3_T1
    mag_1 = mag_1_T1 + mag_2_T1 + mag_3_T1

    mag_1_T2 = [max(df_merge_1.iloc[x:x + 5, 4]) for x in model1_t2]
    dis_1_T2 = [df_merge_1.iloc[x, 9] for x in model1_t2]
    mag_2_T2 = [max(df_merge_2.iloc[x:x + 5, 4]) for x in model2_t2]
    dis_2_T2 = [df_merge_2.iloc[x, 9] for x in model2_t2]
    mag_3_T2 = [max(df_merge_3.iloc[x:x + 5, 4]) for x in model3_t2]
    dis_3_T2 = [df_merge_3.iloc[x, 9] for x in model3_t2]
    dis_2 = dis_1_T2 + dis_2_T2 + dis_3_T2
    mag_2 = mag_1_T2 + mag_2_T2 + mag_3_T2

    mag_1_T3 = [max(df_merge_1.iloc[x + 80:x + 85, 4]) for x in model1_t1]
    dis_1_T3 = [df_merge_1.iloc[x, 10] for x in model1_t1]
    mag_2_T3 = [max(df_merge_2.iloc[x + 80:x + 85, 4]) for x in model2_t1]
    dis_2_T3 = [df_merge_2.iloc[x, 10] for x in model2_t1]
    mag_3_T3 = [max(df_merge_3.iloc[x + 80:x + 85, 4]) for x in model3_t1]
    dis_3_T3 = [df_merge_3.iloc[x, 10] for x in model3_t1]
    dis_3 = dis_1_T3 + dis_2_T3 + dis_3_T3
    mag_3 = mag_1_T3 + mag_2_T3 + mag_3_T3


    '''
    part below is to filter out the noises in the data
    '''
    filter_t1 = []
    for i in range(len(mag_1)):
        if mag_1[i] < 8000000 or mag_1[i] > 15000000:
            continue
        elif mag_1[i] < 9000000 and dis_1[i] < 3.3:
            continue
        elif mag_1[i] > 10800000 and dis_1[i] > 2.5:
            continue
        else:
            filter_t1.append((mag_1[i], dis_1[i]))
    filter_t1 = np.array(filter_t1)

    filter_t2 = []
    for i in range(len(mag_2)):
        if mag_2[i] < 8500000 or mag_2[i] > 15000000:
            continue
        elif mag_2[i] > 10800000 and dis_2[i] > 2.3:
            continue
        else:
            filter_t2.append((mag_2[i], dis_2[i]))
    filter_t2 = np.array(filter_t2)

    filter_t3 = []
    for i in range(len(mag_3)):
        if mag_3[i] < 8500000 or mag_3[i] > 15000000:
            continue
        elif mag_3[i] > 10800000 and dis_3[i] > 2.2:
            continue
        elif mag_3[i] < 10500000 and dis_3[i] < 1.5:
            continue
        elif mag_3[i] < 9300000 and dis_3[i] < 2.3:
            continue
        else:
            filter_t3.append((mag_3[i], dis_3[i]))
    filter_t3 = np.array(filter_t3)

    phyModel = phy_model(filter_t1,filter_t2,filter_t3)
    phyModel.predict_phy()
