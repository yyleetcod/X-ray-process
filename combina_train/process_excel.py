from collections import Counter
import xlrd
import pandas as pd

data = xlrd.open_workbook('label.xls')
table =  data.sheets()[0]
index = table.col_values(0)
index.pop(0)

sex_map = {'女': 0, '男': 1}
sex = table.col_values(6)
sex.pop(0)
for index, cur_sex in enumerate(sex):
    sex[index] = sex_map[cur_sex]

height = table.col_values(10)
height.pop(0)

weight = table.col_values(11)
weight.pop(0)

thigh_map = {1.5: 0, 2.0: 1, 2.5: 2, 3.0: 3, 3.5: 4, 4.0: 5, 5.0: 6, 6.0: 7}
thigh_bone = table.col_values(12)
thigh_bone.pop(0)
for index, thigh_bone_size in enumerate(thigh_bone):
    thigh_bone[index] = thigh_map[thigh_bone_size]

shin_map = {'B': 0, 'B+': 1, 'C': 2, 'C+': 3, 'D': 4, 'D+': 5, 'E': 6, 'F': 7, 'G': 8}
shin_bone = table.col_values(13)
shin_bone.pop(0)
for index, shin_bone_size in enumerate(shin_bone):
    shin_bone[index] = shin_map[shin_bone_size]

direction_map = {'左': 0, '右': 1}
direction = table.col_values(9)
direction.pop(0)
for index, cur_direction in enumerate(direction):
    direction[index] = direction_map[cur_direction[0]]

feature_map = {'sex': sex, 'height': height, 'weight': weight, 'direction': direction, 'thigh_bone': thigh_bone, 'shin_bone': shin_bone}
feature_dataframe = pd.DataFrame.from_dict(feature_map)
feature_dataframe['height'] = feature_dataframe['height'] / 180
feature_dataframe['weight'] = feature_dataframe['weight'] / 100

feature_dataframe.to_csv('./label.csv', index = False)
print (feature_dataframe.describe())
print (Counter(thigh_bone))
print (Counter(shin_bone))
print (Counter(direction))
