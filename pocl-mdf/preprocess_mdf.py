import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime, timedelta

# ================= 配置区 =================
# 输出文件夹
DATA_OUT_DIR = "./Dataset/"
if not os.path.exists(DATA_OUT_DIR):
    os.makedirs(DATA_OUT_DIR)

print("正在读取数据...")
# 1. 读取数据
claims = pd.read_csv('Claims_ovr.csv')
# 使用 Y2 的住院天数作为“标签” (Label source)
# 注意：实际场景中应该对应年份，这里为了跑通代码，我们将 Y2 的标签通过 MemberID 广播给该人的所有记录
targets = pd.read_csv('DaysInHospital_Y2.csv')

# ================= 数据清洗与合成 =================

# 2. 构造标签 (Label Construction)
# 假设：如果住院天数 > 0，我们将其标记为“异常/欺诈” (1)，否则为 0
targets['Label'] = (targets['DaysInHospital'] > 0).astype(int)
# 将标签合并到 Claims 表中
data = pd.merge(claims, targets[['MemberID', 'Label']], on='MemberID', how='left')
# 填充没有标签的记录为 0
data['Label'] = data['Label'].fillna(0).astype(int)

# 3. 合成日期 (Date Synthesis)
# POCL 极其依赖日期排序。我们需要把 "Y1", "0- 1 month" 转换成 "2009-01-01" 这种格式
print("正在合成日期...")
def parse_dsfs(year_str, dsfs_str):
    # 基准年份
    base_year = 2009
    if year_str == 'Y2': base_year = 2010
    elif year_str == 'Y3': base_year = 2011
    
    # 解析月份偏移
    month_offset = 0
    if pd.isna(dsfs_str):
        month_offset = 0
    elif '0- 1' in dsfs_str: month_offset = 0
    elif '1- 2' in dsfs_str: month_offset = 1
    elif '2- 3' in dsfs_str: month_offset = 2
    elif '3- 4' in dsfs_str: month_offset = 3
    elif '4- 5' in dsfs_str: month_offset = 4
    elif '5- 6' in dsfs_str: month_offset = 5
    elif '6- 7' in dsfs_str: month_offset = 6
    elif '7- 8' in dsfs_str: month_offset = 7
    elif '8- 9' in dsfs_str: month_offset = 8
    elif '9-10' in dsfs_str: month_offset = 9
    elif '10-11' in dsfs_str: month_offset = 10
    elif '11-12' in dsfs_str: month_offset = 11
    
    # 构造日期 (默认为该月1号)
    # 为了防止同一天数据过多，我们可以加一个随机天数，或者简单处理
    dt = datetime(base_year, 1, 1) + timedelta(days=month_offset*30)
    return dt

data['ClaimStartDt'] = data.apply(lambda x: parse_dsfs(x['Year'], x['DSFS(Days Since First Service)']), axis=1)
# 按日期排序 (非常重要，POCL是时序模型)
data = data.sort_values('ClaimStartDt').reset_index(drop=True)

# 4. 特征工程 (Feature Engineering)
print("正在处理特征...")
# 选定要使用的特征列
feature_cols = ['Specialty', 'PlaceSvc', 'PrimaryConditionGroup', 'ProcedureGroup', 'CharlsonIndex', 'LengthOfStay']
# 填充缺失值
for col in feature_cols:
    data[col] = data[col].fillna('Unknown').astype(str)

# 将字符串特征转换为数字 (Label Encoding)
label_encoders = {}
for col in feature_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 归一化特征
scaler = MinMaxScaler()
feature_data = scaler.fit_transform(data[feature_cols].values)

# ================= 构建图结构 =================
print("正在构建图网络 (这可能需要几分钟)...")

# 辅助函数：构建边
def create_edges(df_subset):
    edges = []
    # 逻辑：如果两个记录属于同一个 Member (BeneID) 或同一个 Provider，则连线
    # 1. 相同 Member
    grp_member = df_subset.groupby('MemberID').indices
    for member_idx in grp_member.values():
        if len(member_idx) > 1:
            # 全连接过于稠密，这里简化为时序连线 (i -> i+1) 以节省内存
            # 或者像原代码一样构建全连接（这里采用简化版防止内存爆炸）
            idx_list = sorted(member_idx)
            for k in range(len(idx_list)-1):
                edges.append((idx_list[k], idx_list[k+1]))
                edges.append((idx_list[k+1], idx_list[k])) # 无向图

    # 2. 相同 Provider
    grp_provider = df_subset.groupby('ProviderID').indices
    for prov_idx in grp_provider.values():
        if len(prov_idx) > 1:
             # 限制：仅连接时间相近的 (例如30天内)，这里为简化直接连前5个
             idx_list = sorted(prov_idx)
             limit = 5 
             for k in range(len(idx_list)):
                 for m in range(k+1, min(k+limit, len(idx_list))):
                     edges.append((idx_list[k], idx_list[m]))
                     edges.append((idx_list[m], idx_list[k]))

    if len(edges) == 0:
        return np.array([[], []])
    return np.array(list(set(edges))).T

# ================= 数据集切分与保存 =================

# 按月份切分数据 (模拟时间步)
# 将 data 分组
data['YearMonth'] = data['ClaimStartDt'].apply(lambda x: x.strftime('%Y-%m'))
time_steps = sorted(data['YearMonth'].unique())

dataset_online = [] # 用于在线训练
dataset_two = []    # 用于对比学习 (预训练)

print(f"共发现 {len(time_steps)} 个时间步。正在生成 Dataset...")

for t_step in time_steps:
    step_df = data[data['YearMonth'] == t_step]
    
    if step_df.empty: continue
    
    # 1. 获取特征矩阵
    # 这一步非常关键：我们要把 Label 拼接到特征的 LAST COLUMN (最后一列)
    # 因为 tool.py 里是写死的 y = element[:, 5] (或者我们需要修改tool.py)
    # 为了稳妥，我们把 Label 放在最后，并要求修改 tool.py
    
    # 当前步的特征
    curr_feats = feature_data[step_df.index]
    curr_labels = step_df['Label'].values.reshape(-1, 1)
    
    # 拼接: [Feature1, Feature2, ..., Label]
    combined_matrix = np.hstack([curr_feats, curr_labels])
    
    # 2. 构建边
    edge_index = create_edges(step_df.reset_index(drop=True))
    
    # 3. 存入 dataset_online
    dataset_online.append((combined_matrix, edge_index))
    
    # 4. 存入 dataset_two (区分正负样本)
    # 原代码逻辑：datasettwo 包含 [Normal_Graph, Anomaly_Graph]
    # 我们按 Label 切分
    mask_normal = (step_df['Label'] == 0).values
    mask_anomaly = (step_df['Label'] == 1).values
    
    # 注意：如果某个时间步全是正常或全是异常，会导致 crash，需要处理
    if np.sum(mask_normal) > 0:
        feats_norm = combined_matrix[mask_normal]
        # 重新构建子图边 (这里简化，只取特征，边置空或者重新计算)
        # 为跑通代码，这里简单置空边，或者你可以对子集调 create_edges
        dataset_two.append((feats_norm, np.array([[],[]]))) 
    else:
        # 填充空数据防止报错
        dataset_two.append((np.zeros((1, combined_matrix.shape[1])), np.array([[],[]])))

    if np.sum(mask_anomaly) > 0:
        feats_anom = combined_matrix[mask_anomaly]
        dataset_two.append((feats_anom, np.array([[],[]])))
    else:
        dataset_two.append((np.zeros((1, combined_matrix.shape[1])), np.array([[],[]])))

# 保存
print("正在保存文件...")
with open(DATA_OUT_DIR + "datasetonline.dat", "wb") as f:
    pickle.dump(dataset_online, f)
    
with open(DATA_OUT_DIR + "datasettwo.dat", "wb") as f:
    pickle.dump(dataset_two, f)

print("="*30)
print("✅ 数据预处理完成！")
print(f"新生成的特征维度 (Feature Dim) 是: {len(feature_cols)}")
print("请务必执行后续的‘代码修改’步骤，否则 main.py 会报错！")
print("="*30)