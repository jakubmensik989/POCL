import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime, timedelta

# ================= 配置区 =================
DATA_OUT_DIR = "./Dataset/"
if not os.path.exists(DATA_OUT_DIR):
    os.makedirs(DATA_OUT_DIR)

SAMPLE_MEMBER_COUNT = 5000 

print("Step 1: 正在尝试读取数据 (自动处理编码)...")

# 🛠️ 修复点 1: 尝试多种编码格式，防止因中文导致的报错
def read_csv_safe(path):
    encodings = ['utf-8', 'gbk', 'gb18030', 'ISO-8859-1']
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, dtype={'ProviderID': str, 'Vendor': str})
            print(f"✅ 成功使用 {enc} 编码读取 {path}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"❌ 无法读取 {path}，请检查文件编码！")

try:
    claims = read_csv_safe('Claims_ovr.csv')
    targets = read_csv_safe('DaysInHospital_Y2.csv')
except FileNotFoundError:
    print("❌ 错误：找不到文件！请确认 Claims.csv 和 DaysInHospital_Y2.csv 就在当前文件夹里。")
    exit()

# 🛠️ 修复点 2: 清除列名里的空格，防止 KeyError
claims.columns = claims.columns.str.strip()
targets.columns = targets.columns.str.strip()
print("✅ 列名清洗完成")

# 采样逻辑
unique_members = claims['MemberID'].unique()
if len(unique_members) > SAMPLE_MEMBER_COUNT:
    print(f"⚠️ 数据采样: 取前 {SAMPLE_MEMBER_COUNT} 个病人...")
    sampled_members = unique_members[:SAMPLE_MEMBER_COUNT]
    claims = claims[claims['MemberID'].isin(sampled_members)]
    targets = targets[targets['MemberID'].isin(sampled_members)]

# 标签合并
targets['Label'] = (targets['DaysInHospital'] > 0).astype(int)
data = pd.merge(claims, targets[['MemberID', 'Label']], on='MemberID', how='left')
data['Label'] = data['Label'].fillna(0).astype(int)

# 🛠️ 修复点 3: 强壮的日期解析逻辑 (处理 '1月2日' 和 Excel 乱码)
print("Step 2: 正在解析日期...")
def robust_parse_date(row):
    # 年份
    y_str = str(row['Year']).strip()
    base_year = 2009
    if y_str == 'Y2': base_year = 2010
    elif y_str == 'Y3': base_year = 2011
    
    # 月份 DSFS
    dsfs = str(row['DSFS(Days Since First Service)'])
    month_offset = 0
    
    if 'month' in dsfs:
        try:
            # 处理 "8- 9 months"
            month_offset = int(dsfs.split('-')[0].strip())
        except:
            pass
    
    return datetime(base_year, 1, 1) + timedelta(days=month_offset*30)

data['ClaimStartDt'] = data.apply(robust_parse_date, axis=1)
data = data.sort_values('ClaimStartDt').reset_index(drop=True)

# 特征工程
print("Step 3: 正在处理特征...")
data['ProviderID'] = data['ProviderID'].fillna('Unknown')
feature_cols = ['Specialty', 'PlaceSvc', 'PrimaryConditionGroup', 'ProcedureGroup', 'CharlsonIndex', 'LengthOfStay']

for col in feature_cols:
    # 强制转为字符串，防止 '1月2日' 这种乱入的数据导致报错
    data[col] = data[col].fillna('Unknown').astype(str)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

scaler = MinMaxScaler()
feature_data = scaler.fit_transform(data[feature_cols].values)

# 建图
print("Step 4: 正在构建图结构...")
def create_edges_fast(df_subset):
    edges = []
    # 仅连接同一个病人的时序记录
    grp_member = df_subset.groupby('MemberID').indices
    for indices in grp_member.values():
        if len(indices) > 1:
            src = indices[:-1]
            dst = indices[1:]
            edges.extend(zip(src, dst))
            edges.extend(zip(dst, src))
            
    # 仅连接同一个医生的相邻记录 (限制数量防止卡死)
    grp_provider = df_subset.groupby('ProviderID').indices
    for pid, indices in grp_provider.items():
        if pid == 'Unknown': continue
        if len(indices) > 1:
            limit_indices = indices[:20] 
            src = limit_indices[:-1]
            dst = limit_indices[1:]
            edges.extend(zip(src, dst))
            edges.extend(zip(dst, src))

    if not edges: return np.array([[], []])
    return np.array(list(set(edges))).T

# 输出
print("Step 5: 正在保存...")
data['YearMonth'] = data['ClaimStartDt'].apply(lambda x: x.strftime('%Y-%m'))
time_steps = sorted(data['YearMonth'].unique())

dataset_online = []
dataset_two = []

for t_step in time_steps:
    step_df = data[data['YearMonth'] == t_step]
    if step_df.empty: continue
    
    curr_feats = feature_data[step_df.index]
    curr_labels = step_df['Label'].values.reshape(-1, 1)
    combined_matrix = np.hstack([curr_feats, curr_labels])
    
    edge_index = create_edges_fast(step_df.reset_index(drop=True))
    dataset_online.append((combined_matrix, edge_index))
    dataset_two.append((combined_matrix, edge_index))

with open(DATA_OUT_DIR + "datasetonline.dat", "wb") as f:
    pickle.dump(dataset_online, f)
with open(DATA_OUT_DIR + "datasettwo.dat", "wb") as f:
    pickle.dump(dataset_two, f)

print("✅ 成功跑通！")

# ==========================================
# 新增功能：导出处理后的数据为 CSV
# ==========================================
print("Step 6: 正在导出可视化 CSV 文件...")

# 1. 导出【节点特征表】 (Node Features)
# 这张表包含了模型真正使用的所有数据：对齐后的日期、数字化的特征、标签
export_df = data.copy()
# 只保留核心列和特征列
save_cols = ['MemberID', 'ProviderID', 'ClaimStartDt', 'Label'] + feature_cols
# 保存
export_df[save_cols].to_csv(DATA_OUT_DIR + 'processed_node_features.csv', index=False)
print(f"✅ 节点特征表已保存: {DATA_OUT_DIR}processed_node_features.csv")

# 2. 导出【边列表】 (Edge List)
# 只导出前 10000 条边
# (注意：因为之前的 dataset_online 是分时间步存的，我们这里重新生成一个全量的边列表用于展示)
print("正在生成边列表 CSV (仅取前 10000 条示例)...")
all_edges = create_edges_fast(data.head(5000))  # 对前 5000 行数据建图作为示例
if all_edges.shape[1] > 0:
    edge_df = pd.DataFrame(all_edges.T, columns=['Source_Node_Index', 'Target_Node_Index'])
    # 映射回真实的 MemberID (可选，方便理解)
    # 注意：这里的 Index 是相对于 head(5000) 的索引
    edge_df['Source_MemberID'] = data.iloc[edge_df['Source_Node_Index']]['MemberID'].values
    edge_df['Target_MemberID'] = data.iloc[edge_df['Target_Node_Index']]['MemberID'].values

    edge_df.to_csv(DATA_OUT_DIR + 'processed_edge_list.csv', index=False)
    print(f"✅ 边列表已保存: {DATA_OUT_DIR}processed_edge_list.csv")
else:
    print("⚠️ 警告：前 5000 条数据没有生成任何边，可能数据太稀疏。")

print("=" * 30)