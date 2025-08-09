import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import *

def process_data(data,lag_tm,y_column):
    '''
    data:
    lag_tm
    y_column: target name
    '''
    stat_st_idx = 0
    stat_ed_idx = len(data)-1
    X = []
    y = []
    st_idx = 0
    cols = data.columns
    length = len(data.columns) 
    while st_idx <= stat_ed_idx-lag_tm:
        X_s = [data['current_date'].iloc[st_idx+lag_tm]]
        y_s = [data['current_date'].iloc[st_idx+lag_tm]]
        for name in cols:
            X_s.extend(data[name].iloc[st_idx:st_idx+lag_tm].tolist())
        y_s.extend([data[y_column].iloc[st_idx+lag_tm]])
        X.append(X_s)
        y.append(y_s)
        st_idx += 1
    new_col1 = []
    new_col1.append('current_date')
    for i in range(length):
        for j in range(lag_tm):
            n_name = f'{cols[i]}_{j}'
            new_col1.append(n_name)
    X = pd.DataFrame(X,columns=new_col1)
    y = pd.DataFrame(y,columns=['current_date',y_column])
    return X,y
def analyze_feature_selection(res, feature_names, output_csv=None):
    """
    分析多目标优化过程中特征被选择的频率
    
    参数:
    res -- 优化结果对象 (包含历史记录 history)
    feature_names -- 特征名称列表 (顺序与特征索引一致)
    output_csv -- 输出CSV文件路径 (可选)
    
    返回:
    df -- 包含特征名称和选择次数的DataFrame (按次数降序排列)
    """
    # 初始化特征选择计数列表
    feature_selection_count = [0] * len(feature_names)
    
    # 遍历每一代种群
    for gen, entry in enumerate(res.history):
        # 遍历每个解决方案
        for solution in entry.pop:
            # 获取选择的特征索引
            selected_features = [i for i, selected in enumerate(solution.X) if selected]
            
            # 更新特征选择计数
            for feature_idx in selected_features:
                if feature_idx < len(feature_selection_count):
                    feature_selection_count[feature_idx] += 1
    
    # 创建特征名称与选择次数的字典
    feature_dict = dict(zip(feature_names, feature_selection_count))
    
    # 创建DataFrame并排序
    df = pd.DataFrame({
        'Feature': feature_names,
        'Count': feature_selection_count
    })
    
    # 按选择次数降序排序
    df = df.sort_values('Count', ascending=False).reset_index(drop=True)
    
    # 如果需要，保存到CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"特征选择计数已保存至: {output_csv}")
    
    return df
def decise_score(feature_df, X_train_a, X_test_a, y_train_a, y_test_a):
    '''
        feature_df: selected features
        X_train_a:
    '''
    valid_indices = []        # Stores original indices of valid rows
    valid_selections = []     # Stores feature masks for valid rows
    rmse_all = []
    mae_all = []
    mape_all = []
    len_features = []
    

    
    # First pass: evaluate all valid feature sets
    for i in range(len(feature_df)):
        features = feature_df.iloc[i, :]
        selected_mask = features.values.astype(bool)
        selected_features = np.where(selected_mask)[0]
        
        if len(selected_features) > 0:
            valid_indices.append(i)
            valid_selections.append(selected_mask)
            len_features.append(len(selected_features))
            
            X_train_select = X_train_a[:, selected_features]
            X_test_select = X_test_a[:, selected_features]
            
            clf = RandomForestRegressor(random_state=42)
            clf.fit(X_train_select, y_train_a.ravel())
            y_pred_select = clf.predict(X_test_select)
            
            rmse_select = sqrt(mean_squared_error(y_test_a, y_pred_select))
            mae = np.mean(np.abs(y_test_a - y_pred_select))
            mape = np.mean(np.abs((y_test_a - y_pred_select) / y_test_a)) * 100
            
            rmse_all.append(rmse_select)
            mae_all.append(mae)
            mape_all.append(mape)

    # Handle results if we have at least one valid feature set
    if len(valid_selections) >= 1:
        # Rank feature sets by each metric
        sorted_indices_1 = np.argsort(rmse_all)
        sorted_ranks_1 = np.argsort(sorted_indices_1)
        
        sorted_indices_2 = np.argsort(mae_all)
        sorted_ranks_2 = np.argsort(sorted_indices_2)
        
        sorted_indices_3 = np.argsort(mape_all)
        sorted_ranks_3 = np.argsort(sorted_indices_3)
        
        # Combine ranks to find best overall feature set
        combined_ranks = [
            r1 + r2 + r3 
            for r1, r2, r3 in zip(sorted_ranks_1, sorted_ranks_2, sorted_ranks_3)
        ]
        best_idx = np.argmin(combined_ranks)
        
        # Retrieve stored feature mask (avoids recomputation)
        selected_mask = valid_selections[best_idx]
        selected_features = np.where(selected_mask)[0]
        
        # Final training and evaluation
        X_train_select = X_train_a[:, selected_features]
        X_test_select = X_test_a[:, selected_features]
        
        clf = RandomForestRegressor(random_state=42)
        clf.fit(X_train_select, y_train_a.ravel())
        y_pred_select = clf.predict(X_test_select)
        
        # Calculate all metrics
        rmse_select = sqrt(mean_squared_error(y_test_a, y_pred_select))
        mae = np.mean(np.abs(y_test_a - y_pred_select))
        mape = np.mean(np.abs((y_test_a - y_pred_select) / y_test_a)) * 100
        
        print('Metrics: rmse, mae, mape')
        print(rmse_select, mae, mape)
        
        return selected_features 
        
    else:
        print("No valid feature sets found.")
        return None

    