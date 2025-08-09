from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from sklearn.model_selection import KFold

def run_feature_selection_optimization(X_train_a, y_train_a,pop_size = 100,cross = 0.9,kfold = 2):
    """
    使用NSGA-II多目标优化算法进行特征选择
    
    参数:
    X_train_a -- (n_samples, n_features)
    y_train_a --  (n_samples,)
    k: kfold
    
    返回:
    res -- results
    """
    # 定义适应度函数
    def evaluate(ind):
        selected_features = np.where(ind)[0]
        
        # 如果没有选择任何特征，返回极大误差和零特征
        if len(selected_features) == 0:
            return [1e10, 0]
        
        # 构建特征子集
        X_subset_train = X_train_a[:, selected_features]
        
        # 使用十折交叉验证
        kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
        rmse_scores = []

        for train_index, val_index in kf.split(X_subset_train):
            X_train_kf, X_val_kf = X_subset_train[train_index], X_subset_train[val_index]
            y_train_kf, y_val_kf = y_train_a[train_index], y_train_a[val_index]

            clf = RandomForestRegressor(random_state=42)
            clf.fit(X_train_kf, y_train_kf.ravel())
            y_pred_kf = clf.predict(X_val_kf)

            rmse_fold = sqrt(mean_squared_error(y_val_kf, y_pred_kf))
            rmse_scores.append(rmse_fold)

        # 计算平均RMSE
        mean_rmse = np.mean(rmse_scores)
        
        # 返回适应度值 [RMSE, 特征数量]
        return [mean_rmse, len(selected_features)]

    # 定义优化问题
    class FeatureSelectionProblem(ElementwiseProblem):
        def __init__(self):
            n_features = X_train_a.shape[1]
            super().__init__(
                n_var=n_features,
                n_obj=2,
                n_ieq_constr=0,
                xl=np.zeros(n_features),
                xu=np.ones(n_features),
                elementwise_evaluation=True
            )

        def _evaluate(self, x, out, *args, **kwargs):
            f1, f2 = evaluate(x)
            out["F"] = [f1, f2]

    # 初始化问题
    problem = FeatureSelectionProblem()
    
    # 配置NSGA-II算法
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=pop_size,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob = cross),
        mutation=BitflipMutation(),
        eliminate_duplicates=True
    )
    
    # 设置终止条件（3代）
    termination = get_termination("n_gen", 3)
    
    # 运行优化
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=True,
        verbose=True
    )
    
    return res

# 使用示例:
# 假设 X_train, y_train 是准备好的数据
# result = run_feature_selection_optimization(X_train, y_train)