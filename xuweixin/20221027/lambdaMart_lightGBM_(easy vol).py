import numpy as np
import pandas as pd
import lightgbm

# 先生成下数据
df = pd.DataFrame({
    "query_id": [i for i in range(100) for j in range(10)],
    "var1": np.random.random(size=(1000,)),
    "var2": np.random.random(size=(1000,)),
    "var3": np.random.random(size=(1000,)),
    "relevance": list(np.random.permutation([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])) * 100
})

# 划分训练集和验证集
train_df = df[:800]  # first 80%
validation_df = df[800:]  # remaining 20%

qids_train = train_df.groupby("query_id")["query_id"].count().to_numpy()
X_train = train_df.drop(["query_id", "relevance"], axis=1)
y_train = train_df["relevance"]

qids_validation = validation_df.groupby("query_id")["query_id"].count().to_numpy()
X_validation = validation_df.drop(["query_id", "relevance"], axis=1)
y_validation = validation_df["relevance"]

model = lightgbm.LGBMRanker(
    objective="lambdarank",
    boosting_type='gbdt',
    metric="ndcg",
)

model.fit(
    X=X_train,
    y=y_train,
    group=qids_train,
    eval_set=[(X_validation, y_validation)],
    eval_group=[qids_validation],
    eval_at=10,
    verbose=10,
)


# 'task': 'train',  # 执行的任务类型
# 'boosting_type': 'gbrt',  # 基学习器
# 'objective': 'lambdarank',  # 排序任务(目标函数)
# 'metric': 'ndcg',  # 度量的指标(评估函数)
# 'max_position': 10,  # @NDCG 位置优化
# 'metric_freq': 1,  # 每隔多少次输出一次度量结果
# 'train_metric': True,  # 训练时就输出度量结果
# 'ndcg_at': [10],  # ndcg的@值
# 'max_bin': 255,  # 一个整数，表示最大的桶的数量。默认值为 255。lightgbm 会根据它来自动压缩内存。如max_bin=255 时，则lightgbm 将使用uint8 来表示特征的每一个值。
# 'num_iterations': 500,  # 迭代次数
# 'learning_rate': 0.01,  # 学习率
# 'num_leaves': 31,  # 叶子数
# # 'max_depth':6,
# 'tree_learner': 'serial',  # 用于并行学习，‘serial’： 单台机器的tree learner
# 'min_data_in_leaf': 30,  # 一个叶子节点上包含的最少样本数量
# 'verbose': 2  # 显示训练时的信息

# ndcg
def get_dcg(y_pred, y_true, k):
    # 注意y_pred与y_true必须是一一对应的，并且y_pred越大越接近label=1(用相关性的说法就是，与label=1越相关)
    df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})
    df = df.sort_values(by="y_pred", ascending=False)  # 对y_pred进行降序排列，越排在前面的，越接近label=1
    df = df.iloc[0:k, :]  # 取前K个
    dcg = (2 ** df["y_true"] - 1) / np.log2(np.arange(1, df["y_true"].count() + 1) + 1)  # 位置从1开始计数
    dcg = np.sum(dcg)
    return dcg


def get_ndcg(df, k):
    # df包含y_pred和y_true
    dcg = get_dcg(df["y_pred"], df["y_true"], k)
    idcg = get_dcg(df["y_true"], df["y_true"], k)
    ndcg = dcg / idcg
    return ndcg

sum_ndcg = 0
for i in range(0, 200, 10):
    y_pred = model.predict(X_validation[i:i + 10])
    predicted_sorted_indexes = np.argsort(y_pred)[::-1]  # 返回从大到小的索引
    predicted_sorted_indexes = predicted_sorted_indexes % 10
    y_pred=y_validation.values[i:i + 10][predicted_sorted_indexes]
    sum_ndcg += get_ndcg(pd.DataFrame({'y_pred': y_pred, 'y_true': sorted(y_pred,reverse=True)}), 10)

print(sum_ndcg/20)

# t_results = X_validation[predicted_sorted_indexes]  # 返回对应的comments,从大到小的排序
