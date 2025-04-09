import dask.dataframe as dd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

start_time = time.time()
# 1. 分块读取数据
df = dd.read_parquet("30G_data/*.parquet", chunksize="256MB")

# 2. 数据概览（分块计算）
print("DataFrame 结构:")
print(df.dtypes)

# 获取前5行（不需要计算整个DataFrame）
with pd.option_context('display.max_columns', None,  # 显示所有列
                       'display.max_colwidth', 100,  # 设置列宽（避免长文本被截断）
                       'display.width', 1000):       # 控制台输出宽度
    print(df.head())

with pd.option_context('display.max_colwidth', 1000):
    print(df['purchase_history'].head())
# 计算总行数（分块计算）
total_rows = df.shape[0].compute()
print("\n总行数:", total_rows)

# 缺失值统计（分块计算）
missing = df.isnull().sum().compute()
print("缺失值统计:")
print(missing[missing > 0])

# 计算重复行数
duplicate_count = df.shape[0].compute() - df.drop_duplicates().shape[0].compute()
print(f"重复行数: {duplicate_count}")
# 实际去重操作（惰性操作）
df = df.drop_duplicates(keep="first")

# 年龄异常
age_outliers = df[(df["age"] < 18) | (df["age"] > 100)].shape[0].compute()
print(f"年龄异常记录数: {age_outliers}")

# 收入异常值（箱线图法）
Q1 = df["income"].quantile(0.25).compute()
Q3 = df["income"].quantile(0.75).compute()
IQR = Q3 - Q1
income_outliers = df[
    (df["income"] < (Q1 - 1.5*IQR)) |
    (df["income"] > (Q3 + 1.5*IQR))
].shape[0].compute()
print(f"收入异常记录数: {income_outliers}")

# 性别字段异常值
invalid_genders = df[~df["gender"].isin(["男", "女"])].shape[0].compute()
print(f"性别异常记录数: {invalid_genders}")
df["gender"] = df["gender"].where(
    df["gender"].isin(["男", "女"]),
    other="未指定"
)

# 邮箱格式验证（简单正则）
email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
email_invalid = df[~df["email"].str.match(email_pattern)].shape[0].compute()
print(f"邮箱格式异常记录数: {email_invalid}")

# 转换日期字段并检查范围（惰性操作）
df["registration_date"] = dd.to_datetime(df["registration_date"])
date_outliers = df[
    (df["registration_date"] < "2000-01-01") |
    (df["registration_date"] > "2025-03-29")
].shape[0].compute()
print(f"注册时间异常记录数: {date_outliers}")

def clean_timestamp(series):
    return pd.to_datetime(series, errors='coerce').dt.tz_localize(None)
meta = pd.Series([], dtype='datetime64[ns]', name='timestamp')
df['timestamp'] = df['timestamp'].map_partitions(clean_timestamp, meta=meta)
invalid_count = df['timestamp'].isna().sum().compute()
print(f"发现 {invalid_count} 个无效日期")

# 年龄分段分析
def add_age_group(pdf):
    pdf['age_group'] = pd.cut(
        pdf['age'],
        bins=[18, 25, 35, 45, 55, 65, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    )
    return pdf
df = df.map_partitions(add_age_group)
age_dist = df['age_group'].value_counts().compute()
plt.figure(figsize=(10, 6))
sns.barplot(x=age_dist.index, y=age_dist.values)
plt.title('年龄分布')
plt.savefig('age_distribution.png')
plt.close('all')

# 年龄与收入关系
g=sns.jointplot(data=df.sample(frac=0.1).compute(),
             x='age', y='income', kind='hex')
g.fig.set_size_inches(10,11)
plt.suptitle('年龄-收入分布热力图')
plt.savefig('age_income_heatmap.png')
plt.close('all')

# 性别比例
country_gender = df.groupby(['country', 'gender']).size().compute().unstack()
country_gender_pct = country_gender.div(country_gender.sum(axis=1), axis=0) * 100
plt.figure(figsize=(12, 8))
sns.heatmap(country_gender_pct.sort_values('男', ascending=False).head(20),
           annot=True, fmt='.1f', cmap='Blues')
plt.title('Top20国家性别比例热力图(%)')
plt.savefig('gender_country_heatmap.png')
plt.close('all')

# 解析JSON字段（示例）
def parse_purchase(record):
    try:
        data = json.loads(record)
        return (
            len(data['items']),
            data['average_price'] * len(data['items']),
            data['category'],
            data['average_price']
        )
    except:
        return (0, 0.0, None, 0.0)  # 提供默认值

def process_chunk(chunk):
    stats = chunk['purchase_history'].apply(parse_purchase)
    return pd.DataFrame(
        stats.tolist(),
        columns=['purchase_count', 'total_spent', 'fav_category', 'average_price'],
        index=chunk.index
    )

purchase_stats = df.map_partitions(
    process_chunk,
    meta={
        'purchase_count': 'int64',
        'total_spent': 'float64',
        'fav_category': 'object',
        'average_price': 'float64'
    }
)
df = df.assign(
    purchase_count=purchase_stats['purchase_count'],
    total_spent=purchase_stats['total_spent'],
    fav_category=purchase_stats['fav_category'],
    average_price=purchase_stats['average_price']
)
# 按年龄组分析（不分性别）
age_counts = (df.groupby(['age_group', 'fav_category'])
              .size()
              .compute())
age_totals = df.groupby('age_group').size().compute()
age_normalized = (age_counts / age_totals * 100).unstack().fillna(0)
plt.figure(figsize=(14, 6))
sns.heatmap(age_normalized,
           annot=True, fmt='.1f',
           cmap='Blues',
           linewidths=.5,
           cbar_kws={'label': '占比(%)'})
plt.title('各年龄组的商品类别偏好(%)', pad=20)
plt.xlabel('商品类别')
plt.ylabel('年龄组')
plt.tight_layout()
plt.savefig('age_category_preference.png', dpi=300, bbox_inches='tight')
plt.close('all')

# 按性别分析（不分年龄）
gender_counts = (df.groupby(['gender', 'fav_category'])
                .size()
                .compute())
gender_totals = df.groupby('gender').size().compute()
gender_normalized = (gender_counts / gender_totals * 100).unstack().fillna(0)
plt.figure(figsize=(12, 4))
sns.heatmap(gender_normalized,
           annot=True, fmt='.1f',
           cmap='Greens',
           linewidths=.5,
           cbar_kws={'label': '占比(%)'})
plt.title('不同性别的商品类别偏好(%)', pad=20)
plt.xlabel('商品类别')
plt.ylabel('性别')
plt.tight_layout()
plt.savefig('gender_category_preference.png', dpi=300, bbox_inches='tight')
plt.close('all')

# 信用分 vs 收入 vs 年龄 计算相关系数矩阵
corr_matrix = df[['credit_score', 'income', 'age', 'purchase_count', 'total_spent']].corr().compute()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
ax =sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm')
ax.figure.set_size_inches(10, 10)
plt.title('信用评分影响因素相关性')
plt.savefig('credit_correlation.png')
plt.close('all')

#计算RFM指标（需registration_date转为日期类型）
now = pd.to_datetime('now').tz_localize(None)
df['recency'] = (now - df['timestamp']).dt.days
df['recency'] = df['recency'].fillna(-1)  # 对无效日期填充-1
rfm = df.groupby('id').agg({
    'recency': 'min',
    'total_spent': 'max',
    'purchase_count': 'max'
}).compute()
# K-means聚类分群
kmeans = KMeans(n_clusters=4)
rfm['cluster'] = kmeans.fit_predict(rfm)

cluster_avg_spent = rfm.groupby('cluster')['total_spent'].mean().sort_values(ascending=False)
sorted_clusters = cluster_avg_spent.index.tolist()  # 此时索引0对应消费最高的簇
palette = sns.color_palette("Purples_r", n_colors=len(sorted_clusters))
cluster_color_map = {
    cluster: palette[i]
    for i, cluster in enumerate(sorted_clusters)  # 确保索引0（最高消费）取最深的Purples_r[0]
}
sns.set_palette([cluster_color_map[c] for c in sorted_clusters])
g = sns.pairplot(
    rfm,
    hue='cluster',
    hue_order=sorted_clusters,  # 强制按消费降序排列
    palette=cluster_color_map   # 显式传递颜色映射
)
plt.suptitle('RFM用户分群')
plt.savefig('rfm_clusters.png')
# # 可视化分群
# sns.pairplot(rfm, hue='cluster')


end_time = time.time()
total_time = end_time - start_time
print(f"\n全部处理完成，总耗时: {total_time:.2f}秒")