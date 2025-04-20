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
import dask.array as da

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

start_time = time.time()
# 1. 分块读取数据
df = dd.read_parquet("10G_data_new/*.parquet", chunksize="500MB")

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
with pd.option_context('display.max_colwidth', 1000):
    print(df['login_history'].head())
# 计算总行数（分块计算）
total_rows = df.shape[0].compute()
print("\n总行数:", total_rows)

# 缺失值统计（分块计算）
missing = df.isnull().sum().compute()
print("缺失值统计:")
print(missing[missing > 0])

# 计算重复行数
id_counts = df['id'].value_counts().compute()
duplicate_ids = id_counts[id_counts > 1].index.tolist()

# 2. 分离重复和非重复数据
non_duplicate_df = df[~df['id'].isin(duplicate_ids)]
potential_duplicate_df = df[df['id'].isin(duplicate_ids)]

# 3. 定义去重函数
def deduplicate_group(group):
    """处理每个id分组，保留非重复或首个完全重复的行"""
    cols = [c for c in group.columns if c != 'id']
    if len(group.drop_duplicates(subset=cols)) == 1:
        return group.head(1)
    return group

# 4. 对可能重复的数据进行分组去重
deduplicated = (
    potential_duplicate_df
    .groupby('id')
    .apply(deduplicate_group, meta=potential_duplicate_df._meta)
    .compute()  # 显式计算
)

# 5. 合并最终数据
final_df = dd.concat([non_duplicate_df, deduplicated], interleave_partitions=True)

# 6. 统计重复行数
original_count = len(df)
final_count = len(final_df)
duplicate_count = original_count - final_count
print(f"总重复行数: {duplicate_count}")

# 年龄异常
filtered = df[(df["age"] < 18) | (df["age"] > 100)]
age_outliers = filtered.shape[0].compute()
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
meta = pd.Series([], dtype='datetime64[ns]', name='last_login')
df['last_login'] = df['last_login'].map_partitions(clean_timestamp, meta=meta)
invalid_count = df['last_login'].isna().sum().compute()
print(f"发现 {invalid_count} 个无效日期")

# 年龄分段分析
def add_age_group(pdf):
    age_bins = [18, 25, 35, 45, 55, 65, 101]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    pdf['age_group'] = pd.cut(
        pdf['age'],
        bins=age_bins,
        labels=age_labels,
        ordered=True
    ).astype(pd.CategoricalDtype(categories=age_labels, ordered=True))
    return pdf
df = df.map_partitions(add_age_group)
age_dist = df['age_group'].value_counts().compute()
plt.figure(figsize=(10, 6))
sns.barplot(x=age_dist.index, y=age_dist.values)
plt.title('age_distribution')
plt.savefig('age_distribution.png')
plt.close('all')

# 年龄与收入关系
g=sns.jointplot(data=df.sample(frac=0.001).compute(),
             x='age', y='income', kind='hex')
g.fig.set_size_inches(10,11)
plt.suptitle('age_income_heatmap')
plt.savefig('age_income_heatmap.png')
plt.close('all')

def parse_purchase(record):
    try:
        data = json.loads(record)
        return (
            len(data['items']),
            data['avg_price'] * len(data['items']),
            data['categories'],
            data['avg_price'],
            data['purchase_date']
        )
    except:
        return (0, 0.0, None, 0.0)  # 提供默认值

def parse_login(record):
    try:
        data = json.loads(record)
        return (
            data['avg_session_duration'],
            data['login_count']
        )
    except:
        return (0.0, 0)  # 默认值

def process_chunk(chunk):
    # 处理购买历史
    purchase_stats = chunk['purchase_history'].apply(parse_purchase)
    purchase_df = pd.DataFrame(
        purchase_stats.tolist(),
        columns=['purchase_count', 'total_spent', 'fav_category', 'average_price', 'purchase_date'],
        index=chunk.index
    )
    # 处理登录历史
    login_stats = chunk['login_history'].apply(parse_login)
    login_df = pd.DataFrame(
        login_stats.tolist(),
        columns=['avg_session_duration', 'login_count'],
        index=chunk.index
    )
    # 合并结果
    return pd.concat([purchase_df, login_df], axis=1)

purchase_stats = df.map_partitions(
    process_chunk,
    meta={
        'purchase_count': 'int64',
        'total_spent': 'float64',
        'fav_category': 'object',
        'average_price': 'float64',
        'purchase_date': 'datetime64[ns]',
        'avg_session_duration': 'float64',
        'login_count': 'int64'
    }
)

df = df.assign(
    purchase_count=purchase_stats['purchase_count'],
    total_spent=purchase_stats['total_spent'],
    fav_category=purchase_stats['fav_category'],
    average_price=purchase_stats['average_price'],
    purchase_date=purchase_stats['purchase_date'],
    avg_session_duration=purchase_stats['avg_session_duration'],
    login_count=purchase_stats['login_count']
)
# 按年龄组分析（不分性别）
all_age_groups = df['age_group'].cat.categories.tolist()  # 直接从分类类型获取
all_categories = df['fav_category'].unique().compute().tolist()
meta = pd.DataFrame(
    columns=all_categories,
    index=pd.CategoricalIndex(all_age_groups, ordered=True),  # 有序索引
    dtype='float64'
)
def calculate_percentage(partition):
    counts = partition.groupby(['age_group', 'fav_category']).size()
    totals = partition.groupby('age_group').size()
    result = (counts / totals * 100).unstack().fillna(0)
    return result.reindex(columns=all_categories, index=all_age_groups, fill_value=0)
age_normalized = (
    df.map_partitions(calculate_percentage, meta=meta)
    .compute()
    .groupby(level=0).mean()  # 合并分区结果
    .reindex(index=all_age_groups, columns=all_categories, fill_value=0)  # 最终修正顺序
)
plt.figure(figsize=(20, 6))
sns.heatmap(age_normalized,
           annot=True, fmt='.3f',
           cmap='Blues',
           linewidths=.5,
           cbar_kws={'label': 'percentage(%)'})
plt.title('age_category_preference(%)', pad=20)
plt.xlabel('category')
plt.ylabel('age')
plt.tight_layout()
plt.savefig('age_category_preference.png', dpi=300, bbox_inches='tight')
plt.close('all')

# 按性别分析（不分年龄）
gender_counts = (df.groupby(['gender', 'fav_category'])
                .size()
                .compute())
gender_totals = df.groupby('gender').size().compute()
gender_normalized = (gender_counts / gender_totals * 100).unstack().fillna(0)
plt.figure(figsize=(20, 4))
sns.heatmap(gender_normalized,
           annot=True, fmt='.3f',
           cmap='Greens',
           linewidths=.5,
           cbar_kws={'label': 'percentage(%)'})
plt.title('gender_category_preference(%)', pad=20)
plt.xlabel('category')
plt.ylabel('gender')
plt.tight_layout()
plt.savefig('gender_category_preference.png', dpi=300, bbox_inches='tight')
plt.close('all')

plt.figure(figsize=(8, 6))
is_active_counts = df['is_active'].value_counts().compute()
plt.pie(is_active_counts.values,
        labels=['Inactive', 'Active'],
        autopct='%1.1f%%',  # 显示百分比
        startangle=90,      # 起始角度
        colors=['#ff9999','#66b3ff'],  # 自定义颜色
        explode=(0.05, 0),  # 突出显示第一块
        shadow=True)        # 添加阴影
plt.title('Active vs. Inactive Users Distribution', pad=20)
plt.savefig('active_users_distribution_pie.png', dpi=300, bbox_inches='tight')
plt.close('all')

stats = df.groupby('is_active')['login_count'].agg([
    'min',
    'max',
    'median',
    'mean',
    'std',
    'count'
], shuffle='tasks').compute()
quantiles = df.groupby('is_active')['login_count'].apply(
    lambda x: x.quantile([0.25, 0.75]),
    meta=('login_count', 'float64')  # 指定输出类型
).compute().unstack()
stats = stats.join(quantiles)

plt.figure(figsize=(8, 6))
sns.boxplot(
    x='is_active',
    y='login_count',
    data=df[['is_active', 'login_count']].sample(frac=0.1).compute(),  # 抽取10%数据
    whis=[5, 95]  # 调整须线显示5%和95%分位数
)
plt.title('login_count_by_status_dask', pad=20)
plt.xlabel('status')
plt.ylabel('login_count')
plt.xticks([0, 1], ['inactive', 'active'])
plt.tight_layout()
plt.savefig('login_count_by_status_dask.png', dpi=300, bbox_inches='tight')
plt.close('all')

# 3. Average Price Distribution 直方图
price_array = df['average_price'].to_dask_array()
min_price = df['average_price'].min().compute()
max_price = df['average_price'].max().compute()
bins = np.linspace(min_price, max_price, 50)  # 50个均匀分布的bin
hist, edges = da.histogram(price_array, bins=bins)
hist, edges = da.compute(hist, edges)
plt.figure(figsize=(10, 6))
plt.bar(edges[:-1], hist, width=np.diff(edges), align='edge')
plt.title('average_price_distribution', pad=20)
plt.xlabel('average_price')
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.savefig('average_price_distribution_dask.png', dpi=300, bbox_inches='tight')
plt.close('all')

meta = pd.Series([], dtype='datetime64[ns]', name='purchase_date')
df['purchase_date'] = df['purchase_date'].map_partitions(clean_timestamp, meta=meta)
invalid_count = df['purchase_date'].isna().sum().compute()
print(f"发现 {invalid_count} 个无效日期")

#计算RFM指标
from datetime import datetime
now = datetime.now()
df['recency'] = (now - df['purchase_date']).dt.days
df = df.assign(recency=df['recency'].astype('int64'))
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

end_time = time.time()
total_time = end_time - start_time
print(f"\n全部处理完成，总耗时: {total_time:.2f}秒")