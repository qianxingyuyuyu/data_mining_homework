# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 配置显示设置
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
sns.set_palette('pastel')


# 数据加载
def load_data():
    users = pd.read_csv('ecommerce_users.csv', parse_dates=['registration_date'])
    products = pd.read_csv('ecommerce_products.csv', parse_dates=['listing_date'])
    transactions = pd.read_csv('ecommerce_transactions.csv', parse_dates=['purchase_time'])
    return users, products, transactions


# 数据预处理
def preprocess_data(users, products, transactions):
    # 处理缺失值
    users['age'] = users['age'].fillna(users['age'].median())
    users['gender'] = users['gender'].fillna('Unknown')
    transactions['rating'] = transactions['rating'].fillna(0)  # 示例：用0表示未评分
    # 处理异常交易金额（过滤超过3倍标准差的值）
    amount_mean = transactions['amount'].mean()
    amount_std = transactions['amount'].std()
    transactions = transactions[transactions['amount'] <= amount_mean + 3 * amount_std]
    # 日期格式标准化
    transactions['purchase_date'] = transactions['purchase_time'].dt.date
    # 数据合并
    merged_df = transactions.merge(users, on='user_id').merge(products, on='product_id')
    return merged_df


# 探索性分析与可视化
def exploratory_analysis(merged_df):
    # 1. 产品类别销售分布

    category_sales = merged_df.groupby('category')['amount'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=merged_df,
        y='category',  # 横轴：category
        x='amount',  # 纵轴：amount
        order=category_sales.index,  # 按 amount 总和从多到少排列
        #palette='viridis',  # 配色方案
        width=0.6,  # 箱线图宽度
        linewidth=1.5,  # 边框线宽
        #flierprops=dict(marker='o', markersize=8, markerfacecolor='red')  # 异常值样式
    )

    # 图表标注
    plt.title('Category vs Amount Box Plot', fontsize=16, pad=20)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Amount', fontsize=12)

    # 优化布局
    plt.tight_layout()
    plt.savefig('category_amount_boxplot.png', dpi=300)  # 保存图表

    # 2. 用户年龄分布
    plt.figure(figsize=(10, 6))
    sns.histplot(users['age'], bins=20, kde=True)
    plt.title('User Age Distribution')
    plt.savefig('age_distribution.png')

    # 3. 消费金额与评分关系
    # 计算每个分箱和评级的计数
    max_amount = int(merged_df['amount'].max())

    # Dynamic bin generation
    bins = [0, 50, 100, 150, 200, 300, 400, 500]  # Initial bins: 0-500
    if max_amount > 500:
        bins.append(1000)
        next_edge = 1000
        while next_edge <= max_amount:
            next_edge += 1000
            bins.append(next_edge)
    else:
        bins.append(max_amount + 1)

    # Generate bin labels
    labels = [
        f"{bins[i]}-{bins[i + 1] - 1}"
        for i in range(len(bins) - 1)
    ]

    # Perform binning
    merged_df['amount_bin'] = pd.cut(
        merged_df['amount'],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )

    # Calculate counts per bin and rating
    count_data = merged_df.groupby(['amount_bin', 'rating']).size().unstack().fillna(0)
    ratings = count_data.columns.tolist()

    plt.figure(figsize=(12, 6))
    ax = sns.histplot(
        data=merged_df,
        x='amount_bin',
        hue='rating',
        multiple="stack",
        palette="viridis",
        shrink=0.8,
        edgecolor='white',
        hue_order=ratings,
        legend=True
    )

    plt.title('Amount-Rating Distribution', fontsize=14)
    plt.xlabel('Amount Bins', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Get the patch objects (rectangles) from the plot
    patches = [p for p in ax.patches if p.get_height() > 0]

    for patch in patches:
        # Get the center of the patch
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_y() + patch.get_height() / 2

        # Get the count value
        count = int(patch.get_height())

        if count > 0:  # Only label if count > 0
            ax.text(
                x, y, str(count),
                ha='center', va='center',
                color='white' if patch.get_height() > ax.get_ylim()[1] * 0.2 else 'black',
                fontsize=8,
                fontweight='bold'
            )

    # Adjust legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        reversed(handles), reversed(labels),
        title='Rating',
        bbox_to_anchor=(1.15, 1),
        frameon=False
    )

    plt.tight_layout()
    plt.savefig('stacked_rating_dist_linear.png', dpi=300)


# RFM模型实现
def calculate_rfm(transactions):
    now = datetime.now().date()

    rfm = transactions.groupby('user_id').agg({
        'purchase_time': lambda x: (now - x.max().date()).days,  # Recency
        'transaction_id': 'count',  # Frequency
        'amount': 'sum'  # Monetary
    }).reset_index()

    rfm.columns = ['user_id', 'recency', 'frequency', 'monetary']

    # 分箱处理
    rfm['R'] = pd.qcut(rfm['recency'], 4, labels=False) + 1
    rfm['F'] = pd.qcut(rfm['frequency'], 4, labels=False) + 1
    rfm['M'] = pd.qcut(rfm['monetary'], 4, labels=False) + 1

    # 计算RFM得分
    rfm['RFM_Score'] = rfm['R'] + rfm['F'] + rfm['M']

    # 用户分层
    rfm['segment'] = pd.qcut(rfm['RFM_Score'], 3, labels=['Low', 'Medium', 'High'])

    return rfm


if __name__ == "__main__":
    # 数据加载与预处理
    users, products, transactions = load_data()
    merged_df = preprocess_data(users, products, transactions)

    # 探索性分析
    exploratory_analysis(merged_df)

    # 高价值用户识别
    rfm = calculate_rfm(transactions)
    high_value_users = rfm[rfm['segment'] == 'High']

    # 保存结果
    high_value_users.to_csv('high_value_users.csv', index=False)
    print(f"识别到高价值用户数量: {len(high_value_users)}")
    print(high_value_users.describe())