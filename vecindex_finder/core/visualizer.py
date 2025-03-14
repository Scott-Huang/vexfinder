#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from core.logging import logger

class IndexVisualizer:
    """索引可视化工具，用于展示不同索引参数的性能比较"""
    
    def __init__(self, results_data: Optional[List[Dict]] = None, results_file: Optional[str] = None):
        """
        初始化可视化工具
        
        Args:
            results_data: 索引测试结果数据
            results_file: 索引测试结果文件路径
        """
        self.results = []
        
        if results_data:
            self.results = results_data
        elif results_file:
            self.load_results(results_file)
    
    def load_results(self, file_path: str) -> None:
        """
        从文件加载测试结果
        
        Args:
            file_path: 结果文件路径
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                self.results = df.to_dict('records')
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    self.results = json.load(f)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            logger.info(f"成功加载 {len(self.results)} 条测试结果")
        except Exception as e:
            logger.error(f"加载测试结果失败: {e}")
            raise
    
    def plot_qps_vs_recall(self, output_path: str, title: str = "QPS vs 召回率") -> None:
        """
        绘制QPS与召回率的散点图
        
        Args:
            output_path: 输出文件路径
            title: 图表标题
        """
        if not self.results:
            logger.warning("没有测试结果可绘制")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 设置绘图样式
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        
        # 根据索引类型设置不同颜色
        index_types = df['index_type'].unique()
        colors = sns.color_palette("husl", len(index_types))
        
        # 绘制散点图
        for i, index_type in enumerate(index_types):
            subset = df[df['index_type'] == index_type]
            plt.scatter(subset['avg_recall'], subset['qps'], 
                        label=index_type, color=colors[i], s=100, alpha=0.7)
            
            # 为每个点添加标签
            for _, row in subset.iterrows():
                label = self._get_point_label(row)
                plt.annotate(label, (row['avg_recall'], row['qps']), 
                            fontsize=8, alpha=0.7,
                            xytext=(5, 5), textcoords='offset points')
        
        # 添加图表元素
        plt.xlabel("平均召回率")
        plt.ylabel("每秒查询数 (QPS)")
        plt.title(title)
        plt.legend(title="索引类型")
        plt.grid(True)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"QPS vs 召回率图表已保存到 {output_path}")
        
        # 关闭图表
        plt.close()
    
    def plot_creation_time_vs_qps(self, output_path: str, title: str = "索引创建时间 vs QPS") -> None:
        """
        绘制索引创建时间与QPS的散点图
        
        Args:
            output_path: 输出文件路径
            title: 图表标题
        """
        if not self.results:
            logger.warning("没有测试结果可绘制")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 检查是否有索引创建时间
        if 'index_creation_time' not in df.columns:
            logger.warning("测试结果中没有索引创建时间数据")
            return
        
        # 设置绘图样式
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        
        # 根据索引类型设置不同颜色
        index_types = df['index_type'].unique()
        colors = sns.color_palette("husl", len(index_types))
        
        # 绘制散点图
        for i, index_type in enumerate(index_types):
            subset = df[df['index_type'] == index_type]
            plt.scatter(subset['index_creation_time'], subset['qps'], 
                        label=index_type, color=colors[i], s=100, alpha=0.7)
            
            # 为每个点添加标签
            for _, row in subset.iterrows():
                label = self._get_point_label(row)
                plt.annotate(label, (row['index_creation_time'], row['qps']), 
                            fontsize=8, alpha=0.7,
                            xytext=(5, 5), textcoords='offset points')
        
        # 添加图表元素
        plt.xlabel("索引创建时间 (秒)")
        plt.ylabel("每秒查询数 (QPS)")
        plt.title(title)
        plt.legend(title="索引类型")
        plt.grid(True)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"索引创建时间 vs QPS图表已保存到 {output_path}")
        
        # 关闭图表
        plt.close()
    
    def plot_parameter_impact(self, output_path: str, index_type: str, param_name: str) -> None:
        """
        绘制特定参数对QPS和召回率的影响
        
        Args:
            output_path: 输出文件路径
            index_type: 索引类型
            param_name: 参数名称
        """
        if not self.results:
            logger.warning("没有测试结果可绘制")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 筛选特定索引类型的数据
        df = df[df['index_type'] == index_type]
        
        if df.empty:
            logger.warning(f"没有 {index_type} 类型的索引测试结果")
            return
        
        if param_name not in df.columns:
            logger.warning(f"测试结果中没有参数 {param_name}")
            return
        
        # 设置绘图样式
        sns.set(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 绘制参数对QPS的影响
        sns.boxplot(x=param_name, y='qps', data=df, ax=ax1)
        ax1.set_title(f"{param_name} 对 QPS 的影响")
        ax1.set_xlabel(param_name)
        ax1.set_ylabel("每秒查询数 (QPS)")
        
        # 绘制参数对召回率的影响
        sns.boxplot(x=param_name, y='avg_recall', data=df, ax=ax2)
        ax2.set_title(f"{param_name} 对召回率的影响")
        ax2.set_xlabel(param_name)
        ax2.set_ylabel("平均召回率")
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"参数 {param_name} 影响图表已保存到 {output_path}")
        
        # 关闭图表
        plt.close()
    
    def generate_report(self, output_dir: str) -> None:
        """
        生成综合性能报告
        
        Args:
            output_dir: 输出目录
        """
        if not self.results:
            logger.warning("没有测试结果可生成报告")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 生成QPS vs 召回率图表
        self.plot_qps_vs_recall(os.path.join(output_dir, "qps_vs_recall.png"))
        
        # 生成索引创建时间 vs QPS图表
        if 'index_creation_time' in df.columns:
            self.plot_creation_time_vs_qps(os.path.join(output_dir, "creation_time_vs_qps.png"))
        
        # 为每种索引类型生成参数影响图表
        index_types = df['index_type'].unique()
        for index_type in index_types:
            subset = df[df['index_type'] == index_type]
            
            # 获取该索引类型的特定参数
            params = self._get_index_specific_params(index_type)
            
            for param in params:
                if param in subset.columns:
                    self.plot_parameter_impact(
                        os.path.join(output_dir, f"{index_type}_{param}_impact.png"),
                        index_type,
                        param
                    )
        
        # 生成最佳参数报告
        self._generate_best_params_report(output_dir)
        
        logger.info(f"性能报告已生成到目录 {output_dir}")
    
    def _get_point_label(self, row: pd.Series) -> str:
        """
        根据索引类型生成点标签
        
        Args:
            row: 数据行
            
        Returns:
            点标签
        """
        if row['index_type'] == 'ivfflat':
            return f"nlist={row['nlist']}, nprobe={row['nprobe']}"
        elif row['index_type'] == 'ivfpq':
            return f"nlist={row['nlist']}, m={row['m']}, nprobe={row['nprobe']}"
        elif row['index_type'] == 'hnsw':
            return f"M={row['M']}, ef={row['ef']}"
        else:
            return ""
    
    def _get_index_specific_params(self, index_type: str) -> List[str]:
        """
        获取特定索引类型的参数列表
        
        Args:
            index_type: 索引类型
            
        Returns:
            参数列表
        """
        if index_type == 'ivfflat':
            return ['nlist', 'nprobe']
        elif index_type == 'ivfpq':
            return ['nlist', 'm', 'nprobe', 'refine_factor']
        elif index_type == 'hnsw':
            return ['M', 'efConstruction', 'ef']
        else:
            return []
    
    def _generate_best_params_report(self, output_dir: str) -> None:
        """
        生成最佳参数报告
        
        Args:
            output_dir: 输出目录
        """
        df = pd.DataFrame(self.results)
        
        # 按索引类型分组，找出每种类型中QPS最高且召回率满足要求的参数
        report_data = []
        
        for index_type in df['index_type'].unique():
            subset = df[df['index_type'] == index_type]
            
            # 找出召回率大于0.8的结果
            valid_subset = subset[subset['avg_recall'] >= 0.8]
            
            if not valid_subset.empty:
                # 找出QPS最高的结果
                best_row = valid_subset.loc[valid_subset['qps'].idxmax()]
                
                report_data.append({
                    '索引类型': index_type,
                    'QPS': best_row['qps'],
                    '召回率': best_row['avg_recall'],
                    '创建时间(秒)': best_row.get('index_creation_time', 'N/A'),
                    '最佳参数': self._get_point_label(best_row)
                })
        
        # 创建报告DataFrame
        report_df = pd.DataFrame(report_data)
        
        # 保存为CSV
        report_df.to_csv(os.path.join(output_dir, "best_params_report.csv"), index=False)
        
        # 保存为HTML
        html_content = """
        <html>
        <head>
            <title>向量索引最佳参数报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333366; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .highlight { background-color: #e6ffe6; }
            </style>
        </head>
        <body>
            <h1>向量索引最佳参数报告</h1>
            <p>以下是各索引类型的最佳参数配置（召回率≥0.8且QPS最高）：</p>
            <table>
                <tr>
                    <th>索引类型</th>
                    <th>QPS</th>
                    <th>召回率</th>
                    <th>创建时间(秒)</th>
                    <th>最佳参数</th>
                </tr>
        """
        
        # 找出整体最佳行
        best_overall_idx = report_df['QPS'].idxmax() if not report_df.empty else -1
        
        for i, row in report_df.iterrows():
            highlight = ' class="highlight"' if i == best_overall_idx else ''
            html_content += f"""
                <tr{highlight}>
                    <td>{row['索引类型']}</td>
                    <td>{row['QPS']:.2f}</td>
                    <td>{row['召回率']:.4f}</td>
                    <td>{row['创建时间(秒)']}</td>
                    <td>{row['最佳参数']}</td>
                </tr>
            """
        
        html_content += """
            </table>
            <p>注：高亮行表示整体最佳参数配置</p>
        </body>
        </html>
        """
        
        with open(os.path.join(output_dir, "best_params_report.html"), 'w') as f:
            f.write(html_content)
