#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any
from tabulate import tabulate
import math
from core.logging import logger
from core.param_builder import scale_parameters, scale_query_parameters


class Reportor:
    def __init__(self, result_dir: str=None, reports_dir: str=None):
        self.result_dir = result_dir 
        self.reports_dir = reports_dir 
        self.index_type = None
        self.original_table_count = None
        self.sample_table_count = None
        self.analysis_result = None
        self.all_trials = None
        self._load_analysis_result()
    
    def _load_analysis_result(self):
        all_results_file = os.path.join(self.result_dir, "all_results.json")
        if os.path.exists(all_results_file):
            with open(all_results_file, "r") as f:
                self.all_trials = json.load(f)
                logger.info(f"已加载所有试验结果: {all_results_file}")

        best_result_file = os.path.join(self.result_dir, "best_result.json")
        if os.path.exists(best_result_file):
            with open(best_result_file, "r") as f:
                self.analysis_result = json.load(f)
                if "best_performance" not in self.analysis_result and "success" in self.analysis_result:
                    recall = self.analysis_result.get("recall", 0)
                    avg_query_time = self.analysis_result.get("avg_query_time", 0)
                    qps = self.analysis_result.get("qps", 0)
                    self.analysis_result["best_performance"] = {
                        "recall": recall,
                        "avg_query_time": avg_query_time,
                        "qps": qps
                    }
                logger.info(f"已加载最佳结果: {best_result_file}")
        else:
            self.analysis_result = max(self.all_trials, key=lambda x: x["score"])

        if not self.analysis_result:
            logger.warning("无法加载任何分析结果")
            return

        self.index_type = self.analysis_result.get("index_type", None)
        table_info = self.analysis_result.get("table_info", None)
        self.original_table_count = table_info.get("original_table_count", 0)
        self.sample_table_count = table_info.get("sample_table_count", 0)

    def generate_report(self) -> Dict[str, Any]:
        if not self.analysis_result:
            logger.error("未找到分析结果，无法生成报告")
            return {"success": False, "error": "未找到分析结果"}
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir, exist_ok=True)

        result_dir_name = os.path.basename(self.result_dir)
        output_dir = os.path.join(self.reports_dir, result_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        if self.original_table_count and self.sample_table_count:
            scale_factor = self.original_table_count / self.sample_table_count
        else:
            scale_factor = 1.0
            logger.warning("未找到表大小信息，使用默认缩放因子1.0")

        best_index_param = self.analysis_result.get("index_param", {})
        best_query_param = self.analysis_result.get("best_query_param", {})

        recommended_index_param = scale_parameters(self.index_type, best_index_param, scale_factor)
        recommended_query_param = scale_query_parameters(self.index_type, best_query_param, scale_factor)

        report_data = {
            "index_type": self.index_type,
            "create_index_time": self.analysis_result.get("create_index_time", 0),
            "index_size": self.analysis_result.get("index_size", 0) / (1024 * 1024),  # 转为MB
            "best_performance": self.analysis_result.get("best_performance", {}),
            "scale_factor": scale_factor,
            "sample_data": {
                "table_count": self.sample_table_count,
                "index_param": best_index_param,
                "query_param": best_query_param
            },
            "source_data": {
                "table_count": self.original_table_count,
                "recommended_index_param": recommended_index_param,
                "recommended_query_param": recommended_query_param
            }
        }

        self._generate_text_report(report_data, output_dir)
        self._generate_visual_report(report_data, output_dir)

        with open(os.path.join(output_dir, "full_report.json"), "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"报告已生成到目录: {output_dir}")
        return report_data
    
    def _generate_text_report(self, report_data: Dict[str, Any], output_dir: str):
        report_lines = []
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines.append(f"# {report_data['index_type'].upper()} 索引参数分析报告")
        report_lines.append(f"生成时间: {now}\n")
        report_lines.append("## 性能摘要")
        performance = report_data["best_performance"]
        report_lines.append(f"- 召回率: {performance.get('recall', 0):.4f}")
        report_lines.append(f"- 平均查询时间: {performance.get('avg_query_time', 0):.6f} 秒")
        report_lines.append(f"- QPS: {performance.get('qps', 0):.2f}")
        report_lines.append(f"- 创建索引时间: {report_data['create_index_time']:.2f} 秒")
        report_lines.append(f"- 索引大小: {report_data['index_size']:.2f} MB\n")

        report_lines.append("## 采样数据最佳参数")
        report_lines.append(f"- 采样数据大小: {report_data['sample_data']['table_count']} 行")

        report_lines.append("\n### 索引参数")
        index_param_table = [[param, value] for param, value in report_data['sample_data']['index_param'].items()]
        report_lines.append(tabulate(index_param_table, headers=["参数名", "值"], tablefmt="pipe"))

        report_lines.append("\n### 查询参数")
        query_param_table = [[param, value] for param, value in report_data['sample_data']['query_param'].items()]
        report_lines.append(tabulate(query_param_table, headers=["参数名", "值"], tablefmt="pipe"))

        report_lines.append("\n## 源数据推荐参数")
        report_lines.append(f"- 源数据大小: {report_data['source_data']['table_count']} 行")
        report_lines.append(f"- 缩放因子: {report_data['scale_factor']:.2f}x\n")

        report_lines.append("### 推荐索引参数")
        rec_index_param_table = []
        for param, value in report_data['source_data']['recommended_index_param'].items():
            original_value = report_data['sample_data']['index_param'].get(param, "-")
            rec_index_param_table.append([param, original_value, value])
        report_lines.append(tabulate(rec_index_param_table, headers=["参数名", "采样数据值", "推荐值"], tablefmt="pipe"))

        report_lines.append("\n### 推荐查询参数")
        rec_query_param_table = []
        for param, value in report_data['source_data']['recommended_query_param'].items():
            original_value = report_data['sample_data']['query_param'].get(param, "-")
            rec_query_param_table.append([param, original_value, value])
        report_lines.append(tabulate(rec_query_param_table, headers=["参数名", "采样数据值", "推荐值"], tablefmt="pipe"))

        report_lines.append("\n## 参数说明")
        if self.index_type == "ivfflat":
            report_lines.append("- ivf_nlist: IVF聚类中心数量，影响索引质量和查询性能")
            report_lines.append("- ivf_probes: 查询时检查的聚类数量，增加可提高召回率但降低性能")
        elif self.index_type == "graph_index":
            report_lines.append("- m: 每个节点的最大边数，影响索引大小和查询性能")
            report_lines.append("- ef_construction: 构建图时的候选邻居数量，影响索引质量")
            report_lines.append("- quantizer: 向量数据量化类型")
            report_lines.append("- hnsw_ef_search: 查询时的候选数量，增加可提高召回率但降低性能")
        elif self.index_type == "ivfpq":
            report_lines.append("- ivf_nlist: IVF聚类中心数量，影响索引质量和查询性能")
            report_lines.append("- num_subquantizers: PQ编码的子向量数量，影响压缩率和精度")
            report_lines.append("- ivf_probes: 查询时检查的聚类数量，增加可提高召回率但降低性能")
            report_lines.append("- ivfpq_refine_k_factor: 二次精排的比例因子，增加可提高精度")
        elif self.index_type == "diskann":
            report_lines.append("- m: 每个节点的最大边数，影响索引大小和查询性能")
            report_lines.append("- ef_construction: 构建图时的候选邻居数量，影响索引质量")
            report_lines.append("- occlusion_factor: 索引算法裁边参数，影响查询精度和构建，查询时间")
            report_lines.append("- diskann_search_list_size: 查询时的候选集大小，增加可提高召回率但降低性能")

        with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info(f"已生成文本报告: {os.path.join(output_dir, 'report.md')}")
    
    def _generate_visual_report(self, report_data: Dict[str, Any], output_dir: str):
        if not self.all_trials:
            logger.warning("未找到试验数据，无法生成可视化报告")
            return

        try:
            self._create_parameter_comparison(report_data, output_dir)
            if len(self.all_trials) > 1:
                self._create_trials_comparison(output_dir)
            logger.info(f"已生成可视化报告在 {output_dir} 目录下")
        except Exception as e:
            logger.error(f"生成可视化报告失败: {str(e)}")

    def _create_parameter_comparison(self, report_data: Dict[str, Any], output_dir: str):
        fig = make_subplots(rows=2, cols=1, subplot_titles=("索引参数对比", "查询参数对比"))

        index_params = []
        sample_values = []
        recommended_values = []

        for param in report_data['source_data']['recommended_index_param']:
            if param in report_data['sample_data']['index_param']:
                index_params.append(param)
                sample_values.append(report_data['sample_data']['index_param'][param])
                recommended_values.append(report_data['source_data']['recommended_index_param'][param])

        fig.add_trace(
            go.Bar(name="采样数据", x=index_params, y=sample_values, marker_color="blue"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name="推荐参数", x=index_params, y=recommended_values, marker_color="red"),
            row=1, col=1
        )

        query_params = []
        sample_query_values = []
        recommended_query_values = []
        for param in report_data['source_data']['recommended_query_param']:
            if param in report_data['sample_data']['query_param']:
                query_params.append(param)
                sample_query_values.append(report_data['sample_data']['query_param'][param])
                recommended_query_values.append(report_data['source_data']['recommended_query_param'][param])

        fig.add_trace(
            go.Bar(name="采样数据", x=query_params, y=sample_query_values, marker_color="blue"),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name="推荐参数", x=query_params, y=recommended_query_values, marker_color="red"),
            row=2, col=1
        )

        fig.update_layout(
            title=f"{self.index_type.upper()} 索引参数对比",
            barmode='group',
            height=800,
            width=800
        )

        fig.write_html(os.path.join(output_dir, "parameter_comparison.html"))
    
    def _create_trials_comparison(self, output_dir: str):
        trial_numbers = []
        recalls = []
        query_times = []
        index_times = []
        for trial in self.all_trials:
            if "trial_number" in trial and "best_performance" in trial:
                trial_numbers.append(trial["trial_number"])
                recalls.append(trial["best_performance"].get("recall", 0))
                query_times.append(trial["best_performance"].get("avg_query_time", 0) * 1000)  # 转为毫秒
                index_times.append(trial.get("create_index_time", 0))

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("召回率变化", "查询时间变化(毫秒)", "索引创建时间变化(秒)", "召回率与查询时间关系"),
            specs=[[{}, {}], [{}, {}]]
        )

        fig.add_trace(
            go.Scatter(x=trial_numbers, y=recalls, mode='lines+markers', name="召回率", 
                      line=dict(color="blue", width=2)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=trial_numbers, y=query_times, mode='lines+markers', name="查询时间(毫秒)", 
                      line=dict(color="red", width=2)),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=trial_numbers, y=index_times, mode='lines+markers', name="索引创建时间(秒)", 
                      line=dict(color="green", width=2)),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=recalls, y=query_times, mode='markers', name="召回率 vs 查询时间",
                      marker=dict(size=10, color=trial_numbers, colorscale="Viridis", 
                                showscale=True, colorbar=dict(title="试验编号"))),
            row=2, col=2
        )

        fig.update_layout(
            title=f"{self.index_type.upper()} 索引试验结果对比",
            height=1000,
            width=1000,
            showlegend=False
        )

        fig.update_xaxes(title_text="试验编号", row=1, col=1)
        fig.update_yaxes(title_text="召回率", row=1, col=1)

        fig.update_xaxes(title_text="试验编号", row=1, col=2)
        fig.update_yaxes(title_text="查询时间(毫秒)", row=1, col=2)

        fig.update_xaxes(title_text="试验编号", row=2, col=1)
        fig.update_yaxes(title_text="索引创建时间(秒)", row=2, col=1)

        fig.update_xaxes(title_text="召回率", row=2, col=2)
        fig.update_yaxes(title_text="查询时间(毫秒)", row=2, col=2)

        fig.write_html(os.path.join(output_dir, "trials_comparison.html"))


def generate_report(result_dir: str, reports_dir: str) -> Dict[str, Any]:
    reportor = Reportor(result_dir, reports_dir)
    return reportor.generate_report()
