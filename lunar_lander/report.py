import csv
import json
import os
from typing import List

from .simulator import SimulationResult


def save_trajectory_csv(result: SimulationResult, filepath: str):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time", "altitude", "velocity", "mass", "fuel_mass",
            "thrust", "thrust_command", "filtered_altitude", "filtered_velocity",
            "measured_altitude", "measured_velocity",
        ])
        for s in result.states:
            writer.writerow([
                s.time,
                s.altitude,
                s.velocity,
                s.mass,
                s.fuel_mass,
                s.thrust,
                s.thrust_command,
                s.filtered_altitude,
                s.filtered_velocity,
                s.measured_altitude,
                s.measured_velocity,
            ])


def save_scores_json(results: List[SimulationResult], filepath: str):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    data = []
    for result in results:
        data.append({
            "strategy": result.strategy_name,
            "status": result.status.value,
            "status_description": result.status.description,
            "flight_time": result.flight_time,
            "fuel_used": result.fuel_used,
            "final_altitude": result.final_state.altitude,
            "final_velocity": result.final_state.velocity,
            "score": {
                "overall": result.score.overall_score,
                "fuel_efficiency": result.score.fuel_efficiency,
                "landing_accuracy": result.score.landing_accuracy,
                "landing_smoothness": result.score.landing_smoothness,
                "time_efficiency": result.score.time_efficiency,
            },
        })
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_markdown_report(
    results: List[SimulationResult],
    scenario_name: str,
    scenario_description: str,
    filepath: str,
):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    lines = []
    lines.append(f"# 月面着陆策略比较报告")
    lines.append("")
    lines.append(f"**场景**: {scenario_name}")
    lines.append("")
    lines.append(f"**描述**: {scenario_description}")
    lines.append("")
    lines.append("## 策略对比总览")
    lines.append("")

    lines.append("| 策略 | 状态 | 综合评分 | 燃料效率 | 着陆精度 | 着陆平稳度 | 时间效率 | 飞行时间(s) | 燃料消耗(kg) |")
    lines.append("|------|------|----------|----------|----------|------------|----------|-------------|--------------|")

    sorted_results = sorted(results, key=lambda r: r.score.overall_score, reverse=True)

    for r in sorted_results:
        lines.append(
            f"| {r.strategy_name} | {r.status.description} | "
            f"{r.score.overall_score:.2f} | {r.score.fuel_efficiency:.2f} | "
            f"{r.score.landing_accuracy:.2f} | {r.score.landing_smoothness:.2f} | "
            f"{r.score.time_efficiency:.2f} | {r.flight_time:.2f} | {r.fuel_used:.2f} |"
        )

    lines.append("")
    lines.append("## 各策略详细结果")
    lines.append("")

    for r in sorted_results:
        lines.append(f"### {r.strategy_name}")
        lines.append("")
        lines.append(f"- **状态**: {r.status.description}")
        lines.append(f"- **综合评分**: {r.score.overall_score:.2f}/100")
        lines.append(f"- **燃料效率**: {r.score.fuel_efficiency:.2f}/100")
        lines.append(f"- **着陆精度**: {r.score.landing_accuracy:.2f}/100")
        lines.append(f"- **着陆平稳度**: {r.score.landing_smoothness:.2f}/100")
        lines.append(f"- **时间效率**: {r.score.time_efficiency:.2f}/100")
        lines.append(f"- **飞行时间**: {r.flight_time:.2f} s")
        lines.append(f"- **燃料消耗**: {r.fuel_used:.2f} kg")
        lines.append(f"- **最终高度**: {r.final_state.altitude:.2f} m")
        lines.append(f"- **最终速度**: {r.final_state.velocity:.2f} m/s")
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
