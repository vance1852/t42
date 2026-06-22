import argparse
import json
import os
import sys

from .config import LanderConfig, Scenario, load_scenario
from .simulator import LanderSimulator
from .strategies import create_strategy
from .report import save_trajectory_csv, save_scores_json, save_markdown_report
from .parameter_scan import parameter_scan, find_best_strategy


def run_single(args):
    config = LanderConfig()
    if args.scenario:
        scenario = load_scenario(args.scenario)
        config = scenario.config
        strategy_params = scenario.strategy_params
    else:
        strategy_params = {}

    strategy = create_strategy(
        strategy_name=args.strategy,
        max_thrust=config.max_thrust,
        gravity=config.gravity,
        mass_dry=config.mass_dry,
        **strategy_params,
    )

    simulator = LanderSimulator(config, strategy)
    result = simulator.run()

    print(f"策略: {args.strategy}")
    print(f"状态: {result.status.description}")
    print(f"综合评分: {result.score.overall_score:.2f}/100")
    print(f"飞行时间: {result.flight_time:.2f} s")
    print(f"燃料消耗: {result.fuel_used:.2f} kg")
    print(f"最终高度: {result.final_state.altitude:.2f} m")
    print(f"最终速度: {result.final_state.velocity:.2f} m/s")

    if args.csv:
        save_trajectory_csv(result, args.csv)
        print(f"轨迹已保存到: {args.csv}")


def run_compare(args):
    if args.scenario:
        scenario = load_scenario(args.scenario)
        config = scenario.config
        strategy_params = scenario.strategy_params
    else:
        config = LanderConfig()
        strategy_params = {}

    strategies = args.strategies if args.strategies else ["constant_decel", "staged_braking", "pid"]

    results = []
    for s_name in strategies:
        strategy = create_strategy(
            strategy_name=s_name,
            max_thrust=config.max_thrust,
            gravity=config.gravity,
            mass_dry=config.mass_dry,
            **strategy_params.get(s_name, {}),
        )
        simulator = LanderSimulator(config, strategy)
        result = simulator.run()
        results.append(result)

    results_sorted = sorted(results, key=lambda r: r.score.overall_score, reverse=True)

    print(f"{'策略':<20} {'状态':<12} {'评分':>8} {'时间(s)':>8} {'燃料(kg)':>10}")
    print("-" * 60)
    for r in results_sorted:
        print(f"{r.strategy_name:<20} {r.status.description:<12} {r.score.overall_score:>8.2f} "
              f"{r.flight_time:>8.2f} {r.fuel_used:>10.2f}")

    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        for r in results:
            save_trajectory_csv(r, os.path.join(args.csv_dir, f"{r.strategy_name}_trajectory.csv"))
        print(f"轨迹CSV已保存到: {args.csv_dir}")

    if args.json:
        save_scores_json(results, args.json)
        print(f"评分JSON已保存到: {args.json}")

    if args.report:
        scenario_name = os.path.basename(args.scenario) if args.scenario else "default"
        save_markdown_report(
            results,
            scenario_name=scenario_name,
            scenario_description="策略比较",
            filepath=args.report,
        )
        print(f"Markdown报告已保存到: {args.report}")


def run_scan(args):
    if args.scenario:
        scenario = load_scenario(args.scenario)
        config = scenario.config
    else:
        config = LanderConfig()

    if not args.param:
        print("错误: 请指定至少一个参数范围", file=sys.stderr)
        sys.exit(1)

    param_ranges = {}
    for p_spec in args.param:
        name, range_str = p_spec.split("=", 1)
        parts = range_str.split(":")
        if len(parts) == 3:
            start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
            values = []
            v = start
            while v <= end + 1e-9:
                values.append(v)
                v += step
            param_ranges[name] = values
        elif len(parts) == 2:
            start, end = float(parts[0]), float(parts[1])
            param_ranges[name] = [start, end]
        else:
            param_ranges[name] = [float(parts[0])]

    results = parameter_scan(
        base_config=config,
        strategy_name=args.strategy,
        param_ranges=param_ranges,
    )

    results_sorted = sorted(results, key=lambda x: x[1].score.overall_score, reverse=True)

    print(f"{'参数组合':<40} {'状态':<12} {'评分':>8}")
    print("-" * 70)
    for params, result in results_sorted[:10]:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"{param_str:<40} {result.status.description:<12} {result.score.overall_score:>8.2f}")

    if args.output:
        data = []
        for params, result in results:
            data.append({
                "params": params,
                "strategy": args.strategy,
                "status": result.status.value,
                "score": result.score.overall_score,
                "flight_time": result.flight_time,
                "fuel_used": result.fuel_used,
            })
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"扫描结果已保存到: {args.output}")


def run_find_best(args):
    if args.scenario:
        scenario = load_scenario(args.scenario)
        config = scenario.config
        strategy_params = scenario.strategy_params
    else:
        config = LanderConfig()
        strategy_params = {}

    strategies = args.strategies if args.strategies else ["constant_decel", "staged_braking", "pid"]

    param_ranges = {}
    if args.pid_params:
        kp_start, kp_end, kp_step = [float(x) for x in args.pid_params.split(":")]
        param_ranges["pid"] = {
            "kp": [kp_start + i * kp_step for i in range(int((kp_end - kp_start) / kp_step) + 1)],
        }

    best_strategy, best_params, best_result = find_best_strategy(
        base_config=config,
        strategy_names=strategies,
        strategy_param_ranges=param_ranges,
        baseline_strategy=args.baseline,
        baseline_params=strategy_params.get(args.baseline, {}),
    )

    print(f"最佳策略: {best_strategy}")
    print(f"最佳参数: {best_params}")
    print(f"状态: {best_result.status.description}")
    print(f"综合评分: {best_result.score.overall_score:.2f}/100")
    print(f"飞行时间: {best_result.flight_time:.2f} s")
    print(f"燃料消耗: {best_result.fuel_used:.2f} kg")

    if args.csv:
        save_trajectory_csv(best_result, args.csv)
        print(f"轨迹已保存到: {args.csv}")


def main():
    parser = argparse.ArgumentParser(description="月面着陆控制策略模拟器")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    single_parser = subparsers.add_parser("single", help="单次模拟")
    single_parser.add_argument("--strategy", "-s", default="pid", help="控制策略")
    single_parser.add_argument("--scenario", "-c", help="场景配置文件")
    single_parser.add_argument("--csv", help="输出轨迹CSV文件")
    single_parser.set_defaults(func=run_single)

    compare_parser = subparsers.add_parser("compare", help="策略比较")
    compare_parser.add_argument("--strategies", nargs="+", help="要比较的策略列表")
    compare_parser.add_argument("--scenario", "-c", help="场景配置文件")
    compare_parser.add_argument("--csv-dir", help="轨迹CSV输出目录")
    compare_parser.add_argument("--json", help="评分JSON输出文件")
    compare_parser.add_argument("--report", help="Markdown报告输出文件")
    compare_parser.set_defaults(func=run_compare)

    scan_parser = subparsers.add_parser("scan", help="参数扫描")
    scan_parser.add_argument("--strategy", "-s", default="pid", help="控制策略")
    scan_parser.add_argument("--scenario", "-c", help="场景配置文件")
    scan_parser.add_argument("--param", "-p", action="append", help="参数范围，格式 name=start:end:step")
    scan_parser.add_argument("--output", "-o", help="扫描结果JSON输出文件")
    scan_parser.set_defaults(func=run_scan)

    best_parser = subparsers.add_parser("best", help="寻找最佳策略")
    best_parser.add_argument("--strategies", nargs="+", help="候选策略列表")
    best_parser.add_argument("--scenario", "-c", help="场景配置文件")
    best_parser.add_argument("--baseline", default="constant_decel", help="基线策略")
    best_parser.add_argument("--pid-params", help="PID参数扫描范围 kp_start:kp_end:step")
    best_parser.add_argument("--csv", help="最佳策略轨迹CSV输出")
    best_parser.set_defaults(func=run_find_best)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
