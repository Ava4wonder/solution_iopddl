import itertools
from collections import defaultdict

def interval_intersection(intervals):
    # 如果输入的区间列表为空，直接返回空列表
    if not intervals:
        return []
    # 初始化当前的交集区间为第一个区间
    current_intersection = intervals[0]
    result = []
    # 遍历剩余的区间
    for interval in intervals[1:]:
        # 计算当前交集区间和当前遍历区间的交集
        start = max(current_intersection[0], interval[0])
        end = min(current_intersection[1], interval[1])
        # 如果交集存在（即起始点小于等于结束点）
        if start <= end:
            # 更新当前交集区间
            current_intersection = [start, end]
        else:
            # 如果交集不存在，将当前交集区间添加到结果列表中
            if current_intersection[0] <= current_intersection[1]:
                result.append(current_intersection)
            # 更新当前交集区间为当前遍历的区间
            current_intersection = interval
    # 将最后一个交集区间添加到结果列表中
    if current_intersection[0] <= current_intersection[1]:
        result.append(current_intersection)
    return result

# 测试数据
intervals = [
    [30, 70],
    [40, 70],
    [50, 120],
    [110, 140],
    [110, 150]
]

# 调用函数计算交集区间
print(interval_intersection(intervals))

# def eliminate_invalid_subnodes(problem, usage_by_time_slot):
#     intervals = problem['nodes']['intervals']
#     usages = problem['nodes']['usages']
#     usage_limit = problem['usage_limit']

#     valid_subnodes = defaultdict(list)

#     for i, (interval, usage_list) in enumerate(zip(intervals, usages)):
#         start, end = interval
#         min_other_usage = min(
#             sum(usage_by_time_slot[t] - usage_list[j] for t in range(start, end)) for j in range(len(usage_list))
#         )

#         for j, usage in enumerate(usage_list):
#             if usage + min_other_usage <= usage_limit:
#                 valid_subnodes[i].append(j)

#     return valid_subnodes

# # Input problem
# data = {
#     "problem": {
#         "name": "example",
#         "nodes": {
#             "intervals": [
#                 [30, 70],
#                 [40, 70],
#                 [50, 120],
#                 [110, 140],
#                 [110, 150]
#             ],
#             "costs": [
#                 [15],
#                 [55, 65],
#                 [25, 45, 35],
#                 [85, 75],
#                 [95]
#             ],
#             "usages": [
#                 [10],
#                 [25, 25],
#                 [15, 20, 15],
#                 [10, 10],
#                 [15]
#             ]
#         },
#         "edges": {
#             "nodes": [
#                 [0, 1],
#                 [0, 2],
#                 [1, 3],
#                 [2, 4],
#                 [3, 4]
#             ],
#             "costs": [
#                 [30, 40],
#                 [50, 10, 40],
#                 [90, 10, 20, 80],
#                 [60, 20, 30],
#                 [70, 60]
#             ]
#         },
#         "usage_limit": 50
#     }
# }

# problem = data['problem']

# # Step 1: Identify peak memory usage time slots
# peak_slots, usage_by_time_slot = find_peak_memory_slots(problem)
# print("Peak usage time slots:", peak_slots, usage_by_time_slot)

# # Step 2: Eliminate invalid subnodes
# valid_subnodes = eliminate_invalid_subnodes(problem, usage_by_time_slot)
# print("Valid subnodes after elimination:", dict(valid_subnodes))
