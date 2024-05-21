# 基于双层规划模型的控制诱导模型

## 1 双层规划模型

### 1.1 上层模型

上层模型使用Webster延误公式进行处理
$$
D=∑_(i=1)^4▒∑_(j=1)^2▒{q_ij [(C(1-λ_i )^2)/2(1-λ_i x_ij ) +〖x_ij〗^2/(2q_ij (1-x_ij ) )]}
$$
式中：D为总延误，x是饱和度，C为周期， λ是绿信比=绿灯时间/周期时间，qij是流量

通过延误公式计算出延误，

### 1.2 下层模型

下层模型使用Logit模型进行流量分配。

### 1.3 整体思路

​          这部分主要是针对小规模路网进行仿真，通过上层模型计算得出绿信比，更新延误，并在下层模型中使用MSA进行交通流量分配。

## 2 模型求解

### 2.1 基于遗传算法的上层模型求解

​          上层模型的目的是使用遗传算法迭代求解得到绿灯时间，适应度函数为Webster总延误公式，上层规划模型中，使用遗传算法对每个交叉口的各相位绿灯时间和周期时间进行迭代，对每个路口的红路灯信号控制进行更新处理，初值为0，根据

$$
C=(1.5L+5)/(1-Y)
$$
其中，周期时间 (C)，L 是损失时间（通常指所有相位转换时的总损失时间），Y 是周期中所有相位绿灯比例的总和。

绿灯分配时间分配
$$
G_i=(Y∙C∙q_i)/(∑▒q_i )
$$
  ，  Gi是第 i 相位的绿灯时间，  qi是该相位的流量，Y 和 C 分别是周期中绿灯比例的总和和周期时间。



### 2.2 基于MSA的的下层模型求解

​          下层交通流式静态交通流分配问题，采用迭代加权法( method of successive averages，MSA) 进行求解

​          步骤： ①  初始化。令迭代次数 m = 0，初始化t0，根据各路段自由流行驶时间进行全有全无分配，得到初始解  。

​         ②   令迭代次数 m = m + 1，更新路段走行时间  ，

​         ③   按照路段走行时间  ，将 OD 交通量进行全有全无分配，得到各路段的附加交通量  

​         ④   更新路段流量为

 
$$
x_a^(m+1)=x_a^m+(y^m-x_a^m)/m
$$
​        ⑤  如果连续 2 次迭代的结果相差不大，则停止计算，记录分配结果; 否则，返回步骤选用MAPE 作为收敛标准。



MSA函数中：

1. **初始化**：
   - `od_pairs = list(od_demand.keys())` 初始化 OD 对路径。
   - `my_network.paths = find_shortest_paths_dijkstra(graph, od_pairs)` 使用 Dijkstra 算法找到每对 OD 对应的最短路径。
   - `my_network.update_edge_cost()` 更新边缘的初始成本。
   - `x = [od_demand[od] / len(my_network.paths) for od in od_pairs]` 初始化路径流量。
2. **对主要 OD 对进行第一次路径分配后固定部分需求**：
   - `fixed_flows = {od: fixed_percentage * od_demand[od] for od in primary_od_pairs}` 固定主要 OD 对的部分需求。
3. **迭代更新**：
   - `my_network.update_path_flow(x)` 和 `my_network.update_edge_flow()` 更新路径流量和边缘流量。
   - `my_network.update_path_cost()` 和 `my_network.update_path_prob()` 更新路径成本和路径选择概率。
   - `my_network.paths = find_shortest_paths_dijkstra(graph, od_pairs)` 重新计算最短路径。
   - `y.append((od_demand[od] - fixed_flows[od]) * path.logit_prob + fixed_flows[od])` 计算新的流量，固定主要 OD 对部分需求。
   - `x[idx] = x[idx] + 1 / I * (y[idx] - x[idx])` 逐步平均更新流量。
   - `gap = max([abs(path.prob - path.logit_prob) for path in my_network.paths])` 计算收敛指标 gap。

问题：使用MSA进行流量分配时，对代码调试中遗传算法中的流量xa=0，进一步调试发现边的流量只有最后一次可以被正确分配
