# mvdr_of_real_signal_based_on_rir.m 逐行代码解释

> 本文档按源码行号逐行解释，便于对 MVDR 双波束（目标指向 + 干扰指向）处理流程进行精读。

| 行号 | 代码 | 解释 |
|---:|---|---|
| 1 | set(groot, ... | 
| 2 |     'defaultAxesFontName','Times New Roman', ... 
| 3 |     'defaultTextFontName','Times New Roman', ... 
| 4 |     'defaultAxesFontSize',20, ... 
| 5 |     'defaultTextFontSize',24, ... 
| 6 |     'defaultLineLineWidth',1.2); |

设置MATLAB图形的全局默认属性：坐标轴和文本字体统一为Times New Roman；坐标轴字号设为20，文本字号设为24；线条宽度设为1.2。所有后续创建的图形对象将自动应用这些样式，确保绘图风格一致。

| 行号 | 代码 | 解释 |
|---:|---|---|
| 7 |  | 空行，用于分隔代码结构，提升可读性。 |
| 8 | %% MVDR (3D steering: azimuth + elevation) using RIR delays only (no manual delay) | 代码分节标题，标识后续模块功能|
| 9 | clear; close all; clc; | 清理工作区/图窗/命令行，确保脚本从干净状态运行。 |
| 10 |  | 空行，用于分隔代码结构，提升可读性。 |
| 11 | %% ========== 1. Read mic positions (assume mic_positions.xlsx has X Y columns) ========== | 代码分节标题，标识后续模块功能|
| 12 | data = readmatrix('mic_positions.xlsx'); |  
| 13 | X = data(:,1); Y = data(:,2); |  
| 14 | Z = zeros(size(X)); |  
| 15 | mic_pos = [X Y Z] / 1000;|  |
| 16 |  | 空行，用于分隔代码结构，提升可读性。 |
| 17 | % center to given array center (optional) | 注释行|
| 18 | array_center = [2.5 2 1.5];    | 阵列中心 |
| 19 | mic_pos = mic_pos - mean(mic_pos,1) + array_center;    |
| 20 |  | 空行，用于分隔代码结构，提升可读性。 |
| 21 | Nmic = size(mic_pos,1); |  |
| 22 | r_center = mean(mic_pos,1);   % 阵列中心（用于平面波相位参考） |  |
| 23 | fprintf('Nmic = %d, mic positions loaded (meters).\n', Nmic); | 向命令行打印运行状态或诊断信息，便于跟踪执行过程。 |
| 24 |  | 空行，用于分隔代码结构，提升可读性。 |
| 25 | figure(1); | 创建或切换图窗，为可视化结果做准备。 |
| 26 | scatter(X, Y, 40, 'filled'); | 绘制散点图，显示阵元或声源空间位置。 |
| 27 | % scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 40, 'filled'); % 三维阵列图 |  |
| 28 | % axis equal; grid on; | 
| 29 | % xlabel('X (m)'); ylabel('Y (m)'); | 
| 30 | % title('Microphone Array Geometry'); | 
| 31 |  | 空行，用于分隔代码结构，提升可读性。 |
| 32 | set(gca, 'LineWidth', 1); | 设置图形对象的全局/当前坐标轴属性（字体、线宽等）。 |
| 33 | exportgraphics(gcf, 'Microphone Array Geometry.pdf', ... | 将当前图形导出为文件，便于论文或报告使用。 |
| 34 |     'ContentType','image'); | 执行该语句并抑制命令行回显。 |
| 35 |  | 空行，用于分隔代码结构，提升可读性。 |
| 36 |  | 空行，用于分隔代码结构，提升可读性。 |

读取麦克风位置Excel数据（X,Y坐标），将单位从毫米转换为米，并调整阵列中心至指定点[2.5,2,1.5]。计算麦克风数量后，绘制2D位置散点图并导出为PDF文件，用于可视化阵列几何结构。

| 行号 | 代码 | 解释 |
|---:|---|---|
| 37 | %% ========== 2. Room, RIR and signals parameters ========== | 
| 38 | c = 340; | 
| 39 | fs = 16000; | 
| 40 | SNR = 30;          | 
| 41 | % INR = 5;          % dB (interference relative to target amplitude) | 
| 42 | nsample = 4096;    % RIR length | 
| 43 |  | 空行，用于分隔代码结构，提升可读性。 |
| 44 | % source 3D positions | 
| 45 | s_target = [2.5 1 4];   % used for RIR generation (m) | 
| 46 | s_interf  = [2.5 3 4]; | 
| 47 |  | 空行，用于分隔代码结构，提升可读性。 |
| 48 | % optional GPU acceleration (requires Parallel Computing Toolbox) | 
| 49 | use_gpu = true; | 
| 50 | gpu_ok = false; | 
| 51 | if use_gpu &amp;&amp; gpuDeviceCount &gt; 0 | 
| 52 |     g = gpuDevice; | 
| 53 |     gpu_ok = true; | 
| 54 |     fprintf('GPU enabled: %s\n', g.Name); | 
| 55 | else | 条件分支兜底路径；当前面条件都不满足时执行。 |
| 56 |     fprintf('GPU not used. Running on CPU.\n'); | 
| 57 | end | 结束当前控制结构（if/for/parfor 等）的作用域。 |
| 58 |  | 空行，用于分隔代码结构，提升可读性。 |
| 59 | % speed options | 
| 60 | use_parallel_rir = true;   % parallelize per-mic RIR generation when possible | 
| 61 | mvdr_frame_stride = 2;     % update MVDR weights every N frames (1 = full update) | 
| 62 |  | 空行，用于分隔代码结构，提升可读性。 |
| 63 | % source level control for reproducible tone experiments | 
| 64 | normalize_source_rms = true; | 
| 65 | INR_dB = 0;                 % interference-to-target level before room propagation | 
| 66 |  | 空行，用于分隔代码结构，提升可读性。 |
| 67 | % false: one weight per freq (fast), true: per-frame adaptive | 
| 68 | if ~exist('mvdr_use_time_varying', 'var') | 
| 69 |     mvdr_use_time_varying = false; | 
| 70 | end | 
| 71 |  | 空行，用于分隔代码结构，提升可读性。 |
| 72 | mvdr_fmax_hz = 8000;             % MVDR upper band; raise to 8 kHz for full-band processing | 
| 73 | mvdr_fmin_hz = 300;              % very low frequencies have weak spatial selectivity | 
| 74 | mvdr_progress_step = 20;         % print progress every N frequency bins | 
| 75 | if ~exist('use_oracle_intnoi_cov', 'var') | 
| 76 |     use_oracle_intnoi_cov = true;    % true: use interference+noise covariance (debug upper bound) | 
| 77 | end | 
| 78 | if ~exist('use_oracle_tarnoi_cov', 'var') | 
| 79 |     use_oracle_tarnoi_cov = true;    % true: use target+noise covariance for interference-steered MVDR | 
| 80 | end | 
| 81 |  | 空行，用于分隔代码结构，提升可读性。 |

- c, fs, nsample：声速、采样率、RIR 长度。
- s_target, s_interf：目标/干扰声源三维坐标。
- use_gpu：如果有并行工具箱+GPU则启用加速。
- use_parallel_rir：并行生成每个麦克风的 RIR。
- mvdr_frame_stride：时变 MVDR 时每隔几帧更新一次权重，减少计算量。
- normalize_source_rms 与 INR_dB：控制输入目标/干扰的幅度关系。
- mvdr_use_time_varying：false 表示每个频点用一套固定权重，true 表示时变权重。
- mvdr_fmin_hz / mvdr_fmax_hz：MVDR 作用频带下限/上限。
- use_oracle_intnoi_cov / use_oracle_tarnoi_cov：是否用“已知分量”构协方差（更像上界/调试模式）。

**MVDR 的关键数学含义**

每个频点解下面这个约束最优化：
$$
\min_{\mathbf{w}} \mathbf{w}^H \mathbf{R}_{xx}\mathbf{w}
\quad
\text{s.t.}\quad
\mathbf{w}^H \mathbf{a}=1
$$
闭式解是：
$$
\mathbf{w}_{MVDR}=
\frac{\mathbf{R}_{xx}^{-1}\mathbf{a}}
{\mathbf{a}^H\mathbf{R}_{xx}^{-1}\mathbf{a}}
$$

| 行号 | 代码 | 解释 |
|---:|---|---|
| 82 | %% ====================== 可视化房间、声源和麦克风阵列位置 ====================== | 
| 83 | figure(2); clf; | 创建或切换图窗，为可视化结果做准备。 |
| 84 | hold on; grid on; axis equal; | 控制图层叠加模式，决定新绘图是否覆盖旧图。 |
| 85 |  | 空行，用于分隔代码结构，提升可读性。 |
| 86 | % 1. 绘制房间边界（矩形框） | 
| 87 | L = [5 4 6];  % 房间长宽高 | 
| 88 | xv = [0 L(1) L(1) 0 0]; | 
| 89 | yv = [0 0 L(2) L(2) 0]; | 
| 90 | z0 = zeros(size(xv)); | 地面矩形的顶点坐标
| 91 |  | 空行，用于分隔代码结构，提升可读性。 |
| 92 | % 地面和顶面 | 
| 93 | plot3(xv, yv, z0, 'k--', 'LineWidth', 1.2); | 绘制曲线/三维曲线，用于展示波形、方向图或几何关系。 |
| 94 | plot3(xv, yv, L(3)*ones(size(z0)), 'k--', 'LineWidth', 1.2); | 
| 95 |  | 空行，用于分隔代码结构，提升可读性。 |
| 96 | % 垂直边 | 
| 97 | for i = 1:4 | for 循环开始，按索引迭代执行后续代码块。 |
| 98 |     plot3([xv(i) xv(i)], [yv(i) yv(i)], [0 L(3)], 'k--'); | 
| 99 | end | 
| 100 |  | 空行，用于分隔代码结构，提升可读性。 |
| 101 | % 2. 绘制麦克风阵列位置 | 
| 102 | scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 100, 'filled', 'b'); | 绘制散点图，显示阵元或声源空间位置。 |
| 103 | text(mean(mic_pos(:,1)) + 0.2, mean(mic_pos(:,2)), mean(mic_pos(:,3)) + 0.3, 'Mic Array', ... | 在图中添加文字标注，解释关键对象。 |
| 104 |      'Color', 'b', 'FontWeight', 'bold'); | 
| 105 |  | 空行，用于分隔代码结构，提升可读性。 |
| 106 | % 3. 绘制目标声源与干扰源  | 
| 107 | scatter3(s_target(1), s_target(2), s_target(3), 60, 'r', 'filled'); | 
| 108 | text(s_target(1) - 0.6, s_target(2) - 0.6, s_target(3) + 0.4, 'Target', 'Color', 'r', 'FontWeight', 'bold'); | 在图中添加文字标注，解释关键对象。 |
| 109 |  | 空行，用于分隔代码结构，提升可读性。 |
| 110 | scatter3(s_interf(1), s_interf(2), s_interf(3), 60, 'm', 'filled'); | 
| 111 | text(s_interf(1) + 0.2, s_interf(2), s_interf(3) + 0.1, 'Interference', 'Color', 'm', 'FontWeight', 'bold'); | 
| 112 |  | 空行，用于分隔代码结构，提升可读性。 |
| 113 | % 4. 可选：连接阵列中心与声源方向 | 
| 114 | % r_center = mean(mic_pos,1); | 注释行：r_center = mean(mic_pos,1); |
| 115 | plot3([r_center(1) s_target(1)], [r_center(2) s_target(2)], [r_center(3) s_target(3)], 'r--', 'LineWidth', 1.2); | 绘制曲线/三维曲线，用于展示波形、方向图或几何关系。 |
| 116 | plot3([r_center(1) s_interf(1)], [r_center(2) s_interf(2)], [r_center(3) s_interf(3)], 'm--', 'LineWidth', 1.2); | 
| 117 |  | 空行，用于分隔代码结构，提升可读性。 |
| 118 | % 5. 设置视角与标签 | 
| 119 | xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)'); | 设置坐标轴标签，说明物理量及单位。 |
| 120 | % title('Room, Sources, and Microphone Array Layout'); | 
| 121 | view(45, 25); | 设置三维视角（方位角/俯仰角）。 |
| 122 | % legend({'Room boundary','Microphones','Target Source','Interference'}, 'Location','best'); | 
| 123 | %% | 代码分节标题，标识后续模块功能： |
| 124 | set(gca, 'LineWidth', 1); | 设置坐标轴线宽
| 125 | exportgraphics(gcf, 'Room, Sources, and Microphone Array Layout.pdf', ... | 导出当前图到 PDF |
| 126 |     'ContentType','image'); | 
| 127 |  | 空行，用于分隔代码结构，提升可读性。 |
| 128 |  | 空行，用于分隔代码结构，提升可读性。 |

三维场景可视化：把房间、麦克风阵列、目标声源、干扰声源画在同一张 3D 图里，并导出成 PDF。

1. `figure(2); clf; hold on; grid on; axis equal;`  
初始化第 2 幅图并清空，开启叠加绘图、网格、等比例坐标。  
等比例很关键，不然空间几何会被拉伸，看起来方向和距离会失真。

2. `L = [5 4 6];` 到 `z0 = zeros(size(xv));`  
定义房间长宽高（5m×4m×6m），并构造地面矩形的顶点坐标。

3. 两条 `plot3(...)` 画“地面”和“天花板”  
用虚线画出底面和顶面轮廓。

4. `for i = 1:4 ... end`  
画四条竖直边，把底面和顶面连起来，形成完整房间框架。

5. `scatter3(mic_pos...)` + `text(...,'Mic Array',...)`  
画出所有麦克风三维位置，并标注“Mic Array”。

6. 两组 `scatter3 + text`（Target / Interference）  
分别画目标声源和干扰声源，并加文字标签。

7. 两条 `plot3([r_center ...])`  
从阵列中心 `r_center` 分别连到目标/干扰声源，直观看“指向关系”。

8. `xlabel/ylabel/zlabel` + `view(45,25)`  
设置坐标轴标签和视角。`view(45,25)` 表示方位角 45°、俯仰 25°，是一个比较容易观察空间关系的角度。

9. `set(gca,'LineWidth',1); exportgraphics(...)`  
设置坐标轴线宽并导出当前图到 PDF：  
`Room, Sources, and Microphone Array Layout.pdf`。

**这段对后续 MVDR 的作用**
- 它不参与数值计算，只做可视化检查。  
- 但非常有用：你可以快速确认 `mic_pos`、`s_target`、`s_interf` 是否摆放正确，避免后面波束方向错了却难定位原因。


| 行号 | 代码 | 解释 |
|---:|---|---|
| 129 | %% ========== 3. Fraunhofer check (far-field criterion) ========== | 
| 130 | D = 0.15;           % 阵列最大孔径 (m) — 150 mm | 
| 131 | f_max = 8000;       % 关心的最高频率（Hz） | 
| 132 | lambda_min = c / f_max; | 
| 133 | R_far = 2 * D^2 / lambda_min;   % Fraunhofer distance (approx) | 
| 134 |  | 空行，用于分隔代码结构，提升可读性。 |
| 135 | fprintf('Array diameter D=%.3f m, f_max=%d Hz, lambda_min=%.3f m\n', D, f_max, lambda_min); | 
| 136 | fprintf('Fraunhofer distance R_far ≈ %.3f m\n', R_far); | 
| 137 |  | 空行，用于分隔代码结构，提升可读性。 |
| 138 | % compute actual distances of sources to array center | 
| 139 | dist_s_target = norm(s_target - r_center); | 目标声源到阵列中心的实际距离
| 140 | dist_s_interf  = norm(s_interf  - r_center); | 干扰声源到阵列中心的实际距离
| 141 |  | 空行，用于分隔代码结构，提升可读性。 |
| 142 | fprintf('Distance target-&gt;array center = %.3f m, interference = %.3f m\n', dist_s_target, dist_s_interf); | 
| 143 |  | 空行，用于分隔代码结构，提升可读性。 |
| 144 | use_farfield_target = dist_s_target &gt;= R_far; | 
| 145 | use_farfield_interf  = dist_s_interf  &gt;= R_far; | 
| 146 |  | 空行，用于分隔代码结构，提升可读性。 |
| 147 | if ~use_farfield_target | 
| 148 |     fprintf('Warning: target is within near-field (%.3f &lt; %.3f). Using near-field steering for target.\n', dist_s_target, R_far); | 
| 149 | else | 条件分支兜底路径；当前面条件都不满足时执行。 |
| 150 |     fprintf('Target satisfies far-field criterion. Using plane-wave steering for target.\n'); | 
| 151 | end | 结束当前控制结构（if/for/parfor 等）的作用域。 |
| 152 | if ~use_farfield_interf | 条件判断开始；仅当条件为真时执行后续代码块。 |
| 153 |     fprintf('Warning: interference is within near-field. Using near-field steering for interference.\n'); | 
| 154 | else | 
| 155 |     fprintf('Interference satisfies far-field criterion. Using plane-wave steering for interference.\n'); | 
| 156 | end | 结束当前控制结构（if/for/parfor 等）的作用域。 |
| 157 |  | 空行，用于分隔代码结构，提升可读性。 |

做“远场/近场判定”，目的是决定后面导向矢量该用平面波模型还是球面波模型。

1. 定义阵列孔径和最高关注频率  
D = 0.15 m，f_max = 8000 Hz。  
用最高频率来判定是合理的，因为频率越高，远场条件越苛刻。

2. 计算最短波长  
lambda_min = c / f_max。  
你这里 c=340，所以 lambda_min 约等于 0.0425 m。

3. 计算 Fraunhofer 距离  
R_far = 2D^2 / lambda_min。  
代入后大约是 1.06 m。这个值可以理解为“超过这个距离，平面波近似通常可接受”。

4. 计算声源到阵列中心的实际距离  
dist_s_target = norm(s_target - r_center)  
dist_s_interf = norm(s_interf - r_center)  
根据你当前坐标，两者都大约是 2.69 m，明显大于 1.06 m。

5. 得到两个布尔开关  
use_farfield_target = dist_s_target >= R_far  
use_farfield_interf = dist_s_interf >= R_far  
所以当前配置下这两个开关基本都会是 true。

6. 打印提示信息  
如果小于 R_far，会提示近场并建议用近场导向；否则提示满足远场并用平面波导向。

这段和后续的直接关系是：在导向矢量构造处（后面的 steering vector 部分），程序会根据这两个开关分支计算相位。  
- 远场: 用方向向量投影（平面波）  
- 近场: 用到各麦克风与声源距离差（球面波）


| 行号 | 代码 | 解释 |
|---:|---|---|
| 158 | %% ========== 4. Generate RIRs (per-mic) ========== | 
| 159 | % L = [5 4 6];    % room dims | 
| 160 | % beta = 0.4;     % wall reflection coefficient | 
| 161 | % mtype = 'omnidirectional';  % depends on your rir_generator | 
| 162 | % order = -1; | 
| 163 | % dim = 3; | 
| 164 | % orientation = [pi/2 0]; | 
| 165 | % hp_filter = 1; | 
| 166 |  | 空行，用于分隔代码结构，提升可读性。 |
| 167 | L = [5 4 6];    % room dims | 
| 168 | beta = 0;     % wall reflection coefficient | 
| 169 | mtype = 'omnidirectional';  % depends on your rir_generator | 
| 170 | order = -1; | 
| 171 | dim = 3; | 
| 172 | orientation = 0; | 
| 173 | hp_filter = true; | 
| 174 |  | 空行，用于分隔代码结构，提升可读性。 |
| 175 | h_target = zeros(Nmic, nsample);    % 初始化目标和干扰的RIR矩阵 | 
| 176 | h_interf = zeros(Nmic, nsample); | 
| 177 |  | 空行，用于分隔代码结构，提升可读性。 |
| 178 | % beta=0时只有直达声，RIR长度可安全缩短以加速 | 
| 179 | if beta == 0 | 条件判断开始；仅当条件为真时执行后续代码块。 |
| 180 |     max_dist = max([vecnorm(mic_pos - s_target,2,2); vecnorm(mic_pos - s_interf,2,2)]); | 计算并使用向量/矩阵范数，常用于归一化或距离计算。 |
| 181 |     nsample_direct = max(256, ceil(max_dist / c * fs) + 128); | 
| 182 |     nsample_use = min(nsample, nsample_direct); |
| 183 | else | 
| 184 |     nsample_use = nsample; | 
| 185 | end | 
| 186 |  | 空行，用于分隔代码结构，提升可读性。 |
| 187 | if nsample_use &lt; nsample | 
| 188 |     h_target = zeros(Nmic, nsample_use); | 
| 189 |     h_interf = zeros(Nmic, nsample_use); | 
| 190 |     fprintf('Using shortened RIR length: %d (from %d).\n', nsample_use, nsample); | 
| 191 | end | 
| 192 |  | 空行，用于分隔代码结构，提升可读性。 |
| 193 | fprintf('Generating RIRs for %d microphones (this may take a while)...\n', Nmic); | 
| 194 | if use_parallel_rir &amp;&amp; license('test', 'Distrib_Computing_Toolbox') | 
| 195 |     p = gcp('nocreate'); | 
| 196 |     if isempty(p) | 
| 197 |         try | 异常处理的尝试块开始。 |
| 198 |             parpool('Processes'); | 管理并行计算池，以支持 parfor 并行执行。 |
| 199 |         catch ME | 异常捕获分支开始，用于处理 try 块中的错误。 |
| 200 |             msg = ['Failed to start process-based pool: ' char(ME.message) '. Falling back to serial RIR.']; | 
| 201 |             warning('rir:pool', '%s', msg); | 输出警告信息，不中断程序但提示潜在问题。 |
| 202 |             use_parallel_rir = false; | 
| 203 |         end | 结束当前控制结构（if/for/parfor 等）的作用域。 |
| 204 |     elseif contains(lower(class(p)), 'thread') | 分支条件判断；当前面条件不满足时尝试该分支。 |
| 205 |         delete(p); | 
| 206 |         try | 异常处理的尝试块开始。 |
| 207 |             parpool('Processes'); | 管理并行计算池，以支持 parfor 并行执行。 |
| 208 |         catch ME | 
| 209 |             msg = ['Failed to switch to process-based pool: ' char(ME.message) '. Falling back to serial RIR.']; | 
| 210 |             warning('rir:pool', '%s', msg); | 
| 211 |             use_parallel_rir = false; | 
| 212 |         end | 
| 213 |     end | 
| 214 | end | 
| 215 |  | 空行，用于分隔代码结构，提升可读性。 |
| 216 | if use_parallel_rir &amp;&amp; license('test', 'Distrib_Computing_Toolbox') | 
| 217 |     parfor m = 1:Nmic      % 并行按麦克风生成RIR | 
| 218 |         rm = mic_pos(m,:); | 
| 219 |         ht = rir_generator(c, fs, rm, s_target, L, beta, nsample_use, mtype, order, dim, orientation, hp_filter); | 调用 RIR 生成函数，计算声源到麦克风的房间脉冲响应。 |
| 220 |         hi = rir_generator(c, fs, rm, s_interf, L, beta, nsample_use, mtype, order, dim, orientation, hp_filter); | 
| 221 |         if iscolumn(ht), ht = ht.'; end | 
| 222 |         if iscolumn(hi), hi = hi.'; end | 
| 223 |         h_target(m,:) = ht; | 
| 224 |         h_interf(m,:)  = hi; | 
| 225 |     end | 
| 226 | else | 
| 227 |     for m = 1:Nmic      % 循环为每个麦克风生成目标和干扰的RIR，并确保为行向量后存储。 | for 循环开始，按索引迭代执行后续代码块。 |
| 228 |         rm = mic_pos(m,:); | 
| 229 |         ht = rir_generator(c, fs, rm, s_target, L, beta, nsample_use, mtype, order, dim, orientation, hp_filter); | 
| 230 |         hi = rir_generator(c, fs, rm, s_interf, L, beta, nsample_use, mtype, order, dim, orientation, hp_filter); | 
| 231 |         if iscolumn(ht), ht = ht.'; end | 
| 232 |         if iscolumn(hi), hi = hi.'; end | 
| 233 |         h_target(m,:) = ht; | 
| 234 |         h_interf(m,:)  = hi; | 
| 235 |     end | 
| 236 | end | 
| 237 | fprintf('RIR generation done.\n'); | 
| 238 |  | 空行，用于分隔代码结构，提升可读性。 |

生成每个麦克风到目标/干扰声源的 RIR（房间脉冲响应），并根据环境自动选择串行或并行执行。

1. 房间与 RIR 参数初始化  
设置房间尺寸、反射系数、麦克风类型等参数。  
这里 `beta = 0` 表示仅直达声、无反射。

2. 预分配 RIR 存储矩阵  
- `h_target`: 每个麦克风对应目标声源的 RIR
- `h_interf`: 每个麦克风对应干扰声源的 RIR  
矩阵行对应麦克风，列对应时间采样点。

3. 根据是否只有直达声，自动缩短 RIR 长度  
- 如果 `beta == 0`，代码估计最大传播距离，把所需长度改为 `nsample_use`，通常远小于 `nsample`
- 目的：减少后续卷积和 RIR 生成开销，加速明显

4. 准备并行池（如果可用）  
- 先检查并行工具箱许可
- 若没有池则尝试启动 `parpool('Processes')`
- 若已有线程池，先删掉再换成进程池
- 启动失败会降级为串行，并给出 warning  
这段是“健壮性处理”，避免并行环境异常导致脚本中断。

5. 按麦克风逐个生成目标/干扰 RIR  
- 并行模式：`parfor m = 1:Nmic`
- 串行模式：`for m = 1:Nmic`
- 每个麦克风位置 `rm = mic_pos(m,:)`
- 分别调用两次 `rir_generator(...)` 得到 `ht` 和 `hi`
- 如果返回列向量就转成行向量，保证能写入 `h_target(m,:)` 和 `h_interf(m,:)`  
最后打印 `RIR generation done.` 表示这一阶段完成。

这段和后续关系是：后面会用 `conv(x_target, h_target(m,:), 'same')`、`conv(x_interf, h_interf(m,:), 'same')` 合成每个通道观测信号，所以这里的 RIR 质量和长度会直接影响后续 MVDR 效果与运行时间。

**rir_generator函数**

一、函数作用与接口  
函数用于基于镜像声源法生成房间脉冲响应（RIR）。

函数形式：
```matlab
h, beta_hat = rir_generator(c, fs, r, s, L, beta, nsample, mtype, order, dim, orientation, hp_filter)
```

二、参数怎么理解

1. 必选核心参数  
- c：声速，单位 m/s。  
- fs：采样率，单位 Hz。  
- r：接收点坐标，大小 M×3（每行一个麦克风）。  
- s：声源坐标，1×3。  
- L：房间尺寸，1×3，对应 x,y,z 方向长度。  
- beta：两种含义二选一。  
  - 方案A：1×6 反射系数，按六个墙面给出。  
  - 方案B：单个混响时间参数（论文里称 beta2，对应 RT60，单位秒）。

2. 可选/控制参数  
- nsample：输出 RIR 长度（采样点数）。  
- mtype：麦克风指向性类型，可选 omnidirectional、subcardioid、cardioid、hypercardioid、bidirectional。  
- order：最大反射阶数，-1 表示自动按目标 RIR 长度估计最大可用反射阶。
- dim：房间维度，2 或 3。  
- orientation：麦克风朝向，方位角/俯仰角（弧度）。   
  1. 如果确实是全向麦：保持 mtype 为 omnidirectional 即可，则全向麦对入射方向增益近似一致，所以朝向参数通常被忽略。orientation 写 0 或 [0 0] 都可以，结果应几乎不变。

  2. 如果用的是心形/超心形等指向麦。必须同时做两件事：  
    - 在 RIR 里设置正确的 mtype 和每个麦克风的 orientation（方位角+俯仰角，弧度）。  
    - 在波束形成导向矢量里加入指向性响应项（或直接用实测/仿真的传递函数作 steering）。  
  否则会出现模型失配，常见表现是干扰抑制变差、零陷变浅、输出失真增大。

- hp_filter：是否高通，true 开启，false 关闭。

三、输出是什么  
- h：大小 M×nsample，每个接收点一条 RIR。  
- beta_hat：当输入的是 RT60（不是六面反射系数）时，返回换算得到的平均反射系数。

四、论文里最关键的两个“机制”

1. 多通道支持  
论文明确说 r 可以是多行，因此可一次性算多个麦克风的 RIR，不必逐麦克风循环调用。

2. 指向性衰减模型  
信号衰减按极坐标模型：
A(θ) = α + (1-α)cos(θ)  
其中 α 与 mtype 对应：  
- Omnidirectional: 1  
- Subcardioid: 0.75  
- Cardioid: 0.5  
- Hypercardioid: 0.25  
- Bidirectional: 0

五、结合你当前脚本的参数设置来解读  
你在 mvdr_of_real_signal_based_on_rir.m 里用了：
- L = [5 4 6]：三维房间。  
- beta = 0：全吸收边界，几乎只保留直达声。  
- mtype = omnidirectional：全向麦。  
- order = -1：自动反射阶。  
- dim = 3：三维计算。  
- orientation = 0：对全向麦影响很小。  
- hp_filter = true：开启高通。  

这组设置的效果是：更接近“直达声传播 + 很少/无混响”，有利于先验证 MVDR 主流程稳定性。

六、你这份代码里可进一步优化的一点  
论文支持 r 为 M×3 一次输入，你现在是每个麦克风循环调用一次。  
如果你的 rir_generator 实现与论文一致，可以改成一次传入全部 mic_pos，通常更快，也更简洁。

七、建议的使用策略（按实验目标）

1. 先做算法验证  
- beta 设小（甚至 0）  
- mtype 用 omnidirectional  
- order 先用 -1  
目的：先看 MVDR 数值链路是否正确。

2. 再做真实房间模拟  
- beta 改为六面反射系数，或直接给 RT60  
- 适当增大 nsample  
- 尝试 cardioid/hypercardioid + orientation  
目的：更贴近真实采集场景。

3. 做大规模阵列时  
- 优先使用多通道输入 r（M×3）一次计算  
- 配合并行池或 C++ 版本接口（论文提到 MATLAB 封装 C++ 实现更快）


| 行号 | 代码 | 解释 |
|---:|---|---|
| 239 | %% ========== 5. Generate source signals and convolve with RIR (do NOT add manual delays) ========== | 
| 240 | %% ========== Load real source signals (.wav) ========== | 
| 241 | [target_sig, fs_t] = audioread('Normal_part92.wav'); | 读取目标音频文件及采样率。 |
| 242 | [interf_sig, fs_i] = audioread('振安1#反_part46.wav'); | 读取干扰音频文件及采样率。 |
| 243 |  | 
| 244 | % mono conversion | 
| 245 | if size(target_sig,2) &gt; 1 | 目标转单通道
| 246 |     target_sig = mean(target_sig,2); | 
| 247 | end | 
| 248 | if size(interf_sig,2) &gt; 1 | 干扰转单通道
| 249 |     interf_sig = mean(interf_sig,2); | 
| 250 | end | 
| 251 |  | 
| 252 | % resample to system fs | 
| 253 | if fs_t ~= fs | 
| 254 |     target_sig = resample(target_sig, fs, fs_t); | 目标重采样到系统fs
| 255 | end | 
| 256 | if fs_i ~= fs | 
| 257 |     interf_sig = resample(interf_sig, fs, fs_i); | 干扰重采样到系统fs
| 258 | end | 
| 259 |  | 
| 260 | % length alignment | 
| 261 | Nt = min(length(target_sig), length(interf_sig)); | 截成相同长度 Nt
| 262 | x_target = target_sig(1:Nt); | 
| 263 | x_interf = interf_sig(1:Nt); | 
| 264 |  | 
| 265 | if normalize_source_rms | 
| 266 |     x_target = x_target / (sqrt(mean(x_target.^2)) + eps); | RMS归一化
| 267 |     x_interf = x_interf / (sqrt(mean(x_interf.^2)) + eps); | 
| 268 | end | 
| 269 |  | 
| 270 | % set source INR before room propagation | 
| 271 | x_interf = x_interf * 10^(INR_dB/20); | INR_dB调节干扰相对强度
| 272 |  | 
| 273 | t = (0:Nt-1).' / fs; | 
| 274 |  | 
| 275 |  | 
| 276 | Nt = length(x_target);  % 初始化信号矩阵 | 
| 277 | X_target = zeros(Nt, Nmic); |
| 278 | X_interf  = zeros(Nt, Nmic); |
| 279 |  | 
| 280 | % convolution (same length) | 
| 281 | for m = 1:Nmic      % 每个麦克风通道进行卷积，保持相同长度 | 
| 282 |     X_target(:,m) = conv(x_target, h_target(m,:), 'same'); | 目标与各自 RIR 做卷积，得到多通道 X_target
| 283 |     X_interf(:,m)  = conv(x_interf,  h_interf(m,:),  'same'); | 干扰与各自 RIR 做卷积，得到多通道X_interf
| 284 | end | 
| 285 |  | 
| 286 | % % scale interference by INR (dB) 根据INR缩放干扰信号 | 
| 287 | % X_interf = X_interf / (10^(INR/20));     | 
| 288 |  | 
| 289 | % add noise to meet overall SNR per channel (relative to target) | 
| 290 | % 计算目标信号的RMS，根据SNR计算期望的噪声RMS，生成高斯噪声并调整其RMS | 
| 291 | % sig_rms = sqrt(mean(X_target.^2, 1)); | 
| 292 | % desired_noise_rms = sig_rms ./ (10^(SNR/20)); | 
| 293 | % noise = randn(size(X_target)); | 
| 294 | % cur_noise_rms = sqrt(mean(noise.^2, 1)); | 
| 295 | % noise = noise .* (desired_noise_rms ./ cur_noise_rms); | 
| 296 |  | 
| 297 | % direct input from loaded target/interference signals (no synthetic noise) | 
| 298 | noise = zeros(size(X_target)); | 当前不加白噪声
| 299 | X_noisy = X_target + X_interf; | 输入混合信号 X_noisy=X_target+X_interf
| 300 | fprintf('Signals prepared from direct target/interference signals (no added noise).\n'); | 
| 301 |  | 





| 行号 | 代码 | 解释 |
|---:|---|---|
| 302 | %% ========== SNR BEFORE MVDR (reference: mic 1) ========== | 做“波束形成前”的输入端基线评估。
| 303 | ref_mic = 1; | 设定 ref_mic = 1，后续所有输入指标都基于第1个麦通道计算。
| 304 |  | 
| 305 | x_tar_in = X_target(:, ref_mic); | x_tar_in：目标分量
| 306 | x_intnoi_in = X_interf(:, ref_mic) + noise(:, ref_mic); | x_intnoi_in：干扰+噪声分量
| 307 |  | 
| 308 | P_tar_in = mean(x_tar_in.^2); | 计算目标分量功率
| 309 | P_intnoi_in = mean(x_intnoi_in.^2); | 计算干扰+噪声分量功率
| 310 | P_int_in = mean(X_interf(:, ref_mic).^2); | 计算干扰分量功率
| 311 | P_noise_in = mean(noise(:, ref_mic).^2); | 计算噪声分量功率
| 312 |  | 
| 313 | SNR_in_dB = 10*log10(P_tar_in / P_intnoi_in); | SINR
| 314 | SIR_in_dB = 10*log10(P_tar_in / P_int_in); | SIR
| 315 | if P_noise_in &gt; 0 | 
| 316 |     SNR_noise_in_dB = 10*log10(P_tar_in / P_noise_in); | SNR
| 317 | else | 
| 318 |     SNR_noise_in_dB = inf; | 若无噪声则设为 inf
| 319 | end | 
| 320 |  | 
| 321 | fprintf('\n===== INPUT SNR =====\n'); | 把三项结果打印出来，作为后面 MVDR 输出增益比较的“前测值”。
| 322 | fprintf('Input SINR (mic %d, target/(interference+noise)): %.2f dB\n', ref_mic, SNR_in_dB); | 
| 323 | fprintf('Input SIR  (target/interference)              : %.2f dB\n', SIR_in_dB); | 
| 324 | fprintf('Input SNRn (target/noise only)                : %.2f dB\n', SNR_noise_in_dB); | 
| 325 | fprintf('=====================\n'); | 
| 326 |  | 
| 327 |  | 



| 行号 | 代码 | 解释 |
|---:|---|---|
| 328 | %% ========== 6. STFT parameters and compute STFT per channel ========== | 把时域多通道信号变成频域表示，供后续逐频点 MVDR 求权重使用
| 329 | Nfft = 1024; |STFT参数设置。Nfft = 1024：FFT点数，决定频率分辨率。
| 330 | win = hamming(512); | win = hamming(512)：窗长 512 点。
| 331 | noverlap = 256; | noverlap = 256：50% 重叠。
| 332 |  | 
| 333 | % compute STFT channel-wise; we store as S(freq, time, channel) | 
| 334 | % 对每个麦克风通道进行STFT，结果存储在S（频率×时间×通道） | 每个麦克风单独做 STFT
| 335 | for m = 1:Nmic | 
| 336 |     [S(:,:,m), F, T] = stft(X_noisy(:,m), fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft, 'FrequencyRange', 'onesided'); | 对每个通道 X_noisy(:,m) 调用 stft；结果放入 S(:,:,m)，即三维张量 S(频率, 时间帧, 通道)；F 是频率轴，T 是时间帧轴
| 337 | end | 
| 338 | Fbins = length(F);  % 获取频率点和时间帧 | Fbins = length(F)：频率bin数量。
| 339 | Tframes = length(T); | Tframes = length(T)：时间帧数量。
| 340 |  | 
| 341 | % limit up to f_limit (mvdr_fmin_hz-mvdr_fmax_hz) | 选择MVDR处理频段
| 342 | f_start = find(F &gt;= mvdr_fmin_hz, 1, 'first'); | f_start 是第一个满足 F >= mvdr_fmin_hz 的bin。
| 343 | f_limit = find(F &lt;= mvdr_fmax_hz, 1, 'last'); | f_limit 是最后一个满足 F <= mvdr_fmax_hz 的bin。
| 344 | if isempty(f_start), f_start = 1; end | 如果边界没找到，用全频默认值兜底。
| 345 | if isempty(f_limit), f_limit = Fbins; end | 
| 346 | if f_start &gt; f_limit | 如果出现异常区间（f_start > f_limit），回退到从低频开始并限制到 5kHz 以内的有效bin。
| 347 |     f_start = 1; | 
| 348 |     f_limit = min(Fbins, find(F &lt;= 5000, 1, 'last')); | 
| 349 | end | 
| 350 | fprintf('STFT computed: %d freq bins, %d time frames. MVDR bins %d..%d (%.1f-%.1f Hz).\n', ... | 打印实际处理区间
| 351 |     Fbins, Tframes, f_start, f_limit, F(f_start), F(f_limit)); | 输出：总频率bin数、时间帧数。实际MVDR使用的 bin范围和对应Hz范围。
| 352 |  | 
| 353 |  | 

先把多通道信号映射到时频域，再把后续MVDR约束在指定频段，减少无效频率对稳健性的影响并控制计算量。





| 行号 | 代码 | 解释 |
|---:|---|---|
| 354 | %% ========== STFT of target-only and interference+noise (for SNR eval) ========== | 为后续 MVDR 权重求解和性能评估，提前准备“可分离的时频分量”
| 355 | S_tar = zeros(size(S)); | 分配4个三维时频张量，都用 zeros(size(S)) 初始化，说明维度与前面混合信号的 S 完全一致，都是 频率×时间×通道。S_tar：目标信号的 STFT
| 356 | S_interf = zeros(size(S)); | S_interf：干扰信号的 STFT
| 357 | S_intnoi = zeros(size(S)); | S_intnoi：干扰+噪声 的 STFT
| 358 | S_tarnoi = zeros(size(S)); | S_tarnoi：目标+噪声 的 STFT
| 359 |  | 
| 360 | for m = 1:Nmic | 对每个麦克风通道分别做 STFT
| 361 |     S_tar(:,:,m) = stft(X_target(:,m), fs, ... | 对 X_target(:,m) 做 STFT，写入 S_tar(:,:,m)
| 362 |         'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft, 'FrequencyRange', 'onesided'); | 
| 363 |  | 
| 364 |     S_interf(:,:,m) = stft(X_interf(:,m), fs, ... | 对 X_interf(:,m) 做 STFT，写入 S_interf(:,:,m)
| 365 |         'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft, 'FrequencyRange', 'onesided'); | 
| 366 |  | 
| 367 |     S_intnoi(:,:,m) = stft(X_interf(:,m) + noise(:,m), fs, ... | 对 X_interf(:,m)+noise(:,m) 做 STFT，写入 S_intnoi(:,:,m)
| 368 |         'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft, 'FrequencyRange', 'onesided'); | 
| 369 |  | 
| 370 |     S_tarnoi(:,:,m) = stft(X_target(:,m) + noise(:,m), fs, ... | 对 X_target(:,m)+noise(:,m) 做 STFT，写入 S_tarnoi(:,:,m)
| 371 |         'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft, 'FrequencyRange', 'onesided'); | 
| 372 | end | 这些量后面分别用于：目标指向分支：估计/评估目标与干扰噪声分量；干扰指向分支：估计/评估干扰与目标噪声分量
| 373 |  | 
| 374 | if gpu_ok | 若启用 GPU，把这些矩阵搬到显存。S, S_tar, S_interf, S_intnoi, S_tarnoi 都转换为 gpuArray，后续协方差与 MVDR 求解可在 GPU 上运行，加速频点循环计算
| 375 |     S = gpuArray(S); | 
| 376 |     S_tar = gpuArray(S_tar); | 
| 377 |     S_interf = gpuArray(S_interf); | 
| 378 |     S_intnoi = gpuArray(S_intnoi); | 
| 379 |     S_tarnoi = gpuArray(S_tarnoi); | 
| 380 | end | 
| 381 |  | 

构造“可解释的频域分量字典”，让后面的 MVDR 不仅能输出波束结果，还能分别计算目标保留量与干扰抑制量。


| 行号 | 代码 | 解释 |
|---:|---|---|
| 382 | %% ===== Far-field DOA unit vectors (from source position) ===== | 
| 383 | u_target = (s_target - r_center).'; | 
| 384 | u_target = u_target / norm(u_target); | 
| 385 |  | 
| 386 | u_interf = (s_interf - r_center).'; | 
| 387 | u_interf = u_interf / norm(u_interf); | 
| 388 |  | 
| 389 |  | 


构造“到达方向单位向量（DOA unit vector）”，供后面远场导向矢量计算使用

1. 目标方向向量  
先算从阵列中心指向目标声源的向量  
$$
\mathbf{v}_t=\mathbf{s}_{target}-\mathbf{r}_{center}
$$
并转成列向量。

再单位化  
$$
\mathbf{u}_{target}=\frac{\mathbf{v}_t}{\|\mathbf{v}_t\|}
$$

2. 干扰方向向量  
对干扰源做同样处理，得到  
$$
\mathbf{u}_{interf}=\frac{\mathbf{s}_{interf}-\mathbf{r}_{center}}{\|\mathbf{s}_{interf}-\mathbf{r}_{center}\|}
$$

3. 作用  
后续在远场相位模型里，会用这两个单位向量计算各麦克风相位差（投影距离），例如形式是  
$$
\phi \propto (\mathbf{r}_m-\mathbf{r}_{center})^\top \mathbf{u}
$$
所以这一步本质是在定义“波从哪个方向来”。


| 行号 | 代码 | 解释 |
|---:|---|---|
| 390 | %% ========== 7. Far-field steering vector construction ========== | **整份脚本里最关键的“导向矢量建模”部分：按频点、按麦克风构造目标和干扰的 steering vector，供后续 MVDR 求权重使用**
| 391 | if gpu_ok | 如果 GPU 可用，就直接在显存里分配 gpuArray，加速后续循环。
| 392 |     a_target = gpuArray.zeros(Nmic, Fbins); | 分配导向矢量矩阵，a_target 大小 Nmic×Fbins
| 393 |     a_interf = gpuArray.zeros(Nmic, Fbins); | a_interf 大小 Nmic×Fbins
| 394 | else | 
| 395 |     a_target = zeros(Nmic, Fbins); | 
| 396 |     a_interf = zeros(Nmic, Fbins); | 
| 397 | end | 
| 398 |  | 
| 399 | for k = 1:Fbins | 
| 400 |     freq = F(k); | 外层按频点循环，每个频点 freq=F(k) 单独建模
| 401 |     if k &gt; f_limit | k>f_limit 时跳过，表示只处理前面设置的 MVDR 有效频带
| 402 |         continue; | 
| 403 |     end | 
| 404 |  | 
| 405 |     for m = 1:Nmic |内层按麦克风循环，对每个麦克风 m 分别计算目标相位 phase_t 和干扰相位 phase_i
| 406 |         if use_farfield_target | 远场/近场可切换：  
| 407 |             phase_t = -2*pi*freq/c * ((mic_pos(m,:) - r_center) * u_target); | 1. 远场：用阵元相对阵列中心的投影距离，形式是 $(mic\_pos(m,:)-r\_center)\cdot u$。
| 408 |         else | 
| 409 |             d_m_t = norm(s_target - mic_pos(m,:)); | 
| 410 |             d_ref_t = norm(s_target - r_center); | 
| 411 |             phase_t = -2*pi*freq/c * (d_m_t - d_ref_t); | 2. 近场：用真实传播路程差，形式是 $d_m-d_{ref}$，其中 ref 是阵列中心到声源距离
| 412 |         end | 
| 413 |  | 
| 414 |         if use_farfield_interf | 
| 415 |             phase_i = -2 * pi * freq/c * ((mic_pos(m,:) - r_center) * u_interf); | 
| 416 |         else | 
| 417 |             d_m_i = norm(s_interf - mic_pos(m,:)); | 
| 418 |             d_ref_i = norm(s_interf - r_center); | 
| 419 |             phase_i = -2*pi*freq/c * (d_m_i - d_ref_i); | 
| 420 |         end | 
| 421 |  | 
| 422 |         a_target(m,k) = exp(1j * phase_t); | a_target(m,k)=exp(1j*phase_t)，a_interf(m,k) = exp(1j * phase_i)，相位转复指数。这一步把几何延迟映射成频域相位响应。
| 423 |         a_interf(m,k) = exp(1j * phase_i); | 
| 424 |     end | 
| 425 |  | 
| 426 |     % normalize (important for MVDR stability) | 每个频点归一化。a_target(:,k)/norm(a_target(:,k))，a_interf(:,k)/norm(a_interf(:,k))。作用是数值稳定，避免不同频点幅值尺度差异影响 MVDR 线性方程条件数。
| 427 |     a_target(:,k) = a_target(:,k) / norm(a_target(:,k)); | 
| 428 |     a_interf(:,k) = a_interf(:,k) / norm(a_interf(:,k)); | 
| 429 | end | 
| 430 |  | 
| 431 | if use_farfield_target &amp;&amp; use_farfield_interf | 打印当前建模模式。目标和干扰，两者都远场 / 都近场 / 混合模式（一个远场一个近场）都会提示。
| 432 |     fprintf('Using FAR-FIELD steering vectors for target and interference.\n'); | 
| 433 | elseif ~use_farfield_target &amp;&amp; ~use_farfield_interf | 
| 434 |     fprintf('Using NEAR-FIELD steering vectors for target and interference.\n'); | 
| 435 | else | 
| 436 |     fprintf('Using MIXED steering vectors (target/interference far-field flags differ).\n'); | 
| 437 | end | 
| 438 |  | 
| 439 |  | 

为每个频点生成“目标方向响应向量”和“干扰方向响应向量”，相当于告诉 MVDR 在该频点应该“无失真保留谁、压制谁”。




| 行号 | 代码 | 解释 |
|---:|---|---|
| 440 | %% ========== 8. Memory-safe per-frame MVDR (diagonal loading adapted) ========== | 进入 MVDR 主计算前的“初始化与参数设定”
| 441 | fprintf('Running dual MVDR (target-steered + interference-steered) ...\n'); | 启动提示，打印：即将运行双分支 MVDR（目标指向 + 干扰指向）。
| 442 |  | 
| 443 | % target-steered branch: keep target, suppress interference+noise | 初始化目标指向分支的输出容器，用 squeeze(S(:,:,ref_mic)) 作为初值，是为了先拿到正确尺寸的频率×时间矩阵，后面循环会逐频点被 MVDR 结果覆盖。
| 444 | Yf_tgt = squeeze(S(:,:,ref_mic)); | Yf_tgt：目标分支最终输出的时频结果（初始先用参考麦克风谱填充）
| 445 | Y_tar_tgt = squeeze(S_tar(:,:,ref_mic)); | Y_tar_tgt：目标分量经目标分支后的结果
| 446 | Y_intnoi_tgt = squeeze(S_intnoi(:,:,ref_mic)); | Y_intnoi_tgt：干扰+噪声分量经目标分支后的结果
| 447 |  | 
| 448 | % interference-steered branch: keep interference, suppress target+noise | 初始化干扰指向分支的输出容器
| 449 | Yf_int = squeeze(S(:,:,ref_mic)); | Yf_int：干扰分支最终输出的时频结果（同样先以参考麦克风谱初始化）
| 450 | Y_interf_int = squeeze(S_interf(:,:,ref_mic)); |Y_interf_int：干扰分量经干扰分支后的结果
| 451 | Y_tarnoi_int = squeeze(S_tarnoi(:,:,ref_mic)); |Y_tarnoi_int：目标+噪声分量经干扰分支后的结果
| 452 |  | 
| 453 | % Tunable parameters (you can try Mavg=21/epsilon=1e-2, or smaller) | 设置 MVDR 稳定性参数
| 454 | Mavg = 31;                     % averaging window for covariance (frames) | Mavg=31：协方差滑动平均窗口帧数（越大越平滑，响应越慢）
| 455 | epsilon = 1e-4;                % base loading (relative scale) | epsilon=1e-4：对角加载强度（防止协方差矩阵病态/不可逆）
| 456 |  | 
| 457 | % optional shrinkage factor for additional stability (set 0 to disable) | 
| 458 | shrink_alpha = 0.01;           % 0..0.3 typical; 0 means no shrinkage | shrink_alpha=0.01：收缩系数（将协方差向标量对角阵轻微收缩，提高稳健性）
| 459 |  | 
| 460 | tic | tic 开启计时，后面会用于打印每段 MVDR 循环耗时。

这段不是在“算权重”，而是在为后续双分支 MVDR 循环准备输出矩阵和数值稳定参数。


| 行号 | 代码 | 解释 |
|---:|---|---|
| 461 | %% 对每个频率和时间帧，用局部滑窗估计协方差矩阵（目标指向分支） | **“目标指向分支”的 MVDR 主循环，也就是整份代码里真正做抑制干扰的核心求解部分**
| 462 | for k = f_start:f_limit | 逐频点处理，频点从 f_start 到 f_limit。每个频点单独建协方差、单独算权重，这是窄带 MVDR 标准做法。
| 463 |     Xkf = squeeze(S(k,:,:)).';   % Nmic x Tframes | 取该频点下的多通道时频数据。Xkf：混合信号（目标+干扰+噪声），尺寸 Nmic×Tframes
| 464 |     Xkf_tar = squeeze(S_tar(k,:,:)).';        % Nmic x Tframes | Xkf_tar：目标分量
| 465 |     Xkf_intnoi = squeeze(S_intnoi(k,:,:)).';  % Nmic x Tframes | Xkf_intnoi：干扰+噪声分量
| 466 |     if use_oracle_intnoi_cov | 选择用于估计协方差的观测，
| 467 |         Xkf_cov = Xkf_intnoi; | 若 use_oracle_intnoi_cov=true，用 Xkf_intnoi 估计协方差（理想上界，便于调试）。
| 468 |     else | 
| 469 |         Xkf_cov = Xkf; | 否则用真实混合 Xkf（更贴近实际）。
| 470 |     end | 
| 471 |     at = a_target(:,k); | 
| 472 |     if norm(at) &lt; 1e-12 | 读取目标导向矢量并做有效性检查。at = a_target(:,k)：若范数过小则跳过该频点，避免数值异常。
| 473 |         continue; | 
| 474 |     end | 
| 475 |  | 
| 476 |     if mvdr_use_time_varying | 两种权重更新模式——时变模式 mvdr_use_time_varying=true
| 477 |         w_last = zeros(Nmic,1, 'like', at); | 
| 478 |         for n = 1:Tframes | 
| 479 |             % 按步长更新权重，其余帧复用上一次权重以降低求解次数 | 
| 480 |             if n == 1 \|\| mod(n-1, mvdr_frame_stride) == 0 | 
| 481 |                 t1 = max(1, n - floor(Mavg/2)); | 
| 482 |                 t2 = min(Tframes, n + floor(Mavg/2)); | 
| 483 |                 Xloc = Xkf_cov(:, t1:t2); | 
| 484 |  | 
| 485 |                 Rxx = (Xloc * Xloc') / size(Xloc,2); | 
| 486 |  | 
| 487 |                 if shrink_alpha &gt; 0 | 
| 488 |                     mu = trace(Rxx) / Nmic; | 
| 489 |                     Rxx = (1 - shrink_alpha) * Rxx + shrink_alpha * mu * eye(Nmic, 'like', Rxx); |
| 490 |                 end | 
| 491 |  | 
| 492 |                 Rxx = Rxx + epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx); | 
| 493 |                 w = Rxx \ at; | 求解线性方程组（等价于矩阵求逆后的乘法），用于 MVDR 权重计算。 |
| 494 |                 denom = at' * w; | 
| 495 |                 if abs(denom) &lt; 1e-12 |若分母 $a_t^H w$ 太小（接近 0），就把权重置零，防止爆炸。
| 496 |                     w = zeros(Nmic,1, 'like', at); | 
| 497 |                 else | 
| 498 |                     w = w / denom; |
| 499 |                 end | 
| 500 |                 w_last = w; | 
| 501 |             end | 
| 502 |  | 
| 503 |             Yf_tgt(k,n) = w_last' * Xkf(:,n); | 
| 504 |             Y_tar_tgt(k,n) = w_last' * Xkf_tar(:,n); | 
| 505 |             Y_intnoi_tgt(k,n) = w_last' * Xkf_intnoi(:,n); | 
| 506 |         end | 
| 507 |     else | 两种权重更新模式——快速模式 mvdr_use_time_varying=false
| 508 |         % fast mode: one covariance/weight per frequency bin | 
| 509 |         Rxx = (Xkf_cov * Xkf_cov') / Tframes; | 
| 510 |         if shrink_alpha &gt; 0 | 
| 511 |             mu = trace(Rxx) / Nmic; | 
| 512 |             Rxx = (1 - shrink_alpha) * Rxx + shrink_alpha * mu * eye(Nmic, 'like', Rxx); |
| 513 |         end | 
| 514 |         Rxx = Rxx + epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx); | 
| 515 |         w = Rxx \ at; | 
| 516 |         denom = at' * w; | 
| 517 |         if abs(denom) &lt; 1e-12 | 
| 518 |             w = zeros(Nmic,1, 'like', at); | 
| 519 |         else | 
| 520 |             w = w / denom; |
| 521 |         end | 
| 522 |  | 
| 523 |         Yf_tgt(k,:) = w' * Xkf; | 
| 524 |         Y_tar_tgt(k,:) = w' * Xkf_tar; | 
| 525 |         Y_intnoi_tgt(k,:) = w' * Xkf_intnoi; | 
| 526 |     end | 
| 527 |  | 
| 528 |     if mod(k, mvdr_progress_step) == 0 \|\| k == f_limit | 
| 529 |         fprintf('MVDR target-beam progress: %d/%d bins (%.1f%%), elapsed %.1fs\n', k, f_limit, 100*k/f_limit, toc); | 进度打印：每隔 mvdr_progress_step 个频点打印进度与耗时。
| 530 |     end | 
| 531 | end | 
| 532 |  | 


**两种权重更新模式**  
- 时变模式 mvdr_use_time_varying=true  
  1. 按时间帧循环。
  2. 每隔 mvdr_frame_stride 帧更新一次权重，其他帧复用上次 w_last（降计算量）。
  3. 用局部滑窗 [t1,t2] 估计 Rxx：  
     $$R_{xx}=\frac{X_{loc}X_{loc}^H}{N_{loc}}$$
  4. 可选收缩：  
     $$R_{xx}\leftarrow(1-\alpha)R_{xx}+\alpha\mu I,\ \mu=\frac{\mathrm{tr}(R_{xx})}{Nmic}$$
  5. 对角加载：  
     $$R_{xx}\leftarrow R_{xx}+\epsilon\frac{\mathrm{tr}(R_{xx})}{Nmic}I$$
  6. MVDR 权重：
     $$w=\frac{R_{xx}^{-1}a_t}{a_t^H R_{xx}^{-1}a_t}$$
  7. 用该权重同时作用在混合/目标/干扰噪声分量上，得到
     - Yf_tgt(k,n)
     - Y_tar_tgt(k,n)
     - Y_intnoi_tgt(k,n)

- 快速模式 mvdr_use_time_varying=false  
  1. 每个频点只估计一次全帧协方差。
  2. 只算一个 w，整帧复用。
  3. 同样有收缩、加载、无失真归一化。
  4. 一次性得到该频点全部时间帧输出（矩阵乘法）。


在“目标方向保持无失真约束”下，通过协方差逆来最小化输出功率，从而抑制干扰+噪声，并且同时输出可分解的目标/干扰分量，方便后续算 SNR 增益。


| 行号 | 代码 | 解释 |
|---:|---|---|
| 533 | %% 对每个频率和时间帧，用局部滑窗估计协方差矩阵（干扰指向分支） | 干扰指向MVDR分支
| 534 | for k = f_start:f_limit | 循环初始化与数据提取，与目标指向分支相同的按频率循环结构
| 535 |     Xkf = squeeze(S(k,:,:)).';   % Nmic x Tframes | 
| 536 |     Xkf_interf = squeeze(S_interf(k,:,:)).';      % Nmic x Tframes | 
| 537 |     Xkf_tarnoi = squeeze(S_tarnoi(k,:,:)).';      % Nmic x Tframes | 
| 538 |     if use_oracle_tarnoi_cov | 
| 539 |         Xkf_cov = Xkf_tarnoi; | 
| 540 |     else | 
| 541 |         Xkf_cov = Xkf; | 
| 542 |     end | 
| 543 |     ai = a_interf(:,k); | 
| 544 |     if norm(ai) &lt; 1e-12 | 
| 545 |         continue; | 
| 546 |     end | 
| 547 |  | 
| 548 |     if mvdr_use_time_varying | 
| 549 |         w_last = zeros(Nmic,1, 'like', ai); | 
| 550 |         for n = 1:Tframes | 
| 551 |             if n == 1 \|\| mod(n-1, mvdr_frame_stride) == 0 | 
| 552 |                 t1 = max(1, n - floor(Mavg/2)); | 
| 553 |                 t2 = min(Tframes, n + floor(Mavg/2)); | 
| 554 |                 Xloc = Xkf_cov(:, t1:t2); | 
| 555 |  | 
| 556 |                 Rxx = (Xloc * Xloc') / size(Xloc,2); |
| 557 |  | 
| 558 |                 if shrink_alpha &gt; 0 | 
| 559 |                     mu = trace(Rxx) / Nmic; | 
| 560 |                     Rxx = (1 - shrink_alpha) * Rxx + shrink_alpha * mu * eye(Nmic, 'like', Rxx); | 
| 561 |                 end | 
| 562 |  | 
| 563 |                 Rxx = Rxx + epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx); | 
| 564 |                 w = Rxx \ ai; | 
| 565 |                 denom = ai' * w; |
| 566 |                 if abs(denom) &lt; 1e-12 | 
| 567 |                     w = zeros(Nmic,1, 'like', ai); | 
| 568 |                 else | 
| 569 |                     w = w / denom; | 
| 570 |                 end | 
| 571 |                 w_last = w; | 
| 572 |             end | 
| 573 |  | 
| 574 |             Yf_int(k,n) = w_last' * Xkf(:,n); | 
| 575 |             Y_interf_int(k,n) = w_last' * Xkf_interf(:,n); | 
| 576 |             Y_tarnoi_int(k,n) = w_last' * Xkf_tarnoi(:,n); | 
| 577 |         end | 
| 578 |     else | 
| 579 |         Rxx = (Xkf_cov * Xkf_cov') / Tframes; | 
| 580 |         if shrink_alpha &gt; 0 | 
| 581 |             mu = trace(Rxx) / Nmic; | 
| 582 |             Rxx = (1 - shrink_alpha) * Rxx + shrink_alpha * mu * eye(Nmic, 'like', Rxx); | 
| 583 |         end | 
| 584 |         Rxx = Rxx + epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx); | 
| 585 |         w = Rxx \ ai; | 
| 586 |         denom = ai' * w; | 
| 587 |         if abs(denom) &lt; 1e-12 | 
| 588 |             w = zeros(Nmic,1, 'like', ai); |
| 589 |         else | 
| 590 |             w = w / denom; | 
| 591 |         end | 
| 592 |  |
| 593 |         Yf_int(k,:) = w' * Xkf; |
| 594 |         Y_interf_int(k,:) = w' * Xkf_interf; | 
| 595 |         Y_tarnoi_int(k,:) = w' * Xkf_tarnoi; | 
| 596 |     end | 
| 597 |  | 
| 598 |     if mod(k, mvdr_progress_step) == 0 \|\| k == f_limit | 
| 599 |         fprintf('MVDR interf-beam progress: %d/%d bins (%.1f%%), elapsed %.1fs\n', k, f_limit, 100*k/f_limit, toc); | 
| 600 |     end | 
| 601 | end | 
| 602 | toc | 
| 603 |  | 
| 604 | if gpu_ok | 
| 605 |     % gather once before diagnostics/ISTFT to avoid repeated host-device sync | 
| 606 |     Yf_tgt = gather(Yf_tgt); | 
| 607 |     Y_tar_tgt = gather(Y_tar_tgt); | 
| 608 |     Y_intnoi_tgt = gather(Y_intnoi_tgt); | 
| 609 |     Yf_int = gather(Yf_int); | 
| 610 |     Y_interf_int = gather(Y_interf_int); | 
| 611 |     Y_tarnoi_int = gather(Y_tarnoi_int); | 
| 612 |     S = gather(S); | 
| 613 |     S_interf = gather(S_interf); |
| 614 |     S_intnoi = gather(S_intnoi); | 
| 615 |     S_tarnoi = gather(S_tarnoi); | 
| 616 |     a_target = gather(a_target); | 
| 617 |     a_interf = gather(a_interf); | 
| 618 | end |
| 619 |  | 
| 620 | % legacy aliases (keep downstream diagnostics compatible with original script) | 
| 621 | Yf = Yf_tgt; |
| 622 | Y_tar = Y_tar_tgt; | 
| 623 | Y_intnoi = Y_intnoi_tgt; | 
| 624 |  | 
| 625 |  | 


**干扰指向MVDR分支**  

**1. 循环初始化与数据提取**
```matlab
for k = f_start:f_limit
    Xkf = squeeze(S(k,:,:)).';              % Nmic × Tframes (混合信号)
    Xkf_interf = squeeze(S_interf(k,:,:)).';     % Nmic × Tframes (干扰分量)
    Xkf_tarnoi = squeeze(S_tarnoi(k,:,:)).';     % Nmic × Tframes (目标+噪声)
```
- 与目标指向分支相同的**按频率循环**结构
- 关键差异：这里提取的是不同的分量
  - `Xkf_interf`：**干扰信号的STFT**（而非目标信号）
  - `Xkf_tarnoi`：**目标+噪声的STFT**（而非干扰+噪声）
- 物理含义：干扰指向分支要在**干扰方向最大化增益**，同时**压制目标**

**2. 协方差数据选择（Oracle模式切换）**
```matlab
if use_oracle_tarnoi_cov
    Xkf_cov = Xkf_tarnoi;      % 使用目标+噪声的协方差（诊断上界）
else
    Xkf_cov = Xkf;             % 使用混合信号的协方差（实际应用）
end
```
- **关键切换点**：
  - 目标指向分支用 `use_oracle_intnoi_cov` → 用干扰+噪声估计协方差
  - 干扰指向分支用 `use_oracle_tarnoi_cov` → 用目标+噪声估计协方差
- **诊断用途**：Oracle模式给出性能上界（假设完美知道要压制什么）

**3. 转向向量有效性检查**
```matlab
ai = a_interf(:,k);        % 从a_interf中提取干扰方向的转向向量
if norm(ai) < 1e-12
    continue;              % 跳过无效频率（转向向量为零）
end
```
- 使用 **`a_interf`** 而非 `a_target`（这是与目标分支的核心区别）
- 作用：**指向干扰方向**而非目标方向

**4. 时间变化模式（Time-Varying Mode）**

**权重更新条件**
```matlab
if n == 1 || mod(n-1, mvdr_frame_stride) == 0
    t1 = max(1, n - floor(Mavg/2));
    t2 = min(Tframes, n + floor(Mavg/2));
    Xloc = Xkf_cov(:, t1:t2);
```
- 每隔 `mvdr_frame_stride` 帧更新一次权重
- `Mavg=31`：使用31帧的**滑动窗口**估计协方差
- 目的：**自适应跟踪干扰方向变化**

**协方差矩阵计算与正则化**
```matlab
Rxx = (Xloc * Xloc') / size(Xloc,2);

if shrink_alpha > 0
    mu = trace(Rxx) / Nmic;
    Rxx = (1 - shrink_alpha) * Rxx + shrink_alpha * mu * eye(Nmic, 'like', Rxx);
end

Rxx = Rxx + epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx);
```
- **Shrinkage正则化**：混合样本协方差与恒等矩阵
  - `(1 - shrink_alpha) × Rxx + shrink_alpha × μ × I`
  - 作用：提高数值稳定性，防止奇异矩阵
- **对角加载**：加 `ε × (trace(Rxx)/Nmic) × I`
  - 防止Rxx条件数过高导致求逆失败

**MVDR权重求解**
```matlab
w = Rxx \ ai;              % 求解 Rxx × w = ai
denom = ai' * w;           % 计算分母 a_i^H × Rxx^{-1} × a_i

if abs(denom) < 1e-12
    w = zeros(Nmic,1, 'like', ai);
else
    w = w / denom;         % MVDR权重 w = (Rxx^{-1} × a_i) / (a_i^H × Rxx^{-1} × a_i)
end
w_last = w;
```
- **标准MVDR公式**的干扰指向版本
- 分母接近零时设置权重为零（数值稳定性保护）

**输出计算**
```matlab
Yf_int(k,n) = w_last' * Xkf(:,n);          % 混合信号经干扰方向波束
Y_interf_int(k,n) = w_last' * Xkf_interf(:,n);    % 干扰分量经波束
Y_tarnoi_int(k,n) = w_last' * Xkf_tarnoi(:,n);    % 目标+噪声分量经波束
```
- **三个独立的输出**（对应三个输入分量）
  - `Yf_int`：**总输出**（用于后续ISTFT）
  - `Y_interf_int`：干扰**保留情况**（应该**大**，说明干扰通过）
  - `Y_tarnoi_int`：目标**压制情况**（应该**小**，说明目标被压制）

**5. 快速模式（Fast Mode）**
```matlab
else
    Rxx = (Xkf_cov * Xkf_cov') / Tframes;   % 全帧协方差（单个权重）
    % 相同的正则化和MVDR求解...
    Yf_int(k,:) = w' * Xkf;                % 一个权重应用到所有帧
    Y_interf_int(k,:) = w' * Xkf_interf;
    Y_tarnoi_int(k,:) = w' * Xkf_tarnoi;
end
```
- **每个频率一个权重**（而非每个频率每帧一个）
- 计算效率 **~50倍提升**（批量矩阵乘法）
- 代价：**失去时间自适应性**

**6. 进度报告**
```matlab
if mod(k, mvdr_progress_step) == 0 || k == f_limit
    fprintf('MVDR interf-beam progress: %d/%d bins (%.1f%%), elapsed %.1fs\n', ...);
end
```
- 每隔 `mvdr_progress_step=20` 个频率打印一次进度
- 诊断计算耗时

**7. GPU数据传回主机**
```matlab
if gpu_ok
    Yf_tgt = gather(Yf_tgt);      % 目标指向输出
    Y_tar_tgt = gather(Y_tar_tgt);
    Y_intnoi_tgt = gather(Y_intnoi_tgt);
    
    Yf_int = gather(Yf_int);      % 干扰指向输出
    Y_interf_int = gather(Y_interf_int);
    Y_tarnoi_int = gather(Y_tarnoi_int);
    
    S = gather(S);                 % 所有STFT分量返回CPU
    S_interf = gather(S_interf);
    % ...
end
```
- **批量gather**（而非逐个）：减少GPU→CPU传输开销
- 原因：后续需要ISTFT和性能指标计算（都在CPU上）

**8. 向后兼容性别名**
```matlab
Yf = Yf_tgt;
Y_tar = Y_tar_tgt;
Y_intnoi = Y_intnoi_tgt;
```
- **保持原脚本兼容**：下游诊断代码默认使用目标指向输出
- `Yf_int` 等干扰分支结果需要显式访问

---

**关键对比：目标指向 vs. 干扰指向**

| 维度 | 目标指向 | 干扰指向 |
|------|--------|--------|
| **转向向量** | `a_target` | `a_interf` |
| **协方差数据** | 干扰+噪声 | 目标+噪声 |
| **输出1** | `Yf_tgt` | `Yf_int` |
| **输出2** | `Y_tar_tgt`（目标保留） | `Y_interf_int`（干扰保留） |
| **输出3** | `Y_intnoi_tgt`（干扰压制） | `Y_tarnoi_int`（目标压制） |
| **目标** | 最大化目标，压制干扰 | 最大化干扰，压制目标 |
| **用途** | 主要输出 | 诊断对比 |

---


| 行号 | 代码 | 解释 |
|---:|---|---|
| 626 | %% ========== Diagnostic for 2 kHz and Rxx (place right after MVDR loop, before ISTFT) ========== | 
| 627 | % find freq bin for 2kHz and 1kHz | 
| 628 | [~, k2] = min(abs(F - 2000)); | 
| 629 |  | 
| 630 | % pick a robust frame interval (middle 30% of frames) for Rxx estimate | 
| 631 | numFrames = size(Yf,2); | 
| 632 | t1_diag = max(1, round(numFrames*0.35)); | 
| 633 | t2_diag = min(numFrames, round(numFrames*0.65)); | 
| 634 | if t1_diag &gt;= t2_diag | 
| 635 |     t1_diag = 1; t2_diag = numFrames; | 
| 636 | end | 
| 637 |  | 
| 638 | % get Xkf for k2 (freq x time x channel earlier) | 
| 639 | Xkf_k2 = squeeze(S(k2,:,:)).';   % Nmic x Tframes | 
| 640 | % safe bounds | 
| 641 | t2_diag = min(t2_diag, size(Xkf_k2,2)); | 
| 642 | Xloc_diag = Xkf_k2(:, t1_diag:t2_diag); | 
| 643 | Rxx_diag = (Xloc_diag * Xloc_diag') / size(Xloc_diag,2); |
| 644 | reg_diag = epsilon * trace(Rxx_diag) / Nmic; | 
| 645 | Rxx_diag = Rxx_diag + reg_diag * eye(Nmic, 'like', Rxx_diag); | 
| 646 |  | 
| 647 | % eigenspectrum | 
| 648 | [~, D] = eig(Rxx_diag); | 
| 649 | evals = sort(diag(D), 'descend'); | 
| 650 | figure(3); plot(1:Nmic, 10*log10(evals + eps), '-o'); | 
| 651 | xlabel('Index'); ylabel('Eigenvalue (dB)');  | 
| 652 |  |
| 653 | set(gca, 'LineWidth', 1); | 
| 654 | exportgraphics(gcf, 'Rxx eig-spectrum at 2 kHz.pdf', 'Resolution',300,... | 
| 655 |     'ContentType','image'); | 
| 656 | % title('Rxx eig-spectrum at ~2 kHz'); | 
| 657 |  | 
| 658 | % compute MVDR weight used at k2 for the center frame | 
| 659 | center_frame = round(numFrames/2); | 
| 660 | t1c = max(1, center_frame - floor(Mavg/2)); | 
| 661 | t2c = min(numFrames, center_frame + floor(Mavg/2)); |
| 662 | Xlocc = Xkf_k2(:, t1c:t2c); |
| 663 | Rxxc = (Xlocc * Xlocc') / size(Xlocc,2); | 
| 664 | if shrink_alpha &gt; 0 | 
| 665 |     mu_c = trace(Rxxc) / Nmic; |
| 666 |     Rxxc = (1 - shrink_alpha) * Rxxc + shrink_alpha * mu_c * eye(Nmic, 'like', Rxxc); | 
| 667 | end | 
| 668 | Rxxc = Rxxc + (epsilon * trace(Rxxc) / Nmic) * eye(Nmic, 'like', Rxxc); | 
| 669 |  | 
| 670 | at_k2 = a_target(:, k2); | 
| 671 | w_diag = Rxxc \ at_k2; | 
| 672 | denom_diag = (at_k2' * w_diag); | 
| 673 | if abs(denom_diag) &lt; 1e-12 | 
| 674 |     Wdiag = zeros(size(w_diag)); | 
| 675 | else | 
| 676 |     Wdiag = w_diag ./ denom_diag; | 
| 677 | end | 
| 678 |  | 
| 679 | % beampattern at 2 kHz over azimuth (elev=0) | 
| 680 | azs = -180:1:180; | 
| 681 | resp = zeros(size(azs)); | 
| 682 | for ii = 1:length(azs) | 
| 683 |     d_try = [cosd(0)*cosd(azs(ii)); cosd(0)*sind(azs(ii)); sind(0)]; | 
| 684 |     a_try = exp(-1j*2*pi*F(k2) * ((mic_pos - r_center) * d_try) / c); | 
| 685 |     a_try = a_try / norm(a_try); | 
| 686 |     resp(ii) = 20*log10(abs(Wdiag' * a_try) + eps); | 
| 687 | end | 
| 688 | figure(4); plot(azs, resp); grid on; | 
| 689 | xlabel('Azimuth (deg)'); ylabel('Response (dB)');  |
| 690 | % title(sprintf('Beampattern at %.1f Hz', F(k2))); | 
| 691 |  | 
| 692 | set(gca, 'LineWidth', 1); | 
| 693 | exportgraphics(gcf, 'Beampattern at 2k Hz.pdf', 'Resolution',300,... | 
| 694 |     'ContentType','image'); | 
| 695 | fprintf('Diagnostic done. See eig-spectrum and beampattern for 2 kHz.\n'); | 
| 696 |  | 
| 697 |  |
| 698 |  | 


2 kHz 诊断在做什么、为什么这样做，以及每个变量对应的物理意义。它不是主处理链路，而是一个“诊断模块”：检查 2 kHz 处协方差矩阵与波束图是否合理，帮助判断 MVDR 参数是否稳定。

1. 先定位 2 kHz 频点
- 用 F 这个频率轴，找最接近 2000 Hz 的 bin 索引 k2。
- 这样后面所有计算都聚焦在这个频率点上做“切片诊断”。

2. 选中间 30% 时间帧估计诊断协方差
- numFrames 是时帧总数。
- t1_diag 到 t2_diag 取中间 30% 的帧（约 35%~65%）。
- 目的：避免开头/结尾不稳定片段，让协方差估计更稳健。

- Xkf_k2 取出 2 kHz 处所有麦克风、所有帧的 STFT 数据，维度是 Nmic × Tframes。
- Xloc_diag 是中间时间窗内的数据。
- 协方差估计：
  $$R_{xx}=\frac{X X^H}{N}$$
- 再加对角加载：
  $$R_{xx}\leftarrow R_{xx}+\epsilon\cdot\frac{\mathrm{tr}(R_{xx})}{N_{mic}}I$$
  这里 reg_diag 就是加载量，目的是防止病态或接近奇异。

3. 画协方差特征值谱（eigenspectrum）
- 对 Rxx_diag 做特征值分解，按降序排列后转 dB 画图。
- 这张图用于看“能量主子空间”和“噪声子空间”分离是否明显。
- 如果特征值掉得很快，通常说明阵列在该频点可分性较好；如果全都挤在一起，MVDR 抑制能力可能受限。

4. 在中心帧附近重算一次 2 kHz 的 MVDR 权重
- 选 center_frame 附近 Mavg 帧，重新估计一个 Rxxc（更贴近主循环局部估计方式）。
- 若 shrink_alpha > 0，做收缩正则化。
- 再做对角加载。
- 用目标导向向量 at_k2 求权重：
  $$w=\frac{R_{xx}^{-1}a}{a^H R_{xx}^{-1} a}$$
- 分母过小就置零，避免数值爆炸。

这一步的意义是：拿到一个“代表性权重”Wdiag，用它来画方向图，而不是直接用某个缓存结果。

5. 扫描方位角，画 2 kHz 波束图
- 方位角从 -180 到 180 度，仰角固定 0（水平面）。
- 每个角度构造试探导向向量 a_try。
- 计算响应：
  $$20\log_{10}\left|w^H a(\theta)\right|$$
- 最后画出 Beampattern at 2k Hz。

这张图直观展示：
- 主瓣是否对准目标方向。
- 旁瓣是否过高。
- 是否在干扰方向附近形成凹口（null）。

6. 这段诊断的定位
- 它不直接改变最终输出信号。
- 它用于解释“为什么这个 MVDR 配置有效或失效”。
- 常见用途：
  1. 调 epsilon（加载强度）后看特征值谱变化。
  2. 调 shrink_alpha 后看波束图主瓣/旁瓣变化。
  3. 对比不同阵列几何在 2 kHz 的可分辨能力。

| 行号 | 代码 | 解释 |
|---:|---|---|
| 699 | %% ========== Additional Diagnostic for 4 kHz (interference frequency) ========== | 
| 700 | % find freq bin for 4kHz | 
| 701 | [~, k4] = min(abs(F - 4000)); | 
| 702 | center_frame = round(numFrames/2); | 
| 703 | t1c_4k = max(1, center_frame - floor(Mavg/2)); | 
| 704 | t2c_4k = min(numFrames, center_frame + floor(Mavg/2)); |
| 705 |  | 
| 706 | % Use interference+noise covariance if oracle mode is enabled (consistent with MVDR loop) |
| 707 | if use_oracle_intnoi_cov |
| 708 |     Xkf_k4 = squeeze(S_intnoi(k4,:,:)).';   % Nmic x Tframes (interference + noise only) | 
| 709 | else | 
| 710 |     Xkf_k4 = squeeze(S(k4,:,:)).';          % Nmic x Tframes (total signal) | 
| 711 | end |
| 712 |  | 
| 713 | t2c_4k = min(t2c_4k, size(Xkf_k4,2)); |
| 714 | Xlocc_4k = Xkf_k4(:, t1c_4k:t2c_4k); |
| 715 | Rxxc_4k = (Xlocc_4k * Xlocc_4k') / size(Xlocc_4k,2); |
| 716 |  | 
| 717 | if shrink_alpha &gt; 0 | 
| 718 |     mu_c_4k = trace(Rxxc_4k) / Nmic; |
| 719 |     Rxxc_4k = (1 - shrink_alpha) * Rxxc_4k + shrink_alpha * mu_c_4k * eye(Nmic, 'like', Rxxc_4k); | 
| 720 | end | 
| 721 |  | 
| 722 | % Try smaller diagonal loading for better suppression |
| 723 | epsilon_4k = epsilon * 0.1;  % Reduce loading by 10x for diagnostic |
| 724 | Rxxc_4k = Rxxc_4k + (epsilon_4k * trace(Rxxc_4k) / Nmic) * eye(Nmic, 'like', Rxxc_4k); |
| 725 |  |
| 726 | at_k4 = a_target(:, k4); |
| 727 |  | 
| 728 | % Check condition number to diagnose ill-conditioning |
| 729 | cond_num = cond(Rxxc_4k); |
| 730 | fprintf('\n===== 4 kHz Covariance Diagnostics =====\n'); |
| 731 | fprintf('Condition number: %.2e\n', cond_num); |
| 732 | if use_oracle_intnoi_cov | 
| 733 |     fprintf('Using interference+noise covariance\n'); |
| 734 | else |
| 735 |     fprintf('Using total signal covariance\n'); |
| 736 | end | 
| 737 | fprintf('Diagonal loading: %.2e\n', epsilon_4k * trace(Rxxc_4k) / Nmic); |
| 738 |  |
| 739 | w_diag_4k = Rxxc_4k \ at_k4; |
| 740 | denom_diag_4k = (at_k4' * w_diag_4k); |
| 741 | if abs(denom_diag_4k) &lt; 1e-12 |
| 742 |     Wdiag_4k = zeros(size(w_diag_4k)); |
| 743 |     fprintf('Warning: MVDR weight computation failed (denominator too small)\n'); | 
| 744 | else |
| 745 |     Wdiag_4k = w_diag_4k ./ denom_diag_4k; |
| 746 | end | 
| 747 |  |
| 748 | % Check beamformer response in target and interference directions | 
| 749 | resp_target_check = 20*log10(abs(Wdiag_4k' * at_k4) + eps); | 
| 750 | ai_k4 = a_interf(:, k4); |
| 751 | resp_interf_check = 20*log10(abs(Wdiag_4k' * ai_k4) + eps); | 
| 752 | fprintf('Beam response at target DOA:  %.2f dB\n', resp_target_check); | 
| 753 | fprintf('Beam response at interferer DOA: %.2f dB\n', resp_interf_check); | 
| 754 | fprintf('Suppression (target - interf): %.2f dB\n', resp_target_check - resp_interf_check); | 
| 755 | fprintf('========================================\n'); | 
| 756 |  | 
| 757 | % beampattern at 4 kHz over azimuth | 
| 758 | azs_4k = -180:1:180; | 
| 759 | resp_4k = zeros(size(azs_4k)); | 
| 760 | for ii = 1:length(azs_4k) |
| 761 |     d_try = [cosd(0)*cosd(azs_4k(ii)); cosd(0)*sind(azs_4k(ii)); sind(0)]; | 
| 762 |     a_try = exp(-1j*2*pi*F(k4) * ((mic_pos - r_center) * d_try) / c); | 
| 763 |     a_try = a_try / norm(a_try); | 
| 764 |     resp_4k(ii) = 20*log10(abs(Wdiag_4k' * a_try) + eps); | 
| 765 | end | 
| 766 |  | 
| 767 | figure(5); plot(azs_4k, resp_4k); grid on; | 
| 768 | xlabel('Azimuth (deg)'); ylabel('Response (dB)');  | 
| 769 | title(sprintf('Beampattern at %.1f Hz (Interference)', F(k4))); |
| 770 |  | 
| 771 | % Mark target and interference directions | 
| 772 | [~, az_target] = min(abs(azs_4k - atan2d(u_target(2), u_target(1)))); | 
| 773 | [~, az_interf] = min(abs(azs_4k - atan2d(u_interf(2), u_interf(1)))); | 
| 774 | hold on; | 
| 775 | plot(azs_4k(az_target), resp_4k(az_target), 'r^', 'MarkerSize', 10, 'LineWidth', 2); | 
| 776 | plot(azs_4k(az_interf), resp_4k(az_interf), 'mv', 'MarkerSize', 10, 'LineWidth', 2); | 
| 777 | legend({'Beam response', 'Target direction', 'Interference direction'}); | 
| 778 |  | 
| 779 | set(gca, 'LineWidth', 1); | 
| 780 | exportgraphics(gcf, 'Beampattern at 4k Hz.pdf', 'Resolution',300,... | 
| 781 |     'ContentType','image'); | 
| 782 |  |
| 783 | fprintf('4 kHz diagnostic complete. Check beampattern to verify interference suppression.\n'); | 
| 784 |  | 

4 kHz 诊断模块，核心是在“干扰频点”检查协方差条件数、MVDR权重稳定性和方向图抑制效果。作用是专门在干扰主频附近检查 MVDR 是否数值稳定、是否真的把干扰方向压下去。

1. 选定 4 kHz 频点与时间窗  
先找最接近 4000 Hz 的频率 bin: k4。  
然后围绕中心帧取一个长度约 Mavg 的局部时间窗 t1c_4k 到 t2c_4k，用于局部协方差估计。

2. 选择用于协方差估计的数据  
- 若 use_oracle_intnoi_cov=true，用 S_intnoi（干扰+噪声）估协方差。
- 否则用 S（总混合）估协方差。  
这和主 MVDR 循环保持一致，便于诊断结果可对照主流程。

3. 估计并正则化协方差矩阵  
- 样本协方差：$R_{xx}=\frac{X X^H}{N}$  
- shrinkage（若开启）：把 $R_{xx}$ 向 $\mu I$ 收缩，降低病态风险。  
- 对角加载：这里特意用 epsilon_4k = 0.1 * epsilon（更小加载）做诊断，观察抑制能力是否提升。  
这一步本质是在“稳定性”和“可抑制度”之间找平衡。

4. 条件数诊断  
用 cond(Rxxc_4k) 输出条件数：  
- 条件数很大 -> 矩阵接近奇异，权重容易不稳定。  
- 条件数适中 -> 权重更可信。  
同时打印实际加载量，便于你调 epsilon 时有量化依据。

5. 计算 4 kHz MVDR 权重并做保护  
- 用目标导向向量 at_k4 求权重  
  $w=\frac{R^{-1}a}{a^H R^{-1}a}$  
- 如果分母很小（<1e-12），权重置零并报警，防止数值爆炸。

6. 定点方向响应检查（最关键）  
直接算两个方向的响应：  
- resp_target_check：目标方向响应  
- resp_interf_check：干扰方向响应  
并打印 Suppression = target - interf。  
如果 Suppression 为正且较大，说明在 4 kHz 目标方向增益高于干扰方向，干扰被有效压制。

7. 扫描全方位角并画 4 kHz 方向图  
- 方位角 -180 到 180 度逐点构造导向向量 a_try。  
- 计算响应 $20\log_{10}|w^H a(\theta)|$ 得到完整波束图。  
- 图中再标注目标方向和干扰方向（红三角/紫倒三角），便于视觉验证主瓣和零陷位置。  
最终导出 Beampattern at 4k Hz.pdf。

这段代码是在“干扰最敏感频点”做体检，先看矩阵是否健康（条件数），再看权重是否可靠（分母保护），最后看结果是否正确（目标方向高、干扰方向低）。


| 行号 | 代码 | 解释 |
|---:|---|---|
| 785 | %% ========== 9. ISTFT and normalization ========== | 
| 786 | y_mvdr_target = real(istft(Yf_tgt, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft)); | 
| 787 | y_mvdr_interf = real(istft(Yf_int, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft)); | 
| 788 |  | 
| 789 | % robust length alignment   调整输出长度与输入相同 | 
| 790 | if length(y_mvdr_target) &gt;= Nt | 
| 791 |     y_mvdr_target = y_mvdr_target(1:Nt); | 
| 792 | else | 
| 793 |     y_mvdr_target = [y_mvdr_target; zeros(Nt - length(y_mvdr_target), 1)]; | 
| 794 | end | 
| 795 | if length(y_mvdr_interf) &gt;= Nt | 
| 796 |     y_mvdr_interf = y_mvdr_interf(1:Nt); | 
| 797 | else | 
| 798 |     y_mvdr_interf = [y_mvdr_interf; zeros(Nt - length(y_mvdr_interf), 1)]; | 
| 799 | end | 
| 800 |  | 
| 801 | % energy normalization (avoid clipping) | 
| 802 | y_mvdr_target = y_mvdr_target / max(abs(y_mvdr_target) + 1e-12); | 
| 803 | y_mvdr_interf = y_mvdr_interf / max(abs(y_mvdr_interf) + 1e-12); | 
| 804 |  | 
| 805 | % legacy alias | 
| 806 | y_mvdr = y_mvdr_target; | 
| 807 |  | 
| 808 |  | 

“频域结果回到时域 + 输出安全化”。
1. ISTFT 反变换  
在 mvdr_of_real_signal_based_on_rir.m 和 mvdr_of_real_signal_based_on_rir.m：
- 把目标指向分支的频域结果 Yf_tgt 变回时域，得到 y_mvdr_target。
- 把干扰指向分支的频域结果 Yf_int 变回时域，得到 y_mvdr_interf。
- 用的窗函数、重叠、FFT长度与前面 STFT 一致，这样重建才匹配。

2. 长度对齐（鲁棒处理）  
在 mvdr_of_real_signal_based_on_rir.m 到 mvdr_of_real_signal_based_on_rir.m：
- 目标是让输出长度和原始信号长度 Nt 一致。
- 如果 ISTFT 结果比 Nt 长，就截断到 Nt。
- 如果比 Nt 短，就在尾部补零到 Nt。  
这样后面画图、算功率、做指标比较时不会出现维度不一致错误。

3. 幅值归一化（防削顶）  
在 mvdr_of_real_signal_based_on_rir.m 和 mvdr_of_real_signal_based_on_rir.m：
- 每个输出都除以自身最大绝对值，让峰值接近 1。
- 加 1e-12 是防止分母为 0。  
作用：避免后续播放/导出音频时削顶。

4. 兼容旧变量名  
在 mvdr_of_real_signal_based_on_rir.m：
- 把 y_mvdr 指向 y_mvdr_target。
- 这是兼容老代码的写法，后续如果仍用 y_mvdr，就默认代表“目标指向输出”。

一句话总结：这段就是把双分支 MVDR 的频域输出安全地还原成可用时域信号，并统一长度与幅值，方便后续指标计算和可视化。


| 行号 | 代码 | 解释 |
|---:|---|---|
| 809 | %% ========== ISTFT for metric evaluation ========== | 
| 810 | y_tar_out_target = real(istft(Y_tar_tgt, fs, ... | 
| 811 |     'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft)); | 
| 812 | y_intnoi_out_target = real(istft(Y_intnoi_tgt, fs, ... | 
| 813 |     'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft)); | 
| 814 |  |
| 815 | y_interf_out_interf = real(istft(Y_interf_int, fs, ... | 
| 816 |     'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft)); | 
| 817 | y_tarnoi_out_interf = real(istft(Y_tarnoi_int, fs, ... | 
| 818 |     'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft)); | 
| 819 |  | 


“用于指标计算的分量级 ISTFT 还原”。和上一段“还原最终输出波形”不同，这里专门还原各个组成分量，方便后面单独算功率比（SNR/ISR）。

1. 目标指向分支的两路分量还原  
- y_tar_out_target：由 Y_tar_tgt 反变换得到，表示“目标分量经过目标波束后的时域信号”
- y_intnoi_out_target：由 Y_intnoi_tgt 反变换得到，表示“干扰+噪声分量经过目标波束后的时域信号”

后续会用它们计算目标波束输出信噪比：
$$
\mathrm{SNR}_{out}=10\log_{10}\frac{P_{tar,out}}{P_{int+noise,out}}
$$

2. 干扰指向分支的两路分量还原  
- y_interf_out_interf：由 Y_interf_int 反变换，表示“干扰分量经过干扰波束后的时域信号”
- y_tarnoi_out_interf：由 Y_tarnoi_int 反变换，表示“目标+噪声分量经过干扰波束后的时域信号”

后续会用它们计算干扰波束输出比值（ISR）：
$$
\mathrm{ISR}_{out}=10\log_{10}\frac{P_{interf,out}}{P_{tar+noise,out}}
$$

3. 为什么要单独做这 4 次 ISTFT  
- 不是为了听音，而是为了“可解释评估”
- 把总输出拆成可量化的分子/分母，避免只看总波形时混淆贡献来源
- 与前面双分支设计一一对应，能分别评估“保目标能力”和“保干扰能力”

4. 细节点  
- 每次都用和 STFT 一致的 win/noverlap/Nfft，保证重建一致性  
- 外面包 real(...) 是因为数值误差可能产生极小虚部，取实部更稳妥


| 行号 | 代码 | 解释 |
|---:|---|---|
| 820 | %% ===== Length alignment (SAFE) ===== | 
| 821 | Lsig = min([length(y_mvdr_target), length(y_mvdr_interf), ... | 
| 822 |     length(y_tar_out_target), length(y_intnoi_out_target), ... | 
| 823 |     length(y_interf_out_interf), length(y_tarnoi_out_interf)]); | 
| 824 |  | 
| 825 | y_mvdr_target = y_mvdr_target(1:Lsig); | 
| 826 | y_mvdr_interf = y_mvdr_interf(1:Lsig); |
| 827 | y_mvdr = y_mvdr(1:Lsig); | 
| 828 |  | 
| 829 | y_tar_out_target = y_tar_out_target(1:Lsig); | 
| 830 | y_intnoi_out_target = y_intnoi_out_target(1:Lsig); |
| 831 | y_interf_out_interf = y_interf_out_interf(1:Lsig); | 
| 832 | y_tarnoi_out_interf = y_tarnoi_out_interf(1:Lsig); |
| 833 |  | 
| 834 |  | 


“统一所有时域信号长度”，目的是保证后续功率计算和指标比较在同一时间范围内进行。

1. 先找共同长度 Lsig  
代码取六路信号长度的最小值：
- 目标波束总输出
- 干扰波束总输出
- 目标波束下的目标分量
- 目标波束下的干扰噪声分量
- 干扰波束下的干扰分量
- 干扰波束下的目标噪声分量

也就是：
$$
L_{sig}=\min(L_1,L_2,\dots,L_6)
$$

2. 全部裁剪到 Lsig  
每一路都统一做前 Lsig 样本截取。  
这样可避免后面出现：
- 向量长度不一致报错
- 分子分母取样区间不同导致指标失真

3. 为什么叫 SAFE  
因为 ISTFT 后不同分量可能出现轻微长度差（窗函数、边界处理、数值路径差异导致）。这一步用“取最短并统一截断”的方式，保证后续 SNR、ISR 计算是可比且稳定的。

这段是在做“评估前的最后对齐”，保证所有输出分量在同一长度、同一时间段上进行公平比较。


| 行号 | 代码 | 解释 |
|---:|---|---|
| 835 | %% ========== Target-steered MVDR performance ========== | 
| 836 | P_tar_out = mean(y_tar_out_target.^2); | 
| 837 | P_intnoi_out = mean(y_intnoi_out_target.^2); | 
| 838 |  | 
| 839 | SNR_out_dB = 10*log10(P_tar_out / P_intnoi_out); | 
| 840 | SNR_gain_dB = SNR_out_dB - SNR_in_dB; |
| 841 |  | 
| 842 | fprintf('\n===== TARGET-STEERED MVDR =====\n'); | 
| 843 | fprintf('Input  SNR (mic %d, target/(interf+noise)): %.2f dB\n', ref_mic, SNR_in_dB); | 
| 844 | fprintf('Output SNR (target beam)                  : %.2f dB\n', SNR_out_dB); | 
| 845 | fprintf('SNR Improvement                           : %.2f dB\n', SNR_gain_dB); | 
| 846 | fprintf('================================\n'); | 
| 847 |  | 
| 848 |  | 



计算并打印“目标指向波束”的性能指标。
1. 先算输出端两部分功率  
- P_tar_out = mean(y_tar_out_target.^2)  
  目标分量在目标波束输出中的平均功率
- P_intnoi_out = mean(y_intnoi_out_target.^2)  
  干扰+噪声分量在目标波束输出中的平均功率

2. 计算输出 SNR（严格说是 SINR）  
$$
SNR_{out,dB}=10\log_{10}\left(\frac{P_{tar,out}}{P_{int+noise,out}}\right)
$$
因为分母是“干扰+噪声”，本质更接近 SINR，但代码延续命名为 SNR_out_dB。

3. 计算提升量（增益）  
$$
SNR_{gain,dB}=SNR_{out,dB}-SNR_{in,dB}
$$
- 若 > 0：目标波束有净提升  
- 若 < 0：波束处理后反而变差

4. 打印结果用于快速诊断  
- 打印输入参考值（mic1 的基线）
- 打印输出值（目标波束）
- 打印提升量（最关键）

一句话总结：这段代码就是把“目标波束是否有效”量化成一个可对比的 dB 指标，并与输入基线做差得到最终提升值。

| 行号 | 代码 | 解释 |
|---:|---|---|
| 849 | %% ========== Interference-steered MVDR performance ========== | 
| 850 | P_interf_in = mean(X_interf(:, ref_mic).^2); | 
| 851 | P_tarnoi_in = mean((X_target(:, ref_mic) + noise(:, ref_mic)).^2); | 
| 852 | ISR_in_dB = 10*log10(P_interf_in / P_tarnoi_in); | 
| 853 |  | 
| 854 | P_interf_out = mean(y_interf_out_interf.^2); | 
| 855 | P_tarnoi_out = mean(y_tarnoi_out_interf.^2); |
| 856 | ISR_out_dB = 10*log10(P_interf_out / P_tarnoi_out); | 
| 857 | ISR_gain_dB = ISR_out_dB - ISR_in_dB; | 
| 858 |  | 
| 859 | fprintf('\n===== INTERFERENCE-STEERED MVDR =====\n'); | 
| 860 | fprintf('Input  ISR (mic %d, interf/(target+noise)): %.2f dB\n', ref_mic, ISR_in_dB); | 
| 861 | fprintf('Output ISR (interference beam)           : %.2f dB\n', ISR_out_dB); | 
| 862 | fprintf('ISR Improvement                          : %.2f dB\n', ISR_gain_dB); | 
| 863 | fprintf('========================================\n'); | 
| 864 |  | 
| 865 |  | 



评估“干扰指向分支”的性能，和前面的目标指向指标是对称的一套。

1. 先算输入端 ISR 基线  
- P_interf_in：参考麦克风 ref_mic 上“纯干扰”功率  
- P_tarnoi_in：参考麦克风上“目标+噪声”功率  
- ISR_in_dB：
$$
ISR_{in}=10\log_{10}\frac{P_{interf,in}}{P_{tar+noise,in}}
$$
这里 ISR 表示 Interference-to-(Target+Noise) Ratio。

2. 计算输出端 ISR  
- P_interf_out：干扰指向波束输出中的干扰分量功率  
- P_tarnoi_out：干扰指向波束输出中的目标+噪声分量功率  
- ISR_out_dB：
$$
ISR_{out}=10\log_{10}\frac{P_{interf,out}}{P_{tar+noise,out}}
$$

3. 计算提升量  
$$
ISR_{gain}=ISR_{out}-ISR_{in}
$$
- ISR_gain_dB > 0：说明干扰指向分支更“偏向干扰”，即干扰相对目标+噪声的占比提升  
- 这个分支主要用于对照诊断，不是为了增强目标语音

4. 打印结果  
- 输出输入 ISR、输出 ISR、ISR 改善量  
- 用于和“目标指向分支”的 SNR 改善一起看，验证双分支逻辑是否一致

一句话总结：这段代码量化了“干扰波束是否真的更聚焦干扰、抑制目标+噪声”，是双分支 MVDR 设计中的反向验证指标。




| 行号 | 代码 | 解释 |
|---:|---|---|
| 866 | %% ========== Improvement visualization ========== | 
| 867 | figure(6); | 
| 868 | bar([SNR_in_dB, SNR_out_dB; ISR_in_dB, ISR_out_dB]); | 
| 869 | set(gca,'XTickLabel',{'Target beam','Interference beam'}); | 
| 870 | ylabel('Ratio (dB)'); | 
| 871 | legend({'Before MVDR','After MVDR'}, 'Location', 'best'); | 
| 872 | grid on; | 
| 873 |  | 
| 874 | set(gca, 'LineWidth', 1); | 
| 875 | exportgraphics(gcf, 'MVDR Ratio Improvement (Dual Beam).pdf', 'Resolution',300,... | 
| 876 |     'ContentType','image'); | 
| 877 |  | 
| 878 |  | 

做“前后性能对比柱状图”，把目标分支和干扰分支的改进结果放在一张图里。

1. 生成图和数据组织  
- 建立 figure(6)
- bar 输入是一个 2×2 矩阵  
  第一行：[SNR_in_dB, SNR_out_dB]（目标指向）  
  第二行：[ISR_in_dB, ISR_out_dB]（干扰指向）  
含义是每个分支都对比 Before vs After。

2. 坐标与图例  
- x 轴两类：Target beam、Interference beam
- y 轴单位：dB
- 图例说明两根柱子分别是 Before MVDR 和 After MVDR
- grid on 便于读数

3. 美化与导出  
- 设置坐标轴线宽
- 导出为高分辨率 PDF：MVDR Ratio Improvement (Dual Beam).pdf（300 dpi）

一句话总结：这段代码把数值结果可视化成“前后对比图”，用于直观看目标分支的 SNR 提升和干扰分支的 ISR 提升。



| 行号 | 代码 | 解释 |
|---:|---|---|
| 879 | % ========== 10. Diagnostics: waveforms, spectrograms, PSD ========== | 
| 880 | figure(7); |
| 881 | t1 = (0:length(X_target(:,1))-1)/fs; |
| 882 | plot(t1, X_target(:,1)); | 
| 883 | xlabel('Time (s)'); | 
| 884 | ylabel('Amplitude'); | 
| 885 | set(gca, 'LineWidth', 1); | 
| 886 | exportgraphics(gcf, 'Target signal.pdf', ... | 
| 887 |     'ContentType','image'); | 
| 888 |  | 
| 889 | figure(8); | 
| 890 | t2 = (0:length(X_noisy(:,1))-1)/fs; | 
| 891 | plot(t2, X_noisy(:,1)); | 
| 892 | xlabel('Time (s)'); | 
| 893 | ylabel('Amplitude'); | 
| 894 | set(gca, 'LineWidth', 1); | 
| 895 | exportgraphics(gcf, 'Noisy signal.pdf', ... | 
| 896 |     'ContentType','image'); | 
| 897 |  | 
| 898 | figure(9); | 
| 899 | t3 = (0:length(y_mvdr_target)-1)/fs; | 
| 900 | plot(t3, y_mvdr_target, 'b'); hold on; | 
| 901 | plot(t3, y_mvdr_interf, 'm'); | 
| 902 | xlabel('Time (s)'); | 
| 903 | ylabel('Amplitude'); | 
| 904 | legend({'Target-steered output','Interference-steered output'}); | 
| 905 | set(gca, 'LineWidth', 1); | 
| 906 | exportgraphics(gcf, 'MVDR dual outputs.pdf', ... | 
| 907 |     'ContentType','image'); | 
| 908 |  | 
| 909 |  | 
| 910 | figure(10); | 
| 911 | spectrogram(X_target(:,1), hamming(256), 128, 512, fs, 'yaxis'); | 
| 912 | xlabel('Time (ms)'); | 
| 913 | ylabel('Frequency(kHz)'); | 
| 914 | c = colorbar; | 
| 915 | c.Label.String = 'Power/Frequency (dB/Hz)'; | 
| 916 | set(gca, 'LineWidth', 1); | 
| 917 | exportgraphics(gcf, 'Target spectrogram.pdf', 'Resolution',300,... | 
| 918 |     'ContentType','image'); | 
| 919 |  | 
| 920 | figure(11); | 
| 921 | spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis'); | 
| 922 | xlabel('Time (ms)'); | 
| 923 | ylabel('Frequency(kHz)'); | 
| 924 | c = colorbar; | 
| 925 | c.Label.String = 'Power/Frequency (dB/Hz)'; | 
| 926 | set(gca, 'LineWidth', 1); | 
| 927 | exportgraphics(gcf, 'Noisy spectrogram.pdf', 'Resolution',300, ... | 
| 928 |     'ContentType','image'); | 
| 929 |  | 
| 930 | figure(12); |
| 931 | spectrogram(y_mvdr_target, hamming(256), 128, 512, fs, 'yaxis'); | 
| 932 | xlabel('Time (ms)'); | 
| 933 | ylabel('Frequency(kHz)'); | 
| 934 | c = colorbar; | 
| 936 | set(gca, 'LineWidth', 1); | 
| 937 | exportgraphics(gcf, 'MVDR target-beam spectrogram.pdf', 'Resolution',300,... | 
| 938 |     'ContentType','image'); | 
| 939 |  | 
| 940 | figure(13); | 
| 941 | spectrogram(y_mvdr_interf, hamming(256), 128, 512, fs, 'yaxis'); | 
| 942 | xlabel('Time (ms)'); | 
| 943 | ylabel('Frequency(kHz)'); | 
| 944 | c = colorbar; | 
| 945 | c.Label.String = 'Power/Frequency (dB/Hz)'; | 
| 946 | set(gca, 'LineWidth', 1); | 
| 947 | exportgraphics(gcf, 'MVDR interference-beam spectrogram.pdf', 'Resolution',300,... | 
| 948 |     'ContentType','image'); | 
| 949 |  |
| 950 | % 处理前后频谱图（目标指向） | 
| 951 | figure(14); clf; | 
| 952 | subplot(1,2,1); | 
| 953 | spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis'); | 
| 954 | title('Before MVDR (mic1)'); | 
| 955 | xlabel('Time (ms)'); ylabel('Frequency(kHz)'); | 
| 956 | subplot(1,2,2); | 
| 957 | spectrogram(y_mvdr_target, hamming(256), 128, 512, fs, 'yaxis'); | 
| 958 | title('After MVDR (Target-steered)'); | 
| 959 | xlabel('Time (ms)'); ylabel('Frequency(kHz)'); |
| 960 | ax17 = findall(gcf, 'Type', 'Axes'); |
| 961 | clim17 = max(cell2mat(arrayfun(@(ax) ax.CLim, ax17, 'UniformOutput', false)), [], 1); | 
| 962 | arrayfun(@(ax) caxis(ax, clim17), ax17); | 
| 963 | set(gca, 'LineWidth', 1); | 
| 964 | exportgraphics(gcf, 'Before-After Spectrogram (Target Beam).pdf', 'Resolution',300,... | 
| 965 |     'ContentType','image'); |
| 966 |  | 
| 967 | % 处理前后频谱图（干扰指向） | 
| 968 | figure(15); clf; | 
| 969 | subplot(1,2,1); | 
| 970 | spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis'); | 
| 971 | title('Before MVDR (mic1)'); | 
| 972 | xlabel('Time (ms)'); ylabel('Frequency(kHz)'); | 
| 973 | subplot(1,2,2); | 
| 974 | spectrogram(y_mvdr_interf, hamming(256), 128, 512, fs, 'yaxis'); |
| 975 | title('After MVDR (Interference-steered)'); | 
| 976 | xlabel('Time (ms)'); ylabel('Frequency(kHz)'); | 
| 977 | ax18 = findall(gcf, 'Type', 'Axes'); |
| 978 | clim18 = max(cell2mat(arrayfun(@(ax) ax.CLim, ax18, 'UniformOutput', false)), [], 1); | 
| 979 | arrayfun(@(ax) caxis(ax, clim18), ax18); | 
| 980 | set(gca, 'LineWidth', 1); | 
| 981 | exportgraphics(gcf, 'Before-After Spectrogram (Interference Beam).pdf', 'Resolution',300,... | 
| 982 |     'ContentType','image'); | 
| 983 |  | 
| 984 |  | 
| 985 | % PSD comparison | 
| 986 | nfft_psd = 4096; | 
| 987 | [Pxx_in, Fp] = pwelch(X_noisy(:,1), hann(1024), 512, nfft_psd, fs); | 
| 988 | [Pxx_out_target, ~] = pwelch(y_mvdr_target, hann(1024), 512, nfft_psd, fs); | 
| 989 | [Pxx_out_interf, ~] = pwelch(y_mvdr_interf, hann(1024), 512, nfft_psd, fs); | 
| 990 |  | 
| 991 | figure(16); | 
| 992 | plot(Fp, 10*log10(Pxx_in + eps), 'k'); hold on; | 
| 993 | plot(Fp, 10*log10(Pxx_out_target + eps), 'b', 'LineWidth', 1.5); | 
| 994 | plot(Fp, 10*log10(Pxx_out_interf + eps), 'm', 'LineWidth', 1.5); | 
| 995 | xlim([0 fs/2]); | 
| 996 | xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)'); | 
| 997 | legend({'Before MVDR','After Target-steered','After Interference-steered'}); | 
| 998 | grid on; | 
| 999 | set(gca, 'LineWidth', 1); | 
| 1000 | exportgraphics(gcf, 'PSD Fullband Dual MVDR.pdf', 'Resolution',300,... | 
| 1001 |     'ContentType','image'); | 
| 1002 |  | 
| 1003 | figure(17); clf; | 
| 1004 | plot(Fp, 10*log10(Pxx_in + eps), 'k'); hold on; | 
| 1005 | plot(Fp, 10*log10(Pxx_out_target + eps), 'b', 'LineWidth', 1.5); | 
| 1006 | xlim([1800 2200]); | 
| 1007 | xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)'); | 
| 1008 | legend({'Before MVDR','After Target-steered'}); | 
| 1009 | set(gca, 'LineWidth', 1); | 
| 1010 | exportgraphics(gcf, 'PSD around 2 kHz (Target Beam).pdf', 'Resolution',300,... | 
| 1011 |     'ContentType','image'); | 
| 1012 |  | 
| 1013 | figure(18); clf; | 
| 1014 | plot(Fp, 10*log10(Pxx_in + eps), 'k'); hold on; | 
| 1015 | plot(Fp, 10*log10(Pxx_out_target + eps), 'b', 'LineWidth', 1.5); | 
| 1016 | plot(Fp, 10*log10(Pxx_out_interf + eps), 'm', 'LineWidth', 1.5); | 
| 1017 | xlim([3500 4500]); | 
| 1018 | xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)'); | 
| 1019 | legend({'Before MVDR','After Target-steered','After Interference-steered'}); | 
| 1020 | title('PSD around 4 kHz'); | 
| 1021 | grid on; | 
| 1022 |  | 
| 1023 | [~, idx_4k] = min(abs(Fp - 4000)); | 
| 1024 | psd_in_4k = 10*log10(Pxx_in(idx_4k) + eps); | 
| 1025 | psd_tgt_4k = 10*log10(Pxx_out_target(idx_4k) + eps); | 
| 1026 | psd_int_4k = 10*log10(Pxx_out_interf(idx_4k) + eps); | 
| 1027 |  | 
| 1028 | fprintf('\n===== 4 kHz PSD CHECK =====\n'); | 
| 1029 | fprintf('Before MVDR (mic1): %.2f dB/Hz\n', psd_in_4k); | 
| 1030 | fprintf('After Target-steered: %.2f dB/Hz\n', psd_tgt_4k); | 
| 1031 | fprintf('After Interference-steered: %.2f dB/Hz\n', psd_int_4k); | 
| 1032 | fprintf('============================\n'); | 
| 1033 |  | 
| 1034 | set(gca, 'LineWidth', 1); | 
| 1035 | exportgraphics(gcf, 'PSD around 4 kHz (Dual Beam).pdf', 'Resolution',300,... | 
| 1036 |     'ContentType','image'); | 


“结果诊断与可视化总模块”，核心目标是把双分支 MVDR 的效果从时域、时频域、功率谱三个层面展示出来，并导出论文级图片。

1. 时域波形诊断（figure 7, 8, 9）  
- figure 7：画目标原始信号 X_target(:,1) 的时域波形，导出 Target signal.pdf。  
- figure 8：画含干扰混合信号 X_noisy(:,1) 的时域波形，导出 Noisy signal.pdf。  
- figure 9：同时画两路输出 y_mvdr_target 和 y_mvdr_interf，导出 MVDR dual outputs.pdf。  
意义：先直观看“输入有多脏”和“双分支输出是否明显不同”。

2. 单图频谱图诊断（figure 10-13）  
- figure 10：目标输入的谱图。  
- figure 11：混合输入的谱图。  
- figure 12：目标指向输出的谱图。  
- figure 13：干扰指向输出的谱图。  
每张图都加 colorbar，单位为 dB/Hz，并导出 PDF。  
意义：看频率维度上的能量变化，例如目标频带是否保留、干扰频带是否被压制或保留。

3. 前后对照谱图（figure 14, 15）  
- figure 14：左图是处理前（mic1），右图是目标指向处理后。  
- figure 15：左图是处理前（mic1），右图是干扰指向处理后。  
关键点是用 CLim 统一两子图颜色范围（clim17、clim18），保证对比公平。  
意义：避免“颜色自动缩放”造成视觉误判。

4. PSD 全频段对比（figure 16）  
- 用 pwelch 计算三条功率谱：输入、目标指向输出、干扰指向输出。  
- figure 16 显示 0 到 fs/2 全频段，导出 PSD Fullband Dual MVDR.pdf。  
意义：看整体频谱能量重分配，不只看局部频点。

5. 2 kHz 局部 PSD（figure 17）  
- 只看 1800-2200 Hz，重点比较输入与目标指向输出。  
意义：针对目标相关频带验证增强效果，和前面的 2 kHz 波束诊断呼应。

6. 4 kHz 局部 PSD（figure 18 + 数值打印）  
- 图上比较输入、目标指向输出、干扰指向输出在 3500-4500 Hz 的变化。  
- 另外提取 4000 Hz 最近频点，打印三者的 dB/Hz 数值。  
意义：把“看图判断”变成“图+数值”的双重证据，更利于调参数和写报告。

7. 这段代码的整体价值  
- 时域图：看波形形态变化。  
- 谱图：看时频结构变化。  
- PSD：看统计平均频谱变化。  
三者互补，能从不同角度验证双分支 MVDR 是否达到预期。  