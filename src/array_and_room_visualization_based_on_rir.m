set(groot, ...
    'defaultAxesFontName','Times New Roman', ...
    'defaultTextFontName','Times New Roman', ...
    'defaultAxesFontSize',18, ...
    'defaultTextFontSize',20, ...
    'defaultLineLineWidth',1.2);

%% ========== 1. Read mic positions (X Y columns) ==========
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
fig_dir = fullfile(project_root, 'output', 'figures');
if exist(fig_dir, 'dir') ~= 7
    mkdir(fig_dir);
end

mic_xlsx = fullfile(project_root, 'data', 'mic_positions.xlsx');
mic_csv  = fullfile(project_root, 'data', 'mic_positions.csv');

if exist(mic_xlsx, 'file') == 2
    data = readmatrix(mic_xlsx);
elseif exist(mic_csv, 'file') == 2
    data = readmatrix(mic_csv);
else
    error('Cannot find mic position file. Checked: %s and %s', mic_xlsx, mic_csv);
end

X = data(:,1); Y = data(:,2);
Z = zeros(size(X));           % 如果你在Excel里有Z列，把这行改为 Z = data(:,3);
mic_pos = [X Y Z] / 1000;     % 假定Excel单位为 mm，直接换成 m；如果是 cm 用 /100

% center to given array center (optional)
array_center = [2.5 2 1.5];   % 房间位置（m）
mic_pos = mic_pos - mean(mic_pos,1) + array_center;     % 麦克风阵列位置

Nmic = size(mic_pos,1);
r_center = mean(mic_pos,1);   % 阵列中心（用于平面波相位参考）
fprintf('Nmic = %d, mic positions loaded (meters).\n', Nmic);

figure(1);
scatter(X, Y, 40, 'filled','black');
% scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 40, 'filled'); % 三维阵列图
axis equal; 
% grid on;
% xlabel('X (m)'); ylabel('Y (m)');
% title('Microphone Array Geometry');

set(gca, 'LineWidth', 1);
set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, fullfile(fig_dir, 'Microphone Array Geometry.pdf'), ...
    'ContentType','vector', 'BackgroundColor','white', 'Resolution',600);


% figure(1);
% % 绘制麦克风阵列散点图（黑色填充）
% scatter(X, Y, 40, 'filled','black');
% axis equal;  % 保持坐标轴等比例，避免图形变形
% % 核心：完全去除坐标轴相关元素
% set(gca, 'Visible', 'off');  % 隐藏整个坐标轴（轴线、刻度、标签、背景等）
% set(gcf, 'Color', 'white');  % 设置图形背景为白色（避免导出后背景透明/黑色）
% % 保留线条宽度（若需要）
% set(gca, 'LineWidth', 1);
% % 导出图片（高分辨率）
% exportgraphics(gcf, 'Microphone Array Geometry.pdf', 'Resolution',600,...
%     'ContentType','image');

%% ========== 2. Room, RIR and signals parameters ==========
c = 340;
fs = 16000;
t = (0:1/fs:1-1/fs)';
f0 = 1000;         % target freq (Hz)
f1 = 2000;         % interference freq (Hz)
SNR = 30;          % dB
INR = 10;          % dB (interference relative to target amplitude)
nsample = 4096;    % RIR length

% source 3D positions
s_target = [2 1 4];   % used for RIR generation (m)
s_interf  = [2 4 4];

%% ====================== 可视化房间、声源和麦克风阵列位置 ======================
figure(2); clf;
hold on; grid on; axis equal;

% 1. 绘制房间边界（矩形框）
L = [5 4 6];  % 房间长宽高
xv = [0 L(1) L(1) 0 0];
yv = [0 0 L(2) L(2) 0];
z0 = zeros(size(xv));

% 地面和顶面
plot3(xv, yv, z0, 'k--', 'LineWidth', 1.2);
plot3(xv, yv, L(3)*ones(size(z0)), 'k--', 'LineWidth', 1.2);

% 垂直边
for i = 1:4
    plot3([xv(i) xv(i)], [yv(i) yv(i)], [0 L(3)], 'k--');
end

% 2. 绘制麦克风阵列位置
scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 60, 'filled', 'b');
text(mean(mic_pos(:,1)), mean(mic_pos(:,2)), mean(mic_pos(:,3))+0.1, 'Mic Array', ...
     'Color', 'b', 'FontWeight', 'bold');

% 3. 绘制目标声源与干扰源
scatter3(s_target(1), s_target(2), s_target(3), 100, 'r', 'filled');
text(s_target(1), s_target(2), s_target(3)+0.1, 'Target', 'Color', 'r', 'FontWeight', 'bold');

scatter3(s_interf(1), s_interf(2), s_interf(3), 100, 'm', 'filled');
text(s_interf(1), s_interf(2), s_interf(3)+0.1, 'Interference', 'Color', 'm', 'FontWeight', 'bold');

% 4. 可选：连接阵列中心与声源方向
r_center = mean(mic_pos,1);
plot3([r_center(1) s_target(1)], [r_center(2) s_target(2)], [r_center(3) s_target(3)], 'r--', 'LineWidth', 1.2);
plot3([r_center(1) s_interf(1)], [r_center(2) s_interf(2)], [r_center(3) s_interf(3)], 'm--', 'LineWidth', 1.2);

% 5. 设置视角与标签
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
% title('Room, Sources, and Microphone Array Layout');
view(45, 25);
% legend({'Room boundary','Microphones','Target Source','Interference'}, 'Location','best');
% legend({'Room boundary'}, 'Location','northeast');

set(gca, 'LineWidth', 1);
set(gcf,'Color','white'); set(gca,'Color','white');
exportgraphics(gcf, fullfile(fig_dir, 'Microphone Array Layout.pdf'), ...
    'ContentType','vector', 'BackgroundColor','white', 'Resolution',600);
