# IEEE 论文图表配色标准

## 核心原则
IEEE期刊对图表的配色要求：
1. **背景**: 纯白色 (`[1 1 1]` 或 `'white'`)
2. **坐标轴和框**: 黑色 (`'k'` 或 `[0 0 0]`)
3. **文字**: 黑色，使用等宽或Times New Roman字体
4. **线条和标记**: 使用可印刷和黑白友好的颜色
5. **图例**: 白色背景、黑色边框、黑色文字

## 推荐的颜色调色板

### 方案 1: 黑白印刷友好 (推荐用于 Full-band Eigenvalue Spectrum)
```matlab
% 适合5条线条以内的情况
ieee_colors = [
    0.0   0.0   0.0   % 1: 黑色 (Black)
    0.0   0.0   1.0   % 2: 蓝色 (Blue)
    1.0   0.0   0.0   % 3: 红色 (Red)
    0.0   0.5   0.0   % 4: 深绿色 (Dark Green)
    0.75  0.0   0.75  % 5: 紫色 (Purple)
];
```

### 方案 2: IEEE标准线型组合 (最通用)
结合**线型** + **颜色** + **标记**实现最佳区分度：

```matlab
% 结构: {color, line_style, marker}
ieee_styles = {
    {'k', '-',  'o'};    % 1: 黑色实线 + 圆形
    {'b', '-',  's'};    % 2: 蓝色实线 + 方形
    {'r', '--', '^'};    % 3: 红色虚线 + 上三角
    {'g', '-.',  'd'};   % 4: 绿色点划线 + 菱形
    {[0.75 0 0.75], ':', 'v'};  % 5: 紫色点线 + 下三角
};
```

### 方案 3: MATLAB标准(改进版)
```matlab
% 使用MATLAB内置颜色，但增强对比度
ieee_colors = [
    0.0  0.0  0.0      % 黑色
    0.0  0.4470 0.7410 % 深蓝
    0.8500 0.3250 0.098 % 深橙
    0.4660 0.6740 0.1880 % 绿色
    0.9290 0.6940 0.1250 % 黄色(偏金)
];

% 对应的线型和标记
line_styles = {'-', '-', '--', '-.', ':'};
markers = {'o', 's', '^', 'd', 'v'};
```

## 针对Full-band Eigenvalue Spectrum的推荐配置

### MATLAB代码片段
```matlab
% IEEE标准配色方案
figure(302); clf;
ieee_colors = [
    0.0   0.0   0.0      % β=0.0:   黑色
    0.0   0.447 0.741    % β=0.2:   蓝色
    0.85  0.325 0.098    % β=0.4:   红/橙色
    0.466 0.674 0.188    % β=0.6:   绿色
    0.75  0.0   0.75     % β=0.8:   紫色
];

line_styles = {'-', '-', '--', '-.', ':'};
markers = {'o', 's', '^', 'd', 'v'};
marker_sizes = [6, 6, 7, 6, 6];

for b_idx = 1:n_betas
    evals_full = evals_full_all{b_idx};
    Nmic = length(evals_full);
    semilogy(1:Nmic, evals_full + eps, ...
        'LineWidth', 2.2, 'MarkerSize', marker_sizes(b_idx), ...
        'Color', ieee_colors(b_idx,:), ...
        'LineStyle', line_styles{b_idx}, ...
        'Marker', markers{b_idx}, ...
        'DisplayName', sprintf('β = %.1f', beta_values(b_idx)), ...
        'MarkerFaceColor', 'none');
    hold on;
end

xlabel('Eigenvalue Index', 'FontSize', 16);
ylabel('Eigenvalue Magnitude', 'FontSize', 16);
title('Full-Band Covariance Eigenspectra (2 kHz + 4 kHz) vs. Reflection Coefficient', 'FontSize', 18);

% IEEE图例格式
h = legend('FontSize', 14, 'Location', 'best');
set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', ...
    'EdgeColor', 'k', 'LineWidth', 1.5);

grid on;
set(gca, 'FontSize', 13, 'LineWidth', 1.5, 'TickDir', 'out');

% 关键：确保背景为白色（用于印刷）
set(gcf, 'Color', 'white');
set(gca, 'Color', 'white', 'XColor', 'k', 'YColor', 'k');

% 导出时保持白色背景
exportgraphics(gcf, 'Eigenspectra_fullband_comparison.pdf', ...
    'ContentType', 'vector', 'BackgroundColor', 'white', 'Resolution', 600);
```

## IEEE配色原则检查清单

- ☑ 背景: 纯白色（不是浅灰，不是黑色）
- ☑ 轴线和标签: 黑色
- ☑ 线条颜色: 黑、蓝、红、绿、紫（黑白印刷可区分）
- ☑ 线宽: ≥1.5 pt（清晰可见）
- ☑ 标记: 空心标记（不是实心），避免混淆
- ☑ 图例: 白色背景、黑色边框、黑色文字
- ☑ 字体: Times New Roman或Arial
- ☑ 文字大小: 标题≥14pt，标签≥12pt，图例≥11pt
- ☑ 网格: 浅灰色（可选），不能太醒目
- ☑ 导出格式: PDF矢量格式，白色背景

## 为什么黑色背景不符合IEEE标准

1. **印刷问题**: 不支持高质量黑白印刷
2. **成本**: 彩色印刷成本高于黑白
3. **可访问性**: 不利于色盲读者辨识
4. **期刊政策**: IEEE明确要求白色背景
5. **转载**: 其他期刊转载时可能失效

## 推荐的导出参数

```matlab
% 标准IEEE PDF导出配置
exportgraphics(gcf, 'figure_name.pdf', ...
    'ContentType', 'vector', ...           % 矢量格式（不是光栅）
    'BackgroundColor', 'white', ...        % 强制白色背景
    'Resolution', 600);                   % 600 DPI（足够高质量）
```

## 快速参考：黑白兼容颜色代码

| 用途 | RGB值 | MATLAB | 用途说明 |
|------|-------|--------|---------|
| 线1 | [0, 0, 0] | 'k' | 黑色主线 |
| 线2 | [0, 0.447, 0.741] | 'b' | 蓝色 |
| 线3 | [0.85, 0.325, 0.098] | 'r'改进 | 深红/橙 |
| 线4 | [0.466, 0.674, 0.188] | 'g'改进 | 深绿 |
| 线5 | [0.75, 0, 0.75] | 紫色 | 紫色（非红+蓝） |
| 坐标轴 | [0, 0, 0] | 'k' | 黑色 |
| 背景 | [1, 1, 1] | 'white' | 纯白 |
| 网格 | [0.8, 0.8, 0.8] | 浅灰 | 可选辅助线 |

## 完整的IEEE风格图表模板

```matlab
function plot_ieee_style()
    % IEEE风格图表模板
    
    % 定义颜色和线型
    ieee_colors = [
        0.0  0.0  0.0
        0.0  0.447 0.741
        0.85 0.325 0.098
        0.466 0.674 0.188
        0.75 0.0 0.75
    ];
    
    line_styles = {'-', '-', '--', '-.', ':'};
    markers = {'o', 's', '^', 'd', 'v'};
    
    % 创建图形
    figure('Color', 'white', 'Position', [100, 100, 800, 600]);
    
    % 绘制数据
    x = 1:10;
    for i = 1:5
        y = x.^i;
        plot(x, y, 'Color', ieee_colors(i,:), ...
            'LineStyle', line_styles{i}, ...
            'Marker', markers{i}, ...
            'MarkerSize', 6, 'MarkerFaceColor', 'none', ...
            'LineWidth', 2, ...
            'DisplayName', sprintf('Curve %d', i));
        hold on;
    end
    
    % 标签和标题
    xlabel('X Axis', 'FontSize', 14, 'FontName', 'Times New Roman');
    ylabel('Y Axis', 'FontSize', 14, 'FontName', 'Times New Roman');
    title('IEEE Style Figure', 'FontSize', 16, 'FontName', 'Times New Roman');
    
    % 图例
    h = legend('Location', 'best', 'FontSize', 12);
    set(h, 'TextColor', 'k', 'Box', 'on', 'Color', 'white', ...
        'EdgeColor', 'k', 'LineWidth', 1.5);
    
    % 网格和坐标轴
    grid on; grid minor;
    set(gca, 'GridColor', [0.8 0.8 0.8], 'GridLineStyle', '-', ...
        'GridAlpha', 0.3, 'MinorGridLineStyle', ':', ...
        'LineWidth', 1.5, 'FontSize', 12, 'FontName', 'Times New Roman', ...
        'Color', 'white', 'XColor', 'k', 'YColor', 'k', 'TickDir', 'out');
    
    % 导出
    exportgraphics(gcf, 'ieee_figure.pdf', ...
        'ContentType', 'vector', 'BackgroundColor', 'white', 'Resolution', 600);
end
```

---

**版本**: v1.0  
**遵循标准**: IEEE Transaction and Journal Style  
**更新**: 2026-05-26
