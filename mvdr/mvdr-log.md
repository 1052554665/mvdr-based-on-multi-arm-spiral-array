
# 对角加载位置
主要位置在 src/mvdr_processing_module.m 和 src/mvdr_processing_module.m 附近，协方差矩阵会做这一步：
```matlab
Rxx = Rxx + config.epsilon * trace(Rxx)/Nmic * eye(Nmic, 'like', Rxx);
```
如果开启了 `shrink_alpha`，前面还会先做一次收缩稳健化：
```matlab
Rxx = (1 - config.shrink_alpha) * Rxx + config.shrink_alpha * mu * eye(...)
```

但要注意一件事：**对角加载通常提升鲁棒性，但往往会让主瓣更宽一点**。所以如果主瓣已经偏宽，通常不是“加更多对角加载”，而是：

1. **减小 `epsilon`**
- 现在默认是 src/configs/mvdr_default_config.m 里的 `1e-4`
- 可以试 `1e-5`、`1e-6`

1. **减小 `shrink_alpha`**
- 现在默认是 src/configs/mvdr_default_config.m 的 `0.01`
- 如果太保守，可以降到 `0` 或更小

1. **增加协方差估计稳定性**
- 增大 `Mavg`
- 或者适当减少 `M_signal` 做信号子空间抑制

1. **确认不是波束图诊断里重复“稳健化”导致的视觉变宽**
- 你在 src/diagnostics_and_visualization_module.m 也有一处额外的对角加载，主要是为了画波束图时稳定求逆。

结论很直接：
- **可以引入对角加载**
- **但主瓣已经宽时，不建议再加大**
- 更常见的做法是：**保留小量对角加载，往下调 `epsilon`**，同时检查 `Mavg`、`M_signal` 和导向矢量是否准确


# 参数设置及结果
`mvdr_default_config.m` 里相关参数设置如下：

| Mavg | epsilon | shrink_alpha | $BW_{-3dB} \degree$  | Main Lobe | Side Lobes |Side Lobes ratio|azimuth $\degree$|
|---|---|---|---|---|---|---|---|
| 31| 1e-3 | 0 | 59| 8.59139 | 0.567013 |8.024377 | -91 |
| 31 | 1e-4 | 0 | 52.0 | 11.6094 | -3.10 | 10.7094| -7 |
| 31| 1e-5 | 0 | 46.0 | 15.8039 | 6.40 |   9.4039|-91|
| 31| 1e-6 | 0 | 59.0 | 18.4165 | 18.3789| 0.0376  | -7|
---

这组数据已经能够说明：

* `epsilon` 对 MVDR 协方差矩阵稳定性影响非常大
* 波束形态已经发生了明显变化
* 当前实验中已经出现“数值病态”和“主瓣漂移”现象

因此，这组实验是有分析价值的，但目前还不够完整，建议继续补实验，否则论文中的结论会不够扎实。

---

# 一、先分析这组数据

你当前固定：

| 参数           | 值  |
| ------------ | -- |
| Mavg         | 31 |
| shrink_alpha | 0  |

变化的是：

[
\epsilon = 10^{-3} \sim 10^{-6}
]

这是典型：

[
R_\epsilon = R + \epsilon I
]

即：

MVDR 对角加载（Diagonal Loading）实验。

---

# 二、数据趋势分析

你的表：

| epsilon | BW | Main Lobe | Side Lobes | SLR   | azimuth |
| ------- | -- | --------- | ---------- | ----- | ------- |
| 1e-3    | 59 | 8.59      | 0.57       | 8.02  | -91     |
| 1e-4    | 52 | 11.61     | -3.10      | 10.71 | -7      |
| 1e-5    | 46 | 15.80     | 6.40       | 9.40  | -91     |
| 1e-6    | 59 | 18.42     | 18.38      | 0.037 | -7      |

---

# 三、核心现象

---

# 1. epsilon减小 → 主瓣增益增大

趋势：

[
8.59 \rightarrow 18.42
]

这是典型现象。

因为：

[
w_{MVDR}
========

\frac{R^{-1}a(\theta)}
{a^H(\theta)R^{-1}a(\theta)}
]

当：

[
R + \epsilon I
]

里的：

[
\epsilon \downarrow
]

则：

* 正则化减弱
* 权值约束减弱
* 波束形成器更加“激进”
* 主瓣峰值会升高

这很正常。

---

# 2. BW先减小后突然增大

趋势：

[
59 \rightarrow 52 \rightarrow 46 \rightarrow 59
]

说明：

* 前期分辨率提高
* 后期开始数值失稳

尤其：

[
\epsilon = 10^{-6}
]

时：

* 主瓣与旁瓣几乎等高

因为：

| Main  | Side  |
| ----- | ----- |
| 18.42 | 18.38 |

已经接近：

[
SLR \approx 0
]

这意味着：

> MVDR 已经失去有效波束结构。

这是典型：

# 协方差矩阵病态（ill-conditioned）

现象。

---

# 3. 最关键：azimuth 在 -91 与 -7 之间跳变

这是你最应该关注的问题。

因为：

正常情况下：

> 主瓣方向不应该大幅漂移。

尤其：

[
-91^\circ \leftrightarrow -7^\circ
]

这种跨度极大。

这说明：

当前 MVDR 输出已经不稳定。

---

# 四、为什么方位角会变化？

这是核心。

---

# 原因1：协方差矩阵接近奇异（最主要）

当：

[
\epsilon \to 0
]

时：

[
R+\epsilon I
]

越来越接近原始协方差矩阵。

如果：

* 快拍数不足
* 阵元相关性强
* 噪声小
* 存在相干源

则：

[
R
]

可能病态：

[
\kappa(R) \gg 1
]

即条件数极大。

于是：

[
R^{-1}
]

对微小误差极度敏感。

结果：

* 波束峰值位置跳变
* 出现伪峰
* 主瓣漂移

这是非常经典的 MVDR 数值不稳定问题。

---

# 原因2：伪峰超过真实峰

你当前：

| epsilon | Main  | Side  |
| ------- | ----- | ----- |
| 1e-6    | 18.42 | 18.38 |

说明：

旁瓣已经几乎和主瓣一样高。

那么：

程序在寻找：

```python
argmax(P)
```

时：

可能：

* 某个伪峰略高
* 就被识别为主瓣

于是：

azimuth 会突然跳到：

[
-7^\circ
]

而不是：

[
-91^\circ
]

---

# 原因3：阵列几何导致对称模糊

如果你使用：

* 圆阵
* 螺旋阵
* 对称阵列

则：

可能存在：

[
a(\theta_1)\approx a(\theta_2)
]

即：

阵列流形模糊。

当数值不稳定时：

峰值可能在两个方向间跳变。

---

# 原因4：扫描分辨率问题

例如：

```python
theta = np.arange(-90,90,1)
```

若：

* 峰值较宽
* 多峰接近

则：

离散扫描会导致：

最大值索引跳变。

---

# 五、这组实验是否足够？

目前：

# 还不够。

因为：

你现在只能说明：

> epsilon 会影响 MVDR稳定性。

但还不能形成：

* 完整规律
* 可发表结论
* 可信统计结果

---

# 六、建议必须补充的实验

---

# 1. 条件数实验（强烈推荐）

这是最重要补充。

增加：

| epsilon | cond(R) |
| ------- | ------- |
| 1e-3    | xxx     |
| 1e-4    | xxx     |
| 1e-5    | xxx     |
| 1e-6    | xxx     |

代码：

```python
cond_num = np.linalg.cond(R_loaded)
```

你会很可能发现：

[
\epsilon \downarrow
\Rightarrow
cond(R)\uparrow
]

然后：

azimuth 开始漂移。

这会形成非常强的论文逻辑链：

[
\epsilon
\rightarrow
Condition\ Number
\rightarrow
Numerical\ Instability
\rightarrow
Beam\ Drift
]

这是高质量分析。

---

# 2. 波束图可视化（必须）

现在只有表格。

不够。

建议画：

## 不同 epsilon 下：

* 波束方向图
* 主瓣位置
* 旁瓣变化

例如：

[
\epsilon = 10^{-3},10^{-4},10^{-5},10^{-6}
]

四条曲线。

这是论文中最关键图之一。

---

# 3. Monte Carlo统计（推荐）

因为 MVDR 本身有随机性。

建议：

每个 epsilon：

重复：

* 20次
* 50次

统计：

| epsilon | Mean Azimuth | Std |
|---|---|
| 1e-3 | -91.1 | 0.2 |
| 1e-6 | -42.5 | 38.1 |

如果：

std突然暴涨：

就能直接证明：

> 数值不稳定。

这是非常有说服力的。

---

# 4. shrinkage实验（非常建议）

你现在：

```text
shrink_alpha = 0
```

即：

没有收缩协方差。

这是不完整的。

因为：

Shrinkage 本来就是：

解决：

* 病态协方差
* 快拍不足
* 数值不稳定

问题的。

建议：

---

## 固定：

```text
epsilon = 1e-6
```

扫描：

```text
alpha = 0 → 1
```

观察：

* 方位角是否稳定
* SLR是否恢复
* 条件数是否下降

这会形成完整论文逻辑：

---

# 普通MVDR：

[
R+\epsilon I
]

数值不稳定。

---

# Shrinkage MVDR：

[
R_s=(1-\alpha)R+\alpha I
]

稳定性提升。

这是很完整的研究闭环。

---

# 七、你现在这组数据已经透露的重要结论

已经可以初步得出：

---

## 结论1

过小的：

[
\epsilon
]

会导致：

* 协方差矩阵病态
* MVDR数值失稳

---

## 结论2

MVDR 对：

[
\epsilon
]

高度敏感。

---

## 结论3

当：

[
\epsilon \le 10^{-6}
]

时：

主瓣与旁瓣不可分离：

[
SLR \approx 0
]

系统失去有效DOA估计能力。

---

## 结论4

方位角跳变说明：

> 伪峰开始主导空间谱。

这是典型失稳现象。

---

# 八、论文角度建议

你这组实验其实已经很接近：

# “鲁棒MVDR”方向。

建议你后续增加：

* Diagonal Loading
* Shrinkage Covariance
* Eigenvalue Spectrum
* Condition Number
* Snapshot deficiency

这些内容。

因为：

你现在的数据已经明显体现：

# “协方差矩阵稳定性”

是核心问题。




