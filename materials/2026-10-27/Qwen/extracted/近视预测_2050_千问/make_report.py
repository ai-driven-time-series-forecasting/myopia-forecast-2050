import json
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = "C:/Users/echo_/OneDrive/Desktop/近视预测_2050"
s = json.load(open(OUT + "/summary.json", encoding="utf-8"))
m = s['mean_metrics']

doc = Document()
doc.styles['Normal'].font.name = 'SimSun'
doc.styles['Normal'].font.size = Pt(11)

t = doc.add_heading('预测说明（模型：Qoder 大模型智能体）', level=0)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_heading('一、方法与建模思路', level=1)
doc.add_paragraph(
 '本预测由大模型智能体 Qoder 完成。数据为 1995–2022 年 31 省青少年近视患病率面板（1833 条有效观测），'
 '剔除零分母与缺失行后作 logit 变换，保证预测落于 (0,1)。特征兼顾时空：时间上用以 2008 年为中心的线性趋势'
 '并与学段、城乡交互；空间上采用省份固定效应、日照时数排名，以及基于省际距离矩阵的反距离加权邻省患病率滞后项；'
 '并控制学段、城乡与验光方法。模型为按样本量加权的岭回归，以十折前向链式时间序列交叉验证，每折仅用历史年份训练'
 '并预测其后一年，不随机打乱。借此模型可同时捕捉时间趋势、空间异质性与人群结构差异。')

doc.add_heading('二、对 2050 年预测结果的解读', level=1)
doc.add_paragraph(
 '过去二十余年，全国患病率由 2000 年前后约 0.39 升至 2022 年约 0.62，快速上升后增速趋缓，高学段与城市始终更高。'
 '预测显示 2023–2050 年继续上升但增速渐缓：全国均值 2030 年约 0.66、2040 年约 0.73、2050 年约 0.79，各省介于 '
 '0.69–0.86；高中段接近 0.95、小学低段约 0.43，城乡差距收窄，增量集中于低学段与农村。')

doc.add_heading('三、远期预测的不确定性讨论', level=1)
doc.add_paragraph(
 f'十折交叉验证平均指标为 MSE≈{m["MSE"]:.3f}、RMSE≈{m["RMSE"]:.3f}、MAE≈{m["MAE"]:.3f}、'
 f'MAPE≈{m["MAPE"]:.1f}%、R²≈{m["R2"]:.2f}：短期外推把握较好（部分折 R² 超 0.8），但 MAPE 偏高，主因低患病率组'
 '抽样噪声被百分比误差放大。28 年外推依赖 logit 线性趋势延续假设，无法预见防控政策等结构性冲击，故 2050 年数值'
 '应视为趋势参考；图中已用虚线标出 95% 预测区间。总体精度对趋势研判与省份排序令人满意，对单点绝对值宜审慎。'
 '若需更高精度，后续可引入分层模型或政策干预变量加以扩展。')

doc.save(OUT + "/预测说明_Qoder.docx")
print("docx saved")
