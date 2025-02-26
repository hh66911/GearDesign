import math
import streamlit as st
from streamlit_cookies_controller import CookieController
import scipy.optimize as opt
import numpy as np
import pandas as pd


class Angle:
    def __init__(self, degrees):
        # 初始化角度值，以度为单位
        self._degrees = degrees

    def __repr__(self):
        # 输出时只输出角度值
        return f'A({self._degrees})'

    def __str__(self):
        # 字符串表示，输出度分秒
        degrees = self._degrees
        deg = int(degrees)
        minutes = (degrees - deg) * 60
        min = int(minutes)
        seconds = (minutes - min) * 60
        return f'{deg}°{min}′{seconds : .2f}″'
    
    def __float__(self):
        # 数字表示，输出角度值
        return float(self._degrees)

    def to_radians(self):
        # 将角度转换为弧度
        return math.radians(self._degrees)

    def sin(self):
        # 计算正弦值
        return math.sin(self.to_radians())

    def cos(self):
        # 计算余弦值
        return math.cos(self.to_radians())
    
    def tan(self):
        return math.tan(self.to_radians())
    



def input_params_ui():
    st.title('软齿轮系计算')
    
    controller = CookieController(key='cookies')
    input_params = None
    if st.button('读取上次参数'):
        input_params = controller.get('input_params')
        
    if input_params is None:
        input_params = {
            'k': 1., 'time_mode': 0, 'years': 0.,
            'p_in': 5., 'n_in': 100., 'n_out': 1., 'phid': 1., 'coeff': 1.,
            'eta_I': 1., 'eta_II': 1., 'eta_III': 1.,
            'z1': 24, 'z3': 24, 'beta_I': 0, 'beta_II': 0,
            'sigh_lim_13': 100., 'sigh_lim_24': 100.,
            'sigf_e_13': 100., 'sigf_e_24': 100., 'ze': 100.
        }
        
    st.header(body='工况参数')
    input_params['k'] = st.number_input('初选工况系数', value=input_params['k'])
    TIME_MODES = ['单班制', '双班制']
    input_params['time_mode'] = TIME_MODES.index(st.selectbox('选择班制：', TIME_MODES, input_params['time_mode']))
    input_params['years'] = st.number_input('工作年数：', value=input_params['years'])
    
    
    st.header(body='材料参数')
    input_params['beta_I'] = float(input_params['beta_I'])
    input_params['beta_II'] = float(input_params['beta_II'])
    input_params['sigh_lim_13'] = st.number_input(r'小齿轮接触极限 $\text{(MPa)}$', value=input_params['sigh_lim_13'])
    input_params['sigf_e_13'] = st.number_input(r'小齿轮抗弯极限 $\text{(MPa)}$', value=input_params['sigf_e_13'])
    input_params['sigh_lim_24'] = st.number_input(r'大齿轮接触极限 $\text{(MPa)}$', value=input_params['sigh_lim_24'])
    input_params['sigf_e_24'] = st.number_input(r'大齿轮抗弯极限 $\text{(MPa)}$', value=input_params['sigf_e_24'])
    input_params['ze'] = st.number_input(r'$Z_e$ ($\sqrt{MPa}$))', value=input_params['ze'])
    
    st.header('尺寸参数')
    input_params['z1'] = st.number_input('低速级小齿轮齿数', value=input_params['z1'])
    input_params['z3'] = st.number_input('高速级小齿轮齿数', value=input_params['z3'])
    input_params['beta_I'] = st.number_input('高速级螺旋角 β (°)', value=input_params['beta_I'])
    input_params['beta_I'] = st.number_input('低速级螺旋角 β (°)', value=input_params['beta_II'])
    
    st.header('传动参数')
    input_params['p_in'] = st.number_input(r'输入功率 $P$ $\text{(kW)}$', value=input_params['p_in'])
    input_params['n_in'] = st.number_input(r'输入转速 $n$ $\text{(rpm)}$', value=input_params['n_in'])
    input_params['n_out'] = st.number_input(r'目标转速 $n_{out}$ $\text{(rpm)}$', value=input_params['n_out'])
    input_params['eta_I'] = st.number_input(r'$\text{I}$ 轴效率', value=input_params['eta_I'])
    input_params['eta_II'] = st.number_input(r'$\text{II}$ 轴效率', value=input_params['eta_II'])
    input_params['eta_III'] = st.number_input(r'$\text{III}$ 轴效率', value=input_params['eta_III'])
    input_params['phid'] = st.number_input('宽度系数 $Φ_d$', value=input_params['phid'])
    input_params['coeff'] = st.number_input('传动比分配系数', value=input_params['coeff'])
    
    if st.button('保存参数'):
        st.write('已保存')
        controller.set('input_params', input_params)
    
    input_params['beta_I'] = Angle(input_params['beta_I'])
    input_params['beta_II'] = Angle(input_params['beta_II'])
    
    return input_params


def calc_iI_iII(i_total: float, coeff: float) -> tuple[float, float]:
    # 解方程：C * (iI + 1) * iI^4 / ((iI + i_total) * i_total^2) - 1 = 0
    def func(iI: float):
        return coeff * (iI + 1) * iI ** 4 / ((iI + i_total) * i_total ** 2)
    iI: float = opt.fsolve(lambda x: func(x) - 1, i_total**0.5)[0]
    iII: float = i_total / iI
    return iI, iII


input_params = input_params_ui()


# region 计算传动参数
INPUT_POWER: float = input_params['p_in']
INPUT_SPEED: float = input_params['n_in']
PHI_D: float = input_params['phid']

ETA_I: float = input_params['eta_I']
ETA_II: float = input_params['eta_II']
ETA_III: float = input_params['eta_III']
P_I = INPUT_POWER * ETA_I
P_II = P_I * ETA_II
P_III = P_II * ETA_III

I_TOTAL = INPUT_SPEED / input_params['n_out']
i1, i2 = calc_iI_iII(I_TOTAL, input_params['coeff'])
st.write(f'理想高速级转动比 {i1: .2f}，低速级传动比 {i2: .2f}')

z1: int = input_params['z1']
z3: int = input_params['z3']
z2 = round(z1 * i1)
z4 = round(z3 * i2)
i1 = z2 / z1
i2 = z4 / z3

st.write(f'粗算高速级转动比 {i1: .2f}，低速级传动比 {i2: .2f}')

# 计算各级转速
N_I = INPUT_SPEED
N_II = N_I / i1
N_III = N_II / i2
# 计算各级扭矩
T_I = P_I * 30 / (math.pi * N_I) * 1e3
T_II = P_II * 30 / (math.pi * N_II) * 1e3
T_III = P_III * 30 / (math.pi * N_III) * 1e3

st.write(f'功率链（kW）： {P_I: .2f} -> {P_II: .2f} -> {P_III: .2f}')
st.write(f'转速链（rpm）： {N_I: .2f} -> {N_II: .2f} -> {N_III: .2f}')
st.write(f'扭矩链（N·m）： {T_I: .2f} -> {T_II: .2f} -> {T_III: .2f}')
speed_error = abs(INPUT_SPEED / I_TOTAL - (INPUT_SPEED / (i1 * i2)))
speed_re = speed_error / (INPUT_SPEED / I_TOTAL) * 100
if speed_re < 0.005:
    st.write(f'转速误差：{speed_re: .4f}%')
else:
    st.write(f'转速误差：{speed_re: .2f}%')
# endregion 计算传动参数


# region 计算斜齿轮区域系数
ALPHA_N = Angle(20)
BETA_1: Angle = input_params['beta_I']
BETA_2: Angle = input_params['beta_II']

def calc_zh(alpha: Angle, beta: Angle):
    if float(beta) < 8.:
        return 2.5
    alpha_t = math.atan(alpha.tan() / beta.tan())
    beta_b = beta.tan() * alpha_t.cos()
    zh_rad = math.sqrt(2 * beta_b.cos() / (alpha_t.sin() * alpha_t.cos()))
    return Angle(zh_rad / math.pi * 180)

ZH_1 = calc_zh(ALPHA_N, BETA_1)
ZH_2 = calc_zh(ALPHA_N, BETA_2)
# endregion 计算斜齿轮区域系数


st.markdown('---')
st.header('粗算')


verify_passed = True
def fail_verify(reason):
    global verify_passed
    verify_passed = False
    st.error(reason, icon='🚨')


# region 计算齿形系数
def calc_yfa(z, beta) -> float:
    denominator = math.cos(beta) ** 3
    if denominator == 0:
        return 0.0  # 防止分母为零的情况，具体处理需根据实际需求调整
    zv = z / denominator
    if zv < 35:
        term = (zv - 16) / (math.cos(beta) ** 3)
        return (-0.0000570752776635208 * (term ** 3) +
                0.00307677616500968 * (term ** 2) -
                0.0688841305752419 * term +
                3.03422577422526)
    if zv < 110:
        term = (zv / 10 - 2) / (math.cos(beta) ** 3)
        return (-0.00141414141414042 * (term ** 3) +
                0.0267099567099223 * (term ** 2) -
                0.18568542568536 * term +
                2.6785714285711)
    if zv < 160:
        return 2.14
    if zv < 210:
        return 2.12
    return 2.1


def calc_ysa(z, beta) -> float:
    if z < 35:
        term = (z - 16) / (math.cos(beta) ** 3)
        return (0.0000291375291376905 * (term ** 3) -
                0.00079295704295923 * (term ** 2) +
                0.0139880952381617 * term +
                1.50570429570396)
    if z < 130:
        term = (z / 10 - 2) / (math.cos(beta) ** 3)
        return (-0.0027083333 * (term ** 2) +
                0.0474107143 * term +
                1.5825892857)
    if z < 160:
        return 1.83
    if z < 210:
        return 1.865
    return 1.9
    
def calc_yf(z, beta):
    return calc_yfa(z, beta) * calc_ysa(z, beta)

yf = [
    calc_yf(z1, BETA_1),
    calc_yf(z2, BETA_1),
    calc_yf(z3, BETA_2),
    calc_yf(z4, BETA_2)
]
# endregion 计算齿形系数


# region 计算许用值
ZE: float = input_params['ze']
SIG_H_LIM_13 = input_params['sigh_lim_13']
SIG_H_LIM_24 = input_params['sigh_lim_24']
SIG_F_E_13 = input_params['sigf_e_13']
SIG_F_E_24 = input_params['sigf_e_24']

S_MIN_SELECTIONS = [
    '高可考虑（1 / 10,000)',
    '中可考虑（1 / 1,000)',
    '一般可考虑（1 / 100)',
    '低可考虑（1 / 10) 可能在塑性形变前点蚀',
]
S_SEL = S_MIN_SELECTIONS.index(st.selectbox('最小安全系数', S_MIN_SELECTIONS))
S_MIN_H_SELECTIONS = [1.5, 1.25, 1., .85]
S_MIN_F_SELECTIONS = [2., 1.6, 1.25, 1.]
SH_MIN = S_MIN_H_SELECTIONS[S_SEL]
SF_MIN = S_MIN_F_SELECTIONS[S_SEL]
st.write('| $S_{H_{min}}$ | $S_{F_{min}}$ |\n| :-: | :-: |\n' + f'| {SF_MIN} | {SH_MIN} |')

def calc_sigma_h(
    sigh_lim: float, sh_min: float,
    N: float, type: int, exp_adjust = 0.
):
    """
    计算材料的接触疲劳许用值 sigma_h。
    计算参考 GB/T 6366-2-2019 第 11 章节。
    对于N > 10^10：ZNT的较小值可用于点蚀出现最少的严格工况中；
    0.85和1.0之间的值可用于常规传动装置；
    处于最佳的润滑状态、材料与加工制造下可选用1.0作为经验值。

    Parameters
    ----------
    sigh_lim (float): 材料接触疲劳极限。

    sh_min (float): 接触疲劳安全系数。

    N (float): 应力循环次数。

    type (int): 材料类型（曲线类型）。

    exp_adjust (float, optional): 经验系数，决定 `N = 10^10` 时的值。

        范围为 0 到 1，对应 0.85 到 1。默认为 0。

    Returns
    ----------
    float: 计算得到的接触疲劳极限 sigma_h。
    """
    adjust_val = np.interp(exp_adjust, [0., 1.], [0.85, 1.])
    # 定义每种材料的应力循环次数和寿命系数
    materials = [
        {
            'N': [0, 6e5, 1e7, 1e9, 1e10],
            'ZNT': [1.6, 1.6, 1.3, 1.0, adjust_val]
        },
        {
            'N': [0, 1e5, 5e7, 1e10],
            'ZNT': [1.6, 1.6, 1.0, adjust_val]
        },
        {
            'N': [0, 1e5, 2e6, 1e10],
            'ZNT': [1.3, 1.3, 1.0, adjust_val]
        },
        {
            'N': [0, 1e5, 2e6, 1e10],
            'ZNT': [1.1, 1.1, 1.0, adjust_val]
        }
    ]
    log_N = np.log10(N)
    # 获取该材料的载荷循环次数和寿命系数
    N_values = materials[type]['N']
    ZNT_values = materials[type]['ZNT']
    log_N_values = np.log10(N_values)
    ZNT = np.interp(log_N, log_N_values, ZNT_values).item()
    return f'{N: .2e}', ZNT, sigh_lim / sh_min * ZNT

def calc_sigma_f(
    sigf_lim: float, sf_min: float,
    N: float, type: int, exp_adjust = 0.
):
    """
    计算材料的接触疲劳许用值 `sigma_h`。
    计算参考 GB/T 6366-3-2019 第 12 章节。
    在应力循环次数 `NL >= 10^10` 时，寿命系数 `YNT` 的取值范围为 0.85 到 1.0 。
    其中，较低的 `YNT` 值适用于仅有微小齿根裂纹的苛刻工况。
    在一般情况下，对于齿轮传动， `YNT` 的取值可以在 0.85 到 1.0 之间选择。
    当满足最佳的润滑、材料、制造和经验条件时， `YNT` 可以取1.0。

    Parameters
    ----------
    sigh_lim (float): 材料接触疲劳极限。
    
    sh_min (float): 接触疲劳安全系数。
    
    N (float): 应力循环次数。
    
    type (int): 材料类型（曲线类型）。
    
    exp_adjust (float, optional): 经验系数，决定 N = 10^10 时的值。
        范围为 0 到 1，对应 0.85 到 1。默认为 0。

    Returns
    -------
    float: 计算得到的接触疲劳极限 sigma_h。
    """
    # 定义每种材料的应力循环次数和寿命系数
    adjust_val = np.interp(exp_adjust, [0., 1.], [0.85, 1.])
    materials = [
        {
            'N': [0, 1e4, 3e6, 1e10],
            'YNT': [2.5, 2.5, 1.0, adjust_val]
        },
        {
            'N': [0, 1e3, 3e6, 1e10],
            'YNT': [2.5, 2.5, 1.0, adjust_val]
        },
        {
            'N': [0, 1e3, 3e6, 1e10],
            'YNT': [1.6, 1.6, 1.0, adjust_val]
        },
        {
            'N': [0, 1e3, 3e6, 1e10],
            'YNT': [1.1, 1.1, 1.0, adjust_val]
        }
    ]
    log_N = np.log10(N)
    # 获取该材料的载荷循环次数和寿命系数
    N_values = materials[type]['N']
    YNT_values = materials[type]['YNT']
    log_N_values = np.log10(N_values)
    YNT = np.interp(log_N, log_N_values, YNT_values).item()
    return f'{N: .2e}', YNT, sigf_lim / sf_min * YNT

MATERIAL_TYPES = [
    '允许一定点蚀的结构钢；调质钢；球墨铸铁（珠光体、贝氏体）；珠光体和可锻铸铁；渗碳淬火的渗碳钢',
    '结构钢；调质钢；渗碳淬火钢；火焰、感应淬火；球墨铸铁；珠光体、可锻铁',
    '灰铸铁；球墨铸铁（铁素体）；渗氮钢、调质钢、渗碳钢',
    '碳氮共渗钢、渗碳钢'
]
ma_type_13 = MATERIAL_TYPES.index(st.selectbox('选择你的小齿轮材料类型：', MATERIAL_TYPES))
ma_type_24 = MATERIAL_TYPES.index(st.selectbox('选择你的大齿轮材料类型：', MATERIAL_TYPES))
time_per_day = [8, 16][input_params['time_mode']]
n_years = input_params['years']
time_hours = n_years * 365 * time_per_day
nloop_1 = 1 * 60 * time_hours * N_I
nloop_2 = 1 * 60 * time_hours * N_II
nloop_3 = 1 * 60 * time_hours * N_II
nloop_4 = 1 * 60 * time_hours * N_III
SIGH = [
    calc_sigma_h(SIG_H_LIM_13, SH_MIN, nloop_1, ma_type_13),
    calc_sigma_h(SIG_H_LIM_24, SH_MIN, nloop_2, ma_type_24),
    calc_sigma_h(SIG_H_LIM_13, SH_MIN, nloop_3, ma_type_13),
    calc_sigma_h(SIG_H_LIM_24, SH_MIN, nloop_4, ma_type_24)
]
SIGF = [
    calc_sigma_f(SIG_F_E_13, SF_MIN, nloop_1, ma_type_13),
    calc_sigma_f(SIG_F_E_24, SF_MIN, nloop_2, ma_type_24),
    calc_sigma_f(SIG_F_E_13, SF_MIN, nloop_3, ma_type_13),
    calc_sigma_f(SIG_F_E_24, SF_MIN, nloop_4, ma_type_24)
]

st.subheader('接触疲劳强度计算')
st.write('寿命系数计算方法参考 GB/T 6366-2019 中的表格。')
st.table(pd.DataFrame(SIGH, range(1, 5), [r'应力循环', r'$Z_N$', r'$\sigma_H$']))
st.table(pd.DataFrame(SIGF, range(1, 5), [r'应力循环', r'$Y_N$', r'$\sigma_F$']))
# endregion 计算许用值


k: float = input_params['k']
    
def calc_dmin(
    zh  : float, # 区域系数，普通圆柱齿轮 = 2.5
    t   : float, # 小齿轮扭矩
    k_  : float, # 载荷系数
    i   : float, # 传动比
    sigh: float, # 接触疲劳强度
    beta: Angle  # 斜齿轮螺旋角
    ):
    return (2 * k_ * t * (i + 1) / i * beta.cos() * (zh * ze / sigh) ** 2)**(1 / 3)
    

st.subheader('计算直径')