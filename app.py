import math
import io
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
        return f"{deg}°{min}'{seconds : <.2f}″"

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
    if 'run_count' not in st.session_state:
        st.session_state.run_count = 0
    st.session_state.run_count += 1
    st.title('软齿轮系计算')

    controller = CookieController(key='cookies')
    input_params = controller.get('input_params')
    INIT_PARAMS = {
        'k': 1., 'time_mode': 0, 'years': 0.,
        'p_in': 5., 'n_in': 100., 'n_out': 1., 'phid': 1., 'coeff': 1.,
        'eta_I': 1., 'eta_II': 1., 'eta_III': 1.,
        'z1': 24, 'z3': 24, 'beta_I': 0, 'beta_II': 0,
        'sigh_lim_13': 100., 'sigh_lim_24': 100.,
        'sigf_e_13': 100., 'sigf_e_24': 100., 'ze': 100.
    }
    if input_params is None:
        input_params = INIT_PARAMS
    else:
        para_keys = INIT_PARAMS.keys()
        for k in para_keys:
            if k not in input_params:
                input_params[k] = INIT_PARAMS[k]

    for k in input_params:
        if k not in ('z1', 'z3', 'time_mode'):
            input_params[k] = float(input_params[k])

    with st.container(border=True):
        st.subheader(body='工况参数')
        input_params['k'] = st.number_input('初选工况系数', value=input_params['k'])
        TIME_MODES = ['单班制', '双班制']
        input_params['time_mode'] = TIME_MODES.index(
            st.selectbox('选择班制：', TIME_MODES, input_params['time_mode']))
        input_params['years'] = st.number_input(
            '工作年数：', value=input_params['years'])
        input_params['beta_I'] = input_params['beta_I']
        input_params['beta_II'] = input_params['beta_II']

        st.subheader(body='材料参数')
        input_params['sigh_lim_13'] = st.number_input(
            r'小齿轮接触极限 $\text{(MPa)}$', value=input_params['sigh_lim_13'])
        input_params['sigf_e_13'] = st.number_input(
            r'小齿轮抗弯极限 $\text{(MPa)}$', value=input_params['sigf_e_13'])
        input_params['sigh_lim_24'] = st.number_input(
            r'大齿轮接触极限 $\text{(MPa)}$', value=input_params['sigh_lim_24'])
        input_params['sigf_e_24'] = st.number_input(
            r'大齿轮抗弯极限 $\text{(MPa)}$', value=input_params['sigf_e_24'])
        input_params['ze'] = st.number_input(
            r'$Z_e$ ($\sqrt{MPa}$))', value=input_params['ze'])

        st.subheader('尺寸参数')
        input_params['z1'] = st.number_input(
            '低速级小齿轮齿数', value=input_params['z1'])
        input_params['z3'] = st.number_input(
            '高速级小齿轮齿数', value=input_params['z3'])
        input_params['beta_I'] = st.number_input(
            '高速级螺旋角 β (°)', value=input_params['beta_I'])
        input_params['beta_II'] = st.number_input(
            '低速级螺旋角 β (°)', value=input_params['beta_II'])

        st.subheader('传动参数')
        input_params['p_in'] = st.number_input(
            r'输入功率 $P$ $\text{(kW)}$', value=input_params['p_in'])
        input_params['n_in'] = st.number_input(
            r'输入转速 $n$ $\text{(rpm)}$', value=input_params['n_in'])
        input_params['n_out'] = st.number_input(
            r'目标转速 $n_{out}$ $\text{(rpm)}$', value=input_params['n_out'])
        input_params['eta_I'] = st.number_input(
            r'$\text{I}$ 轴效率', value=input_params['eta_I'])
        input_params['eta_II'] = st.number_input(
            r'$\text{II}$ 轴效率', value=input_params['eta_II'])
        input_params['eta_III'] = st.number_input(
            r'$\text{III}$ 轴效率', value=input_params['eta_III'])
        input_params['phid'] = st.number_input(
            '宽度系数 $Φ_d$', value=input_params['phid'])
        input_params['coeff'] = st.number_input(
            '传动比分配系数', value=input_params['coeff'])

    # print(input_params)
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


st.write(r'建议放大 25% 查看该网页')
input_params = input_params_ui()


# ------------------------------------------------------------
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
st.write(f'功率链（kW）： {P_I: .2f} -> {P_II: .2f} -> {P_III: .2f}')

I_TOTAL = INPUT_SPEED / input_params['n_out']
i1, i2 = calc_iI_iII(I_TOTAL, input_params['coeff'])
# st.write(f'理想高速级转动比 {i1: .2f}，低速级传动比 {i2: .2f}')

z1: int = input_params['z1']
z3: int = input_params['z3']
z2 = round(z1 * i1)
z4 = round(z3 * i2)


def calc_i1_i2():
    global i1, i2
    i1 = z2 / z1
    i2 = z4 / z3


calc_i1_i2()
st.write(f'粗算高速级转动比 {i1: .2f}，低速级传动比 {i2: .2f}')

# 计算各级转速


def calc_Ns(show=False):
    global N_I, N_II, N_III
    N_I = INPUT_SPEED
    N_II = N_I / i1
    N_III = N_II / i2
    if show:
        st.write(f'转速链（rpm）： {N_I: .2f} -> {N_II: .2f} -> {N_III: .2f}')
        # 计算转速误差
        speed_error = abs(INPUT_SPEED / I_TOTAL - (INPUT_SPEED / (i1 * i2)))
        speed_re = speed_error / (INPUT_SPEED / I_TOTAL) * 100
        if speed_re < 0.005:
            st.write(f'转速误差：{speed_re: .4f}%')
        else:
            st.write(f'转速误差：{speed_re: .2f}%')


calc_Ns(True)
# 计算各级扭矩


def calc_Ts(show=False):
    global T_I, T_II, T_III
    T_I = P_I * 30 / (math.pi * N_I) * 1e3
    T_II = P_II * 30 / (math.pi * N_II) * 1e3
    T_III = P_III * 30 / (math.pi * N_III) * 1e3
    if show:
        st.write(f'扭矩链（N·m）： {T_I: .2f} -> {T_II: .2f} -> {T_III: .2f}')


calc_Ts(True)
# endregion 计算传动参数


# ------------------------------------------------------------
st.markdown('---')
st.header('粗算')

# ------------------------------------------------------------
# region 计算齿形系数
BETA_1: Angle = input_params['beta_I']
BETA_2: Angle = input_params['beta_II']
# print(BETA_1, BETA_2)


def calc_yfa(z: int, beta: Angle) -> float:
    denominator = beta.cos() ** 3
    if denominator == 0:
        return 0.0  # 防止分母为零的情况，具体处理需根据实际需求调整
    zv = z / denominator
    if zv < 35:
        term = (zv - 16) / (beta.cos() ** 3)
        return (-0.0000570752776635208 * (term ** 3) +
                0.00307677616500968 * (term ** 2) -
                0.0688841305752419 * term +
                3.03422577422526)
    if zv < 110:
        term = (zv / 10 - 2) / (beta.cos() ** 3)
        return (-0.00141414141414042 * (term ** 3) +
                0.0267099567099223 * (term ** 2) -
                0.18568542568536 * term +
                2.6785714285711)
    if zv < 160:
        return 2.14
    if zv < 210:
        return 2.12
    return 2.1


def calc_ysa(z: int, beta: Angle) -> float:
    if z < 35:
        term = (z - 16) / (beta.cos() ** 3)
        return (0.0000291375291376905 * (term ** 3) -
                0.00079295704295923 * (term ** 2) +
                0.0139880952381617 * term +
                1.50570429570396)
    if z < 130:
        term = (z / 10 - 2) / (beta.cos() ** 3)
        return (-0.0027083333 * (term ** 2) +
                0.0474107143 * term +
                1.5825892857)
    if z < 160:
        return 1.83
    if z < 210:
        return 1.865
    return 1.9


def calc_yf(z: int, beta: Angle):
    '''
    计算齿形系数 YFa。该方法基于方法 B。精度较低，仅供初步计算使用。
    '''
    return calc_yfa(z, beta) * calc_ysa(z, beta)


def calc_all_yfs_and_show():
    global YF
    # print(z1, z2, z3, z4)
    # print(BETA_1, BETA_2)
    YF = [
        calc_yf(z1, BETA_1),
        calc_yf(z2, BETA_1),
        calc_yf(z3, BETA_2),
        calc_yf(z4, BETA_2)
    ]
    st.table(pd.DataFrame([YF], columns=[
        rf'$Y_{{F_{i + 1}}}$' for i in range(4)
    ]))


st.subheader('齿形系数')
calc_all_yfs_and_show()
# endregion 计算齿形系数


# ------------------------------------------------------------
# region 计算许用值
st.subheader('计算许用接触和弯曲应力')
ZE: float = input_params['ze']
SIG_H_LIM_13 = input_params['sigh_lim_13']
SIG_H_LIM_24 = input_params['sigh_lim_24']
SIG_F_E_13 = input_params['sigf_e_13']
SIG_F_E_24 = input_params['sigf_e_24']

S_MIN_SELECTIONS = [
    '高可靠率（1 / 10,000)',
    '中可靠率（1 / 1,000)',
    '一般可靠率（1 / 100)',
    '低可靠率（1 / 10) 可能在塑性形变前点蚀',
]
S_SEL = S_MIN_SELECTIONS.index(st.selectbox('最小安全系数', S_MIN_SELECTIONS))
S_MIN_H_SELECTIONS = [1.5, 1.25, 1., .85]
S_MIN_F_SELECTIONS = [2., 1.6, 1.25, 1.]
SH_MIN = S_MIN_H_SELECTIONS[S_SEL]
SF_MIN = S_MIN_F_SELECTIONS[S_SEL]
st.table(pd.DataFrame([[SF_MIN, SH_MIN]], columns=[
         r'$S_{H_{min}}$', r'$S_{F_{min}}$']))

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


def calc_sigma_h(
    sigh_lim: float, sh_min: float,
    N: float, type: int, exp_adjust=0.
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
            'N': [1e-10, 6e5, 1e7, 1e9, 1e10],
            'ZNT': [1.6, 1.6, 1.3, 1.0, adjust_val]
        },
        {
            'N': [1e-10, 1e5, 5e7, 1e10],
            'ZNT': [1.6, 1.6, 1.0, adjust_val]
        },
        {
            'N': [1e-10, 1e5, 2e6, 1e10],
            'ZNT': [1.3, 1.3, 1.0, adjust_val]
        },
        {
            'N': [1e-10, 1e5, 2e6, 1e10],
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
    N: float, type: int, exp_adjust=0.
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
            'N': [1e-10, 1e4, 3e6, 1e10],
            'YNT': [2.5, 2.5, 1.0, adjust_val]
        },
        {
            'N': [1e-10, 1e3, 3e6, 1e10],
            'YNT': [2.5, 2.5, 1.0, adjust_val]
        },
        {
            'N': [1e-10, 1e3, 3e6, 1e10],
            'YNT': [1.6, 1.6, 1.0, adjust_val]
        },
        {
            'N': [1e-10, 1e3, 3e6, 1e10],
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

SIGH = [SIGH[i][2] for i in range(4)]

YF_DIV_SIGF = [YF[i] / SIGF[i][2] for i in range(4)]
st.table(pd.DataFrame([[
    f'{v: .4e}' for v in YF_DIV_SIGF]], columns=[
    rf'$\frac{{Y_{{F_{i + 1}}}}}{{\sigma_{{F_{i + 1}}}}}$' for i in range(4)
]))
# endregion 计算许用值


# ------------------------------------------------------------
# region 计算最小值
k: float = input_params['k']
ALPHA_N = Angle(20)


def calc_dmin(
    t_: float, u_: float,
    k_: float, sigh: float,
    beta: Angle
) -> float:
    """
    计算齿轮的最小直径 d_min。

    Parameters
    ----------
        t_ (float): 小齿轮扭矩，单位为牛米 (Nm)。

        k_ (float): 载荷系数，考虑不同工况下的载荷变化。

        i (float): 传动比，即大齿轮与小齿轮的齿数比。

        sigh (float): 接触疲劳强度，单位为兆帕 (MPa)。

        beta (Angle): 斜齿轮螺旋角，表示齿轮齿的倾斜角度。

    Returns
    -------
        float: 计算得到的最小直径 d_min，单位为毫米 (mm)。

    Notes
    -------
        - 该计算基于方法 B。
        - 确保所有输入单位一致，以便获得正确的结果。
    """
    zh = 2.5  # 计算区域系数。例如，普通圆柱齿轮通常为 2.5。
    if float(beta) >= 7.:
        # 认为只有 β >= 7° 才算斜齿轮
        alpha_t = math.atan(ALPHA_N.tan() / beta.cos())
        beta_b = math.atan(beta.tan() * math.cos(alpha_t))
        alpha_t_1 = alpha_t  # 没有变位
        zh = math.sqrt(2 * math.cos(beta_b) * math.cos(alpha_t_1) / (
            math.cos(alpha_t) ** 2 * math.sin(alpha_t_1)
        ))  # 计算区域系数
    t_ = 1e3 * t_  # N·mm
    # print(k_, t_, u_, beta, zh, ZE, sigh)
    return (
        2 * k_ * t_ / PHI_D *
        (u_ + 1) / u_ * beta.cos() *
        (zh * ZE / sigh) ** 2
    ) ** (1 / 3)


def calc_mmin_tol(
    t_: float, z_: float, k_: float,
    yf_div_sigf: float, beta: Angle
) -> float:
    """
    计算齿轮的最小模数 m_min。

    Parameters
    ----------
        t_ (float): 小齿轮扭矩，单位为牛米 (Nm)。

        z_ (float): 齿数。

        k_ (float): 载荷系数，考虑不同工况下的载荷变化。

        yf_div_sigf (float): 齿形系数与应力系数的比值。

        beta (Angle): 斜齿轮螺旋角，表示齿轮齿的倾斜角度。

    Returns
    -------
        float: 计算得到的最小模数 m_min。

    Notes
    -------
        - 该计算基于方法 B。
        - 确保所有输入单位一致，以便获得正确
    """
    t_ = 1e3 * t_  # N·mm
    zv = z_ / beta.cos()  # 修正齿数
    return (2 * k_ * t_ / PHI_D / zv**2 * yf_div_sigf) ** (1 / 3)


st.subheader('计算最小直径')
D_MIN = [
    calc_dmin([T_I, T_II, T_II, T_III][i],
              [i1, i2][i // 2], k,
              [min(SIGH[0], SIGH[1]), min(SIGH[2], SIGH[3])][i // 2],
              [BETA_1, BETA_2][i // 2]
              ) for i in range(4)
]
diameters = [D_MIN[0], D_MIN[0] * i1, D_MIN[2], D_MIN[2] * 2]
st.table(pd.DataFrame([D_MIN, diameters], index=[
    '原始计算值', '传动比计算的大轮'
], columns=[
    rf'$d_{{min_{i + 1}}}$' for i in range(4)
]))
# endregion 计算最小值


st.markdown('---')
st.header('精算')


# ------------------------------------------------------------
# region 选取模数、中心距
st.subheader('选取模数')
M_SERIES = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6]
zs = [z1, z2, z3, z4]
MS_RAW = [[BETA_1.cos(), BETA_2.cos()][i // 2] * di / z
          for i, (di, z) in enumerate(zip(diameters, zs))]
# 找到最接近的模数


def select_m_value(mi):
    return min(M_SERIES, key=lambda x: x - mi if x > mi else math.inf)


ms = [select_m_value(mi) for mi in MS_RAW]
ms[0] = ms[1] = max(ms[0], ms[1])
ms[2] = ms[3] = max(ms[2], ms[3])
st.table(pd.DataFrame([MS_RAW, ms], index=[
    '$z, d$ 计算 $m$', '取用'
], columns=[
    rf'$m_{i + 1}$' for i in range(4)
]))
st.write(r'计算最小模数（注意：为了方便，计算使用 $Y_\beta = 1$ 作为安全冗余）')
M_MIN = [
    calc_mmin_tol([T_I, T_II, T_II, T_III][i],
                  [z1, z2, z3, z4][i],
                  k, YF_DIV_SIGF[i],
                  [BETA_1, BETA_2][i // 2]
                  ) for i in range(4)
]
M_PASSED = ['OK' if ms[i] > M_MIN[i] else 'Failed' for i in range(4)]
M_MIN = [
    f'{mi: .2f}' for mi in M_MIN
]
st.table(pd.DataFrame([M_MIN, M_PASSED], index=[
    '计算最小值', '是否通过'
], columns=[
    rf'$m_{{min_{i + 1}}}$' for i in range(4)
]))
ms[0] = ms[1] = st.select_slider(
    '精调高速级模数', M_SERIES[M_SERIES.index(ms[0]):], ms[0])
ms[2] = ms[3] = st.select_slider(
    '精调低速级模数', M_SERIES[M_SERIES.index(ms[2]):], ms[2])


st.subheader('选取中心距')


def calc_ds(zs, ms):
    ds = []
    for i in range(4):
        ds.append(zs[i] * ms[i] / [BETA_1, BETA_2][i // 2].cos())
    return ds


diameters = calc_ds(zs, ms)
D_STR = [f'{d: .2f}' for d in diameters]
st.table(pd.DataFrame([D_STR], columns=[
    rf'$d_{i + 1}$' for i in range(4)
]))
z1, z2, z3, z4 = zs
a1 = (diameters[0] + diameters[1]) / 2
a2 = (diameters[2] + diameters[3]) / 2

if float(BETA_1) < 8.:
    a1_delta = 5 * ms[0] / math.gcd(5, ms[0])
    a1_round = round(a1 / a1_delta) * a1_delta
    a1_new = st.number_input('高速级中心距精调', value=a1_round, step=a1_delta)
    z_12_delta = 2 * a1_new / ms[0] - zs[0] - zs[1]
    z1 = z1 + z_12_delta // 2
    z2 = z2 + (z_12_delta + 1) // 2
    z12_bias = st.number_input('高速级齿数增减偏置：', value=0)
    z1 += z12_bias
    z2 -= z12_bias
else:
    z1 = st.select_slider('$Z_1$', range(z1 - 10, z1 + 12), value=z1)
    z2 = st.select_slider('$Z_2$', range(z2 - 10, z2 + 12), value=z2)
    zs[0], zs[1] = z1, z2
    diameters = calc_ds(zs, ms)
    a1_new = (diameters[0] + diameters[1]) / 2
    a1_round = round(a1_new / 5) * 5
    BETA_1 = Angle(math.acos(ms[0] * (z1 + z2) / 2 / a1_round) / math.pi * 180)
    st.write(rf'高速级 $\beta$ ：{BETA_1}')
st.write(f'高速级中心距：{a1: .2f} -> {a1_round}')

if float(BETA_2) < 8.:
    a2_delta = 5 * ms[2] / math.gcd(5, ms[2])
    a2_round = round(a2 / a2_delta) * a2_delta
    a2_new = st.number_input('低速级中心距精调', value=a2_round, step=a2_delta)
    z_34_delta = 2 * a2_new / ms[2] - zs[2] - zs[3]
    z3 = z3 + z_34_delta // 2
    z4 = z4 + (z_34_delta + 1) // 2
    z34_bias = st.number_input('低速级齿数增减偏置：', value=0)
    z3 += z34_bias
    z4 -= z34_bias
else:
    z3 = st.select_slider('$Z_3$', range(z3 - 10, z3 + 12), value=z3)
    z4 = st.select_slider('$Z_4$', range(z4 - 10, z4 + 12), value=z4)
    zs[2], zs[3] = z3, z4
    diameters = calc_ds(zs, ms)
    a2_new = (diameters[2] + diameters[3]) / 2
    a2_round = round(a2_new / 5) * 5
    BETA_2 = Angle(math.acos(ms[0] * (z3 + z4) / 2 / a2_round) / math.pi * 180)
    st.write(rf'高速级 $\beta$ ：{BETA_1}')
st.write(f'低速级中心距：{a2: .2f} -> {a2_round}')

zs = [z1, z2, z3, z4]
zs = [int(zi) for zi in zs]
diameters = calc_ds(zs, ms)
ZS_STR = [str(zi) for zi in zs]
D_STR = [f'{d: .2f}' for d in diameters]
st.table(pd.DataFrame([ZS_STR, D_STR], index=[
    '齿数', '直径'
], columns=[
    f'齿轮 {i + 1}' for i in range(4)
]))
st.write(r'提示：如果发现 $d_2$ $d_4$ 差的比较大，调最上面的传动比分配系数。系数越大，$\frac{d_2}{d_4}$ 越大')

st.subheader('传动参数')
calc_i1_i2()
st.write(f'再算高速级转动比 {i1: .2f}，低速级传动比 {i2: .2f}')
calc_Ns(True)
calc_Ts(True)
st.subheader('齿形系数')
calc_all_yfs_and_show()
YF_DIV_SIGF = [YF[i] / SIGF[i][2] for i in range(4)]
st.table(pd.DataFrame([[
    f'{v: .4e}' for v in YF_DIV_SIGF]], columns=[
    rf'$\frac{{Y_{{F_{i + 1}}}}}{{\sigma_{{F_{i + 1}}}}}$' for i in range(4)
]))
# endregion 选取模数、中心距


# ------------------------------------------------------------
# region 计算载荷系数
k_param = {
    'kA': 1.,
    'ka': 1.,
    'kb': 1.,
    'kv': 1.,
}

k_param['kA'] = st.number_input(r'$K_A$', value=1.)
k_param['kb'] = st.number_input(r'$K_\beta$', value=1.)

k_param = [dict(k_param), dict(k_param)]
st.write('暂支支持 8 级齿轮')

force_12 = 1e3 * T_I / (diameters[0] / 2)
force_34 = 1e3 * T_II / (diameters[2] / 2)
kfb_12 = force_12 / (diameters[0] * PHI_D)
kfb_34 = force_34 / (diameters[2] * PHI_D)
KFB_SHOW = [[f'{kfb_12: .2f}', f'{kfb_34: .2f}']]
st.table(pd.DataFrame(KFB_SHOW, columns=[
    rf'$\frac{{K_A F_{{t_{i + 1}}}}}{{b_{i + 1}}}$' for i in [0, 2]
]))
k_param[0]['ka'] = st.number_input(r'高速级 $K_\alpha$', value=1.)
k_param[1]['ka'] = st.number_input(r'低速级 $K_\alpha$', value=1.)

vel_12 = math.pi * N_I * diameters[0] / 60e3
vel_34 = math.pi * N_II * diameters[2] / 60e3
st.write(f'$v_I$：{vel_12: .2f}，$v_{{II}}$：{vel_34: .2f}')
k_param[0]['kv'] = st.number_input('高速级动载系数', value=1.)
k_param[1]['kv'] = st.number_input('低速级动载系数', value=1.)

k_param = [
    np.prod(list(k_param[0].values())),
    np.prod(list(k_param[1].values()))
]

st.write(f'$K_I$：{k_param[0]: .4f}，$K_{{II}}$：{k_param[1]: .4f}')

D_MIN = [
    calc_dmin([T_I, T_II, T_II, T_III][i],
              [i1, i2][i // 2],
              k_param[i // 2],
              [min(SIGH[0], SIGH[1]), min(SIGH[2], SIGH[3])][i // 2],
              [BETA_1, BETA_2][i // 2]
              ) for i in range(4)
]
D_PASSED = ['√' if diameters[i] > D_MIN[i] else '×' for i in range(4)]
D_MIN = [f'{di: .2f}' for di in D_MIN]
st.table(pd.DataFrame([D_STR, D_MIN, D_PASSED], index=[
    '当前值', '计算最小值', '是否通过'
], columns=[
    rf'$d_{{min_{i + 1}}}$' for i in range(4)
]))
M_MIN = [
    calc_mmin_tol([T_I, T_II, T_II, T_III][i],
                  zs[i], k_param[i // 2],
                  YF_DIV_SIGF[i],
                  [BETA_1, BETA_2][i // 2]
                  ) for i in range(4)
]
M_PASSED = ['√' if ms[i] > M_MIN[i] else '×' for i in range(4)]
M_MIN = [f'{mi: .2f}' for mi in M_MIN
         ]
M_STR = [f'{mi: .2f}' for mi in ms]
st.table(pd.DataFrame([M_STR, M_MIN, M_PASSED], index=[
    '当前值', '计算最小值', '是否通过'
], columns=[
    rf'$m_{{min_{i + 1}}}$' for i in range(4)
]))
# endregion 计算载荷系数


# ------------------------------------------------------------
# region 计算所有尺寸
# 上文定义了 Zs, diameters, ms
def calc_gear(m_n, d1, d2, beta, z1, z2, nin, tin):
    def try_int_parse(x):
        return str(int(x)) if x == int(x) else f'{x: .3f}'
    ha_star, c_star = 1, 0.25
    # 齿根高
    hf = (ha_star + c_star) * m_n
    # 齿顶高
    ha = ha_star * m_n
    # 全齿高
    h = ha + hf
    # 顶隙
    c = try_int_parse(c_star * m_n)
    # 节圆直径（标准安装）
    d_prime = try_int_parse(min(d1, d2))
    b2 = try_int_parse(d_prime * PHI_D)
    b1 = try_int_parse(b2 + 6)
    # 传动比
    i = f'{d2 / d1: .3f}'
    # 中心距
    a = try_int_parse((d1 + d2) / 2)
    # 模数，法向压力角，螺旋角，分度圆直径，齿根高，齿顶高，全齿高，齿顶圆直径，齿根圆直径，顶隙，中心距，节圆直径，传动比
    d = d1
    # 齿顶圆直径
    da = d + 2 * (int(ha) if ha == int(ha) else ha)
    da = try_int_parse(da)
    # 齿根圆直径
    df = d - 2 * (int(hf) if hf == int(hf) else hf)
    df = try_int_parse(df)
    force = try_int_parse(tin / (d / 2))
    gear1 = [
        str(m_n), '20', str(beta), try_int_parse(d),
        try_int_parse(hf), try_int_parse(ha), try_int_parse(h),
        da, df, c, a, d_prime, i, b1, force, str(z1), try_int_parse(nin)
    ]
    d = d2
    # 齿顶圆直径
    da = d + 2 * (int(ha) if ha == int(ha) else ha)
    da = try_int_parse(da)
    # 齿根圆直径
    df = d - 2 * (int(hf) if hf == int(hf) else hf)
    df = try_int_parse(df)
    force = try_int_parse(tin / (d / 2))
    nin = try_int_parse(nin / i)
    gear2 = [
        str(m_n), '20', str(beta), try_int_parse(d),
        try_int_parse(hf), try_int_parse(ha), try_int_parse(h),
        da, df, c, a, d_prime, i, b2, force, str(z2), nin
    ]
    return gear1, gear2


small_gear, big_gear = calc_gear(
    ms[0], diameters[0],
    diameters[1], BETA_1,
    z1, z2, N_I, T_I)
gears = [small_gear, big_gear]
small_gear, big_gear = calc_gear(
    ms[2], diameters[2],
    diameters[3], BETA_2,
    z2, z3, N_II, T_II)
gears = [*gears, small_gear, big_gear]
index_names = [
    "模数 (mm)", "法向压力角 (°)", "螺旋角 (度分秒)",
    "分度圆直径 (mm)", "齿根高 (mm)", "齿顶高 (mm)",
    "全齿高 (mm)", "齿顶圆直径 (mm)", "齿根圆直径 (mm)",
    "顶隙 (mm)", "中心距 (mm)", "节圆直径 (mm)",
    "传动比", "齿宽 (mm)", "齿数", "转速 (rpm)",
    "切向力 (N)", '径向力 (N)', '轴向力 (N)',
]
gear_names = ['高速级小齿轮', '高速级大齿轮', '低速级小齿轮', '低速级大齿轮']
gears = list(zip(*gears))
table = pd.DataFrame(gears, index_names, gear_names)
table = table.rename_axis('项目')
st.table(table)

file_bytes = io.BytesIO()
table.to_excel(file_bytes)
st.download_button('下载最终数据表格', file_bytes, '减速器数据.xlsx')
# endregion
