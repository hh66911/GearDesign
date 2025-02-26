import math
import streamlit as st
import scipy.optimize as opt


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
    
    input_params = {
        'p_in': 0., 'n_in': 0., 'i': 0., 'phid': 1., 'coeff': 1.,
        'eta_I': 1., 'eta_II': 1., 'eta_III': 1.,
        'z1': 24, 'z3': 24, 'k': 1.,
        'beta_I': Angle(0), 'beta_II': Angle(0),
        'sigh_lim_13': 0., 'sigh_lim_24': 0.,
        'sigf_e_13': 0., 'sigf_e_24': 0.,
    }

    st.header('材料参数')
    input_params['sigh_lim_13'] = st.number_input('小齿轮接触极限 (MPa)', value=input_params['sigh_lim_13'])
    input_params['sigf_e_13'] = st.number_input('小齿轮抗弯极限 (MPa)', value=input_params['sigf_e_13'])
    input_params['sigh_lim_24'] = st.number_input('大齿轮接触极限 (MPa)', value=input_params['sigh_lim_24'])
    input_params['sigf_e_24'] = st.number_input('大齿轮抗弯极限 (MPa)', value=input_params['sigf_e_24'])
    
    st.header('尺寸参数')
    input_params['z1'] = st.number_input('低速级小齿轮齿数', value=input_params['z1'])
    input_params['z3'] = st.number_input('高速级小齿轮齿数', value=input_params['z3'])
    input_params['beta_I'] = Angle(st.number_input('高速级螺旋角 β (°)',
                                                 value=float(input_params['beta_I'])))
    input_params['beta_I'] = Angle(st.number_input('低速级螺旋角 β (°)',
                                                 value=float(input_params['beta_II'])))
    
    st.header('传动参数')
    input_params['k'] = st.number_input('初选工况系数', value=input_params['k'])
    input_params['p_in'] = st.number_input('输入功率 P (kW)', value=input_params['p_in'])
    input_params['n_in'] = st.number_input('输入转速 n (rpm)', value=input_params['n_in'])
    input_params['i'] = st.number_input('总传动比 i', value=input_params['i'])
    input_params['eta_I'] = st.number_input('I轴效率', value=input_params['eta_I'])
    input_params['eta_II'] = st.number_input('II轴效率', value=input_params['eta_II'])
    input_params['eta_III'] = st.number_input('III轴效率', value=input_params['eta_III'])
    input_params['phid'] = st.number_input('宽度系数 Φd', value=input_params['phid'])
    input_params['coeff'] = st.number_input('传动比分配系数', value=input_params['coeff'])

    st.session_state.input_params = input_params


input_params_ui()


def calc_iI_iII(i_total: float, coeff: float) -> tuple[float, float]:
    # 解方程：C * (iI + 1) * iI^4 / ((iI + i_total) * i_total^2) - 1 = 0
    def func(iI: float):
        return coeff * (iI + 1) * iI ** 4 / ((iI + i_total) * i_total ** 2)
    iI: float = opt.fsolve(lambda x: func(x) - 1, i_total**0.5)[0]
    iII: float = i_total / iI
    return iI, iII


input_params = st.session_state.input_params
INPUT_POWER: float = input_params['p_in']
INPUT_SPEED: float = input_params['n_in']
PHI_D: float = input_params['phid']

ETA_I: float = input_params['eta_I']
ETA_II: float = input_params['eta_II']
ETA_III: float = input_params['eta_III']
P_I = INPUT_POWER * ETA_I
P_II = P_I * ETA_II
P_III = P_II * ETA_III

i1, i2 = calc_iI_iII(input_params['i'], input_params['coeff'])
st.write(f'高速级转动比 {i1: .2f}，低速级传动比 {i2: .2f}')

z2 = z1 * i1
z4 = z3 * i2

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


verify_passed = True
def fail_verify(reason):
    global verify_passed
    verify_passed = False
    st.error(reason, icon='🚨')


def calc_yfa(z, beta):
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


def calc_ysa(z, beta):
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
    

k: float = input_params['k']

    
def verify():
    d1n = (2 * k * T_I * ())