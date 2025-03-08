import math
import io
import streamlit as st
from streamlit_cookies_controller import CookieController
import scipy.optimize as opt
import numpy as np
import pandas as pd

from modeling import (
    Angle, GearDraft,
    CalcType, MaterialType,
    pack_table
)


def input_params_ui():
    st.title('软齿轮系计算')

    controller = CookieController(key='cookies')
    if st.button('加载上次保存的参数'):
        input_params = controller.get('input_params')
    else:
        input_params = None
    INIT_PARAMS = {
        'k': 1., 'time_mode': 0, 'years': 0.,
        'p_in': 5., 'n_in': 100., 'n_out': 1., 'phid': 1., 'coeff': 1.,
        'eta_I': 1., 'eta_II': 1.,
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

    tabs = st.tabs(['工况参数', '材料参数', '尺寸参数', '传动参数'])
    with st.container(border=True):
        with tabs[0]:
            input_params['k'] = st.number_input('初选工况系数', value=input_params['k'])
            TIME_MODES = ['单班制', '双班制']
            input_params['time_mode'] = TIME_MODES.index(
                st.selectbox('选择班制：', TIME_MODES, input_params['time_mode']))
            input_params['years'] = st.number_input(
                '工作年数：', value=input_params['years'])
            input_params['beta_I'] = input_params['beta_I']
            input_params['beta_II'] = input_params['beta_II']

        with tabs[1]:
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

        with tabs[2]:
            input_params['z1'] = st.number_input(
                '低速级小齿轮齿数', value=input_params['z1'])
            input_params['z3'] = st.number_input(
                '高速级小齿轮齿数', value=input_params['z3'])
            input_params['beta_I'] = st.number_input(
                '高速级螺旋角 β (°)', value=input_params['beta_I'])
            input_params['beta_II'] = st.number_input(
                '低速级螺旋角 β (°)', value=input_params['beta_II'])

        with tabs[3]:
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
            input_params['phid'] = st.number_input(
                '宽度系数 $Φ_d$', value=input_params['phid'])
            input_params['coeff'] = st.number_input(
                '传动比分配系数', value=input_params['coeff'])

    if st.button('保存所有数据'):
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

I_TOTAL = INPUT_SPEED / input_params['n_out']
i1, i2 = calc_iI_iII(I_TOTAL, input_params['coeff'])
# st.write(f'理想高速级转动比 {i1: .2f}，低速级传动比 {i2: .2f}')

z1: int = input_params['z1']
z3: int = input_params['z3']

time_per_day = [8, 16][input_params['time_mode']]
n_years = input_params['years']
time_hours = n_years * 365 * time_per_day

gears = GearDraft.create_gears(time_hours, z1, z1 * i1, z3, z3 * i2)
GearDraft.set_val(
    gears, 'phid',
    PHI_D, PHI_D, PHI_D, PHI_D
)

ETA_I: float = input_params['eta_I']
ETA_II: float = input_params['eta_II']
GearDraft.set_val(
    gears, 'eta',
    ETA_I, 1, ETA_II, 1
)

# 计算各级运动学参数
gears[0].power = INPUT_POWER
gears[0].speed = INPUT_SPEED
g1 = gears[0] @ gears[1]
g2 = g1 - gears[2]
g3 = g2 @ gears[3]

i1 = gears[0].gear_ratio(g1)
i2 = gears[2].gear_ratio(g3)
st.write(f'粗算高速级转动比 {i1: .2f}，低速级传动比 {i2: .2f}')

kine = GearDraft.batch_calc(gears, CalcType.KINEMATICS)
st.subheader('运动学参数')
table = pack_table(kine)
st.table(table)

# endregion 计算传动参数


# ------------------------------------------------------------
st.markdown('---')
st.header('粗算')

# ------------------------------------------------------------
# region 计算齿形系数
BETA_1: Angle = input_params['beta_I']
BETA_2: Angle = input_params['beta_II']
GearDraft.set_val(
    gears, 'beta',
    BETA_1, BETA_1, BETA_2, BETA_2
)


st.subheader('齿形系数')
form_factors = GearDraft.batch_calc(gears, CalcType.FORM_FACTOR)
form_factors = pack_table(form_factors)
st.table(form_factors)
# endregion 计算齿形系数


# ------------------------------------------------------------
# region 计算许用值
st.subheader('计算许用接触和弯曲应力')
ZE: float = input_params['ze']
SIG_H_LIM_13 = input_params['sigh_lim_13']
SIG_H_LIM_24 = input_params['sigh_lim_24']
SIG_F_E_13 = input_params['sigf_e_13']
SIG_F_E_24 = input_params['sigf_e_24']

GearDraft.set_val(
    gears, 'stress_lim',
    *[{'contact': c, 'bending': b} for c, b in zip(
        [SIG_H_LIM_13, SIG_H_LIM_24, SIG_H_LIM_13, SIG_H_LIM_24],
        [SIG_F_E_13, SIG_F_E_24, SIG_F_E_13, SIG_F_E_24]
    )]
)
GearDraft.set_val(
    gears, 'z_elastic',
    ZE, ZE, ZE, ZE
)

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
GearDraft.set_val(
    gears, 'safe_factor',
    *([{'contact': SH_MIN, 'bending': SF_MIN}] * 4))
st.markdown(
    '| $S_{H_{min}}$ | $S_{F_{min}}$ |\n| :-: | :-: |\n' +
        f'| {SH_MIN} | {SF_MIN} |')

m13 = st.selectbox('选择你的小齿轮材料类型：', MaterialType.TYPES)
m24 = st.selectbox('选择你的大齿轮材料类型：', MaterialType.TYPES)
GearDraft.set_val(
    gears, 'material',
    *[MaterialType.index(mtype) for mtype in [m13, m24, m13, m24]])

st.subheader('接触疲劳强度计算')
st.write('寿命系数计算方法参考 GB/T 6366-2019 中的表格。')
material_table = GearDraft.batch_calc(gears, CalcType.MATERIAL)
st.table(pack_table(material_table))
# endregion 计算许用值


# ------------------------------------------------------------
# region 计算最小值
k: float = input_params['k']
GearDraft.set_val(gears, 'k', k, k, k, k)

st.subheader('计算最小直径')
solve_result = GearDraft.batch_calc(gears, CalcType.SOLVE_CONTACT)
st.table(pack_table(solve_result))

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
