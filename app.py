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
    if 'input_params' not in st.session_state:
        st.session_state.input_params = controller.get('input_params')
    _input_params = st.session_state.input_params
    INIT_PARAMS = {
        'k': 1., 'time_mode': 0, 'years': 0.,
        'p_in': 5., 'n_in': 100., 'n_out': 1., 'phid': 1., 'coeff': 1.,
        'eta_I': 1., 'eta_II': 1.,
        'z1': 24, 'z3': 24, 'beta_I': 0, 'beta_II': 0,
        'sigh_lim_13': 100., 'sigh_lim_24': 100.,
        'sigf_e_13': 100., 'sigf_e_24': 100., 'ze': 100.
    }
    if _input_params is None:
        _input_params = INIT_PARAMS
    else:
        para_keys = INIT_PARAMS.keys()
        for k in para_keys:
            if k not in _input_params:
                _input_params[k] = INIT_PARAMS[k]

    for k in _input_params:
        if k not in ('z1', 'z3', 'time_mode'):
            _input_params[k] = float(_input_params[k])

    tabs = st.tabs(['工况参数', '材料参数', '尺寸参数', '传动参数'])
    with st.container(border=True):
        with tabs[0]:
            _input_params['k'] = st.number_input(
                '初选工况系数', value=_input_params['k'])
            TIME_MODES = ['单班制', '双班制']
            _input_params['time_mode'] = TIME_MODES.index(
                st.selectbox('选择班制：', TIME_MODES, _input_params['time_mode']))
            _input_params['years'] = st.number_input(
                '工作年数：', value=_input_params['years'])
            _input_params['beta_I'] = _input_params['beta_I']
            _input_params['beta_II'] = _input_params['beta_II']

        with tabs[1]:
            _input_params['sigh_lim_13'] = st.number_input(
                r'小齿轮接触极限 $\text{(MPa)}$', value=_input_params['sigh_lim_13'])
            _input_params['sigf_e_13'] = st.number_input(
                r'小齿轮抗弯极限 $\text{(MPa)}$', value=_input_params['sigf_e_13'])
            _input_params['sigh_lim_24'] = st.number_input(
                r'大齿轮接触极限 $\text{(MPa)}$', value=_input_params['sigh_lim_24'])
            _input_params['sigf_e_24'] = st.number_input(
                r'大齿轮抗弯极限 $\text{(MPa)}$', value=_input_params['sigf_e_24'])
            _input_params['ze'] = st.number_input(
                r'$Z_e$ ($\sqrt{MPa}$))', value=_input_params['ze'])

        with tabs[2]:
            _input_params['z1'] = st.number_input(
                '低速级小齿轮齿数', value=_input_params['z1'])
            _input_params['z3'] = st.number_input(
                '高速级小齿轮齿数', value=_input_params['z3'])
            _input_params['beta_I'] = st.number_input(
                '高速级螺旋角 β (°)', value=_input_params['beta_I'])
            _input_params['beta_II'] = st.number_input(
                '低速级螺旋角 β (°)', value=_input_params['beta_II'])

        with tabs[3]:
            _input_params['p_in'] = st.number_input(
                r'输入功率 $P$ $\text{(kW)}$', value=_input_params['p_in'])
            _input_params['n_in'] = st.number_input(
                r'输入转速 $n$ $\text{(rpm)}$', value=_input_params['n_in'])
            _input_params['n_out'] = st.number_input(
                r'目标转速 $n_{out}$ $\text{(rpm)}$', value=_input_params['n_out'])
            _input_params['eta_I'] = st.number_input(
                r'$\text{I}$ 轴效率', value=_input_params['eta_I'])
            _input_params['eta_II'] = st.number_input(
                r'$\text{II}$ 轴效率', value=_input_params['eta_II'])
            _input_params['phid'] = st.number_input(
                '宽度系数 $Φ_d$', value=_input_params['phid'])
            _input_params['coeff'] = st.number_input(
                '传动比分配系数', value=_input_params['coeff'])

    if st.button('保存所有数据'):
        controller.set('input_params', _input_params)

    _input_params['beta_I'] = Angle(_input_params['beta_I'])
    _input_params['beta_II'] = Angle(_input_params['beta_II'])

    return _input_params


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

Z1: int = input_params['z1']
Z3: int = input_params['z3']

time_per_day = [8, 16][input_params['time_mode']]
n_years = input_params['years']
time_hours = n_years * 365 * time_per_day

gears = GearDraft.create_gears(time_hours, Z1, Z1 * i1, Z3, Z3 * i2)
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
# endregion 计算最小值


st.markdown('---')
st.header('精算')


# ------------------------------------------------------------
# region 选取模数、中心距
check_result = GearDraft.batch_calc(gears, CalcType.CHECK_BENDING)
st.table(pack_table(check_result))

mI = st.select_slider(
    '精调高速级模数', GearDraft.M_SERIES[GearDraft.M_SERIES.index(
        solve_result[0].module):], solve_result[0].module)
mII = st.select_slider(
    '精调低速级模数', GearDraft.M_SERIES[GearDraft.M_SERIES.index(
        solve_result[2].module):], solve_result[2].module)


st.subheader('选取中心距')
GearDraft.batch_calc(gears, CalcType.ALL)
aI = (gears[0].d + gears[1].d) / 2
aII = (gears[2].d + gears[3].d) / 2


def create_z_slider(z, i, s=10):
    return st.select_slider(
        f'$Z_{{\\text{i}}}$', range(z - s, z + s + 1), value=z)


def check_a_for(gear1, gear2):
    _gears = [gear1, gear2]
    if gear1.is_helical():  # 斜齿轮
        z1 = create_z_slider(gear1.z, 1)
        z2 = create_z_slider(gear2.z, 2)
        GearDraft.set_val(_gears, 'z', z1, z2)
        GearDraft.batch_calc(_gears, CalcType.ALL)
        aI_new = (gear1.d + gear2.d) / 2
        aI_new = round(aI_new / 5) * 5
        GearDraft.set_val(_gears, 'a', [aI_new], [aI_new])
        size = GearDraft.batch_calc(_gears, CalcType.ADJUST_BY_CENTER_DIST)
        st.write(rf'$\beta$ ：{size[0].beta}')
    else:  # 直齿轮
        m = gear1.module
        aI_delta = 5 * m / math.gcd(5, m)
        aI_round = round(aI / aI_delta) * aI_delta
        aI_new = st.number_input('中心距精调', value=aI_round, step=aI_delta)
        z1, z2 = gear1.z, gear2.z
        z_delta = 2 * aI_new / m - z1 - z2
        z1 = z1 + z_delta // 2
        z2 = z2 + (z_delta + 1) // 2
        z_bias = st.number_input('齿数增减偏置：', value=0)
        GearDraft.set_val(_gears, 'z', z1 + z_bias, z2 - z_bias)
        GearDraft.batch_calc(_gears, CalcType.ALL)
    st.write(f'中心距：{aI: .2f} -> {aI_new}')


st.subheader('高速级精调')
check_a_for(*gears[:2])
st.subheader('低速级精调')
check_a_for(*gears[2:])


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
