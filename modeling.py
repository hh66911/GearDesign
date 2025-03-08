from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from functools import total_ordering


class MaterialType:
    TYPES = [
        '允许一定点蚀的结构钢；调质钢；球墨铸铁（珠光体、贝氏体）；珠光体和可锻铸铁；渗碳淬火的渗碳钢',
        '结构钢；调质钢；渗碳淬火钢；火焰、感应淬火；球墨铸铁；珠光体、可锻铁',
        '灰铸铁；球墨铸铁（铁素体）；渗氮钢、调质钢、渗碳钢',
        '碳氮共渗钢、渗碳钢'
    ]

    ZNT_TABLE = [
        [
            [1e-10, 6e5, 1e7, 1e9, 1e10],
            [1.6, 1.6, 1.3, 1.0, None]
        ],
        [
            [1e-10, 1e5, 5e7, 1e10],
            [1.6, 1.6, 1.0, None]
        ],
        [
            [1e-10, 1e5, 2e6, 1e10],
            [1.3, 1.3, 1.0, None]
        ],
        [
            [1e-10, 1e5, 2e6, 1e10],
            [1.1, 1.1, 1.0, None]
        ]
    ]

    YNT_TABLE = [
        [
            [1e-10, 1e4, 3e6, 1e10],
            [2.5, 2.5, 1.0, None]
        ],
        [
            [1e-10, 1e3, 3e6, 1e10],
            [2.5, 2.5, 1.0, None]
        ],
        [
            [1e-10, 1e3, 3e6, 1e10],
            [1.6, 1.6, 1.0, None]
        ],
        [
            [1e-10, 1e3, 3e6, 1e10],
            [1.1, 1.1, 1.0, None]
        ]
    ]

    def __init__(self, idx):
        self.idx = idx

    @staticmethod
    def index(val):
        for i, t in enumerate(MaterialType.TYPES):
            if t == val:
                return MaterialType(i)
        raise ValueError(f"没有找到 '{val}' 的类型。")

    def _calculate_life_factor(
        self, N, exp_adjust, table
    ):
        adjust_val = np.interp(exp_adjust, [0., 1.], [0.85, 1.0])

        material = table[self.idx]
        N_values = material[0]
        coeff_values = material[1].copy()
        # 使用经验系数更新阴影部分的曲线
        coeff_values[-1] = adjust_val

        log_N = np.log10(N)
        log_N_values = np.log10(N_values)

        coeff = np.interp(log_N, log_N_values, coeff_values).item()
        return coeff

    def calc_contact_limit(self, limit, smin, N, exp_adjust=0.):
        life_factor = self._calculate_life_factor(
            N, exp_adjust, MaterialType.ZNT_TABLE)
        return limit / smin * life_factor, life_factor

    def calc_bending_limit(self, limit, smin, N, exp_adjust=0.):
        life_factor = self._calculate_life_factor(
            N, exp_adjust, MaterialType.YNT_TABLE)
        return limit / smin * life_factor, life_factor


@total_ordering
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
        return np.deg2rad(self._degrees)

    def __eq__(self, other):
        return self._degrees == other._degrees

    def __lt__(self, other):
        return self._degrees < other._degrees

    def sin(self):
        # 计算正弦值
        return np.sin(self.to_radians())

    def cos(self):
        # 计算余弦值
        return np.cos(self.to_radians())

    def tan(self):
        return np.tan(self.to_radians())


def try_parse_intstr(fval: float, decimals=2):
    if isinstance(fval, str):
        return fval
    return f'{fval:.{decimals}f}'.rstrip('0').rstrip('.')


@dataclass
class GearDataBase:
    gear_str: str


@dataclass
class GearKinematics(GearDataBase):
    speed: float  # 转速
    torque: float  # 扭矩
    power: float  # 功率
    nloops: float  # 应力循环

    @staticmethod
    def pack_table(kinematics: list['GearKinematics']):
        data = []
        for kin in kinematics:
            temp_dict = {
                r'转速 ($rpm$)': kin.speed,
                r'扭矩 ($N \cdot mm$)': kin.torque,
                r'功率 ($\text{kW}$)': kin.power,
                r'应力循环次数': f'{kin.nloops: .2e}',
            }
            for key, value in temp_dict.items():
                temp_dict[key] = try_parse_intstr(value, 2)
            data.append(temp_dict)
        df = pd.DataFrame(data)
        df.index = [f'齿轮 ${kin.gear_str}$' for kin in kinematics]
        return df


@dataclass
class GearFormFactor(GearDataBase):
    ysa: float
    yfa: float
    yf: float

    @staticmethod
    def pack_table(form_factors: list['GearFormFactor']):
        data = []
        for form in form_factors:
            temp_dict = {
                r'$Y_{sa}$': form.ysa,
                r'$Y_{fa}$': form.yfa,
                r'$Y_f$': form.yf
            }
            for key, value in temp_dict.items():
                temp_dict[key] = try_parse_intstr(value, 4)
            data.append(temp_dict)
        df = pd.DataFrame(data)
        df.index = [f'齿轮 ${form.gear_str}$' for form in form_factors]
        return df


@dataclass
class GearMaterial(GearDataBase):
    stress_lim: dict[str, float]
    safe_factor: dict[str, float]
    life_factor: dict[str, float]
    stress_safe: dict[str, float]
    z_elastic: float

    @staticmethod
    def pack_table(materials: list['GearMaterial']):
        data = []
        for mat in materials:
            temp_dict = {
                r'$\sigma_{H_{lim}}$': mat.stress_lim['contact'],
                r'$\sigma_{FE}$': mat.stress_lim['bending'],
                r'$S_{H_{min}}$': mat.safe_factor['contact'],
                r'$S_{F_{min}}$': mat.safe_factor['bending'],
                r'$Z_N$': mat.life_factor['contact'],
                r'$Y_N$': mat.life_factor['bending'],
                r'$\sigma_H$ (MPa)': mat.stress_safe['contact'],
                r'$\sigma_{F}$ (MPa)': mat.stress_safe['bending'],
                r'$\Z_E$': mat.z_elastic
            }
            for key, value in temp_dict.items():
                temp_dict[key] = try_parse_intstr(value, 2)
            data.append(temp_dict)
        df = pd.DataFrame(data)
        df.index = [f'齿轮 ${mat.gear_str}$' for mat in materials]
        return df


@dataclass
class GearSize(GearDataBase):
    module: float
    z: int
    d: float
    da: float
    df: float
    alpha: Angle
    beta: Angle
    ha: float
    hf: float
    h: float

    @staticmethod
    def pack_table(sizes: list['GearSize']):
        data = []
        for size in sizes:
            temp_dict = {
                r'模数 ($mm$)': size.module,
                r'齿数': size.z,
                r'分度圆直径 ($mm$)': size.d,
                r'齿顶圆直径 ($mm$)': size.da,
                r'齿根圆直径 ($mm$)': size.df,
                r'压力角 ($°$)': size.alpha,
                r'螺旋角 ($°$)': size.beta,
                r'齿顶高 ($mm$)': size.ha,
                r'齿根高 ($mm$)': size.hf,
                r'全齿高 ($mm$)': size.h
            }
            for key, value in temp_dict.items():
                if isinstance(value, float):
                    temp_dict[key] = try_parse_intstr(value, 2)
                else:
                    temp_dict[key] = str(value)
            data.append(temp_dict)
        df = pd.DataFrame(data)
        df.index = [f'齿轮 ${size.gear_str}$' for size in sizes]
        return df


@dataclass
class GearSolveResult(GearDataBase):
    d_min: float
    module: float
    d: float

    @staticmethod
    def pack_table(sizes: list['GearSolveResult']):
        data = []
        for size in sizes:
            temp_dict = {
                r'$d_{min}$': size.d_min,
                r'$m$': size.module,
                r'$d$': size.d,
            }
            for key, value in temp_dict.items():
                temp_dict[key] = try_parse_intstr(value, 2)
            data.append(temp_dict)
        df = pd.DataFrame(data)
        df.index = [f'齿轮 ${size.gear_str}$' for size in sizes]
        return df


def pack_table(datalist) -> pd.DataFrame:
    ele_sample = datalist[0]
    for cls in GearDataBase.__subclasses__():
        if isinstance(ele_sample, cls):
            return cls.pack_table(datalist)


class CalcType(Enum):
    ALL = 'all'
    KINEMATICS = 'kinematics'
    FORM_FACTOR = 'form_factor'
    MATERIAL = 'material'
    SOLVE_CONTACT = 's_c'


class GearDraft:
    M_SERIES = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6]

    @staticmethod
    def create_gears(hours, *teeth) -> list['GearDraft']:
        gears = []
        for i, z in enumerate(teeth):
            gear = GearDraft(str(i + 1), hours)
            gear.z = round(z)
            gears.append(gear)
        return gears

    @staticmethod
    def set_val(gears, name, *values):
        if name == 'contact':
            for gear, value in zip(gears, values):
                gear.stress_lim['contact'] = value
        elif name == 'bending':
            for gear, value in zip(gears, values):
                gear.stress_lim['bending'] = value
        else:
            for gear, value in zip(gears, values):
                setattr(gear, name, value)

    @staticmethod
    def batch_calc(gears: list['GearDraft'],
                   item=CalcType.ALL) -> list[GearDataBase]:
        for gear in gears:
            gear.process_features()
        result: list[GearDataBase] = []
        if item == CalcType.ALL:
            for gear in gears:
                result.append(gear.name)
            return result
        if item == CalcType.KINEMATICS:
            for gear in gears:
                result.append(gear.get_kinematics())
            return result
        if item == CalcType.FORM_FACTOR:
            for gear in gears:
                result.append(gear.get_form_factor())
            return result
        if item == CalcType.MATERIAL:
            for gear in gears:
                result.append(gear.get_material())
            return result
        if item == CalcType.SOLVE_CONTACT:
            d_min_list = []
            for gear in gears:
                d_min_list.append(gear.solve_by(0, 'contact'))
                gear.process_features()
            d_min_list.reverse()
            for gear in gears:
                gear.check_mesh(0, True)
                result.append(GearSolveResult(
                    gear.name, d_min_list[-1],
                    gear.module, gear.d
                ))
                d_min_list.pop()
            return result

    def __init__(self, idx: str, hours: float):
        self.name = idx  # 齿轮名称
        self.time = hours  # 使用时间
        self.z = None  # 齿数
        self.module = None  # 模数
        self.phid = None  # 宽径比
        self.form_factor = [None, None, None]  # 齿形系数
        self.d = None  # 分度圆直径
        self.b = None  # 齿宽
        self.da = None  # 齿顶圆直径
        self.df = None  # 齿根圆直径
        self.alpha = Angle(20)  # 压力角
        self.beta: Angle = None  # 螺旋角
        self.ha = None  # 齿顶高
        self.hf = None  # 齿根高
        self.h = None  # 全齿高
        self.torque = None  # 扭矩
        self.speed = None  # 转速
        self.power = None  # 功率
        self.axial_force = None  # 轴向力
        self.n_round = None  # 总转数
        self.eta = None  # 效率
        self.material: MaterialType = None  # 材料
        self.stress_lim = {
            'contact': None,  # 接触应力极限
            'bending': None,  # 弯曲应力极限
        }
        self.safe_factor = {
            'contact': None,  # 接触应力安全系数
            'bending': None,  # 弯曲应力安全系数
        }
        self.life_factor = {
            'contact': None,  # 接触应力寿命系数
            'bending': None,  # 弯曲应力寿命系数
        }
        self.stress_safe = {
            'contact': None,  # 接触应力许用值
            'bending': None,  # 弯曲应力许用值
        }
        self.z_elastic = None  # 弹性系数
        self.k = None  # 载荷系数

        self.mesh_with: list[GearDraft] = []  # 与其他齿轮啮合
        self.fixed_with: list[GearDraft] = []  # 与其他齿轮在一根轴上

    def process_features(self):
        '''
        计算齿轮的几何特征
        '''
        def notNone(*args):
            def check_not_none(value):
                # 如果值是字典，检查字典中的每个值
                if isinstance(value, dict):
                    return all(check_not_none(v) for v in value.values())
                # 对于非字典类型，直接检查是否为 None
                return value is not None
            # 使用 all() 函数和生成器表达式检查所有参数
            values = (check_not_none(arg) for arg in args)
            false_pos = [i for i, v in enumerate(values) if not v]
            if false_pos:
                print(f"找到 None: {false_pos}")
            return len(false_pos) == 0

        def ysa(z, beta):
            zv = z / beta.cos() ** 3
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

        def yfa(z, beta):
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

        if notNone(self.z, self.beta):
            ysa = ysa(self.z, self.beta)
            yfa = yfa(self.z, self.beta)
            self.form_factor = [ysa, ysa, yfa * yfa]

        if notNone(self.power, self.speed):
            self.torque = (self.power * 60e6) / (2 * np.pi * self.speed)
        elif notNone(self.torque, self.power):
            self.speed = (60e6 * self.power) / (2 * np.pi * self.torque)
        elif notNone(self.speed, self.torque):
            self.power = (self.torque * 2 * np.pi * self.speed) / 60e6
        else:
            print("无法计算齿轮的功率、扭矩和转速")

        if self.speed is not None:
            self.n_round = self.time * self.speed * 60
        else:
            print("无法计算齿轮的总转数")

        if notNone(self.n_round, self.stress_lim,
                   self.safe_factor, self.material):
            sigma, coeff = self.material.calc_contact_limit(
                self.stress_lim['contact'],
                self.safe_factor['contact'],
                self.n_round
            )
            self.stress_safe['contact'] = sigma
            self.life_factor['contact'] = coeff
            sigma, coeff = self.material.calc_bending_limit(
                self.stress_lim['bending'],
                self.safe_factor['bending'],
                self.n_round
            )
            self.stress_safe['bending'] = sigma
            self.life_factor['bending'] = coeff
        else:
            print("无法计算齿轮的接触应力和弯曲应力")

        if notNone(self.z, self.module):
            self.d = self.module * self.z
            self.b = self.d * self.phid
            self.ha = self.module
            self.hf = 1.25 * self.module
            self.h = self.ha + self.hf
            self.da = self.d + 2 * self.ha
            self.df = self.d - 2 * self.hf
        else:
            print("无法计算齿轮的几何特征")

    def __matmul__(self, gear2: 'GearDraft') -> "GearDraft":
        '''
        将两个齿轮啮合在一起，计算第二个齿轮上的运动学参数
        '''
        self.process_features()
        gear2.process_features()
        i = self.z / gear2.z
        gear2.torque = self.torque / i
        gear2.speed = self.speed * i
        gear2.power = self.power * self.eta
        if self.axial_force is not None:
            gear2.axial_force = -self.axial_force
        self.mesh_with.append(gear2)
        gear2.mesh_with.append(self)
        return gear2

    def __sub__(self, gear2: 'GearDraft') -> "GearDraft":
        '''
        将两个齿轮放在一个轴上，计算第二个齿轮上的运动学参数
        '''
        gear2.torque = self.torque
        gear2.speed = self.speed
        gear2.power = self.power
        self.fixed_with.append(gear2)
        gear2.fixed_with.append(self)
        return gear2

    def get_kinematics(self):
        return GearKinematics(
            self.name, self.speed,
            self.torque, self.power,
            self.n_round
        )

    def get_material(self):
        return GearMaterial(
            self.name,
            self.stress_lim,
            self.safe_factor,
            self.life_factor,
            self.stress_safe,
            self.z_elastic
        )

    def gear_ratio(self, gear2: 'GearDraft'):
        return gear2.z / self.z

    def get_form_factor(self) -> float:
        return GearFormFactor(self.name, *self.form_factor)

    def _calc_min_d(self):
        zh = 2.5  # 计算区域系数。例如，普通圆柱齿轮通常为 2.5。
        if float(self.beta) >= 7.:
            # 认为只有 β >= 7° 才算斜齿轮
            alpha_t = np.arctan(self.alpha.tan() / self.beta.cos())
            beta_b = np.arctan(self.beta.tan() * np.cos(alpha_t))
            alpha_t_1 = alpha_t  # 没有变位
            zh = np.sqrt(2 * np.cos(beta_b) * np.cos(alpha_t_1) / (
                np.cos(alpha_t) ** 2 * np.sin(alpha_t_1)
            ))  # 计算区域系数
        # print(k_, t_, u_, beta, zh, ZE, sigh)
        z1, z2 = self.z, self.mesh_with[0].z
        u = max(z1, z2) / min(z1, z2)
        return (
            2 * self.k * self.torque / self.phid *
            (u + 1) / u * self.beta.cos() *
            (zh * self.z_elastic / self.stress_safe['contact']) ** 2
        ) ** (1 / 3)

    def _calc_min_m(self, explicit=False):
        if not explicit:
            zv = self.z / self.beta.cos()  # 修正齿数
            return (
                2 * self.k * self.torque / self.phid / zv**2 *
                self.form_factor[2] / self.stress_safe['bending']
            ) ** (1 / 3)
        # 计算 Y_{\beta}
        beta = min(self.beta, Angle(30))
        eps_beta = self.b * self.beta.sin() / np.pi / self.module
        eps_beta = min(eps_beta, 1)
        y_beta_min = 1 - 0.25 * eps_beta
        y_beta = 1 - eps_beta * float(beta) / 120
        y_beta = max(y_beta_min, y_beta, 0.75)
        return (
            2 * self.k * y_beta * self.torque / self.phid / zv**2 *
            self.form_factor[2] / self.stress_safe['bending']
        ) ** (1 / 3)

    def solve_by(self, mesh_idx=0, stype='contact'):
        if stype == 'contact':
            d_min = self._calc_min_d()
            m = d_min / self.z * self.beta.cos()
            m_std = min(GearDraft.M_SERIES,
                        key=lambda x: x - m if x > m else np.inf)
            self.module = m_std
            return d_min
        if stype == 'bending':
            raise NotImplementedError("弯曲应力计算尚未实现")
            # m_min = self._calc_min_m()
        raise ValueError("错误的计算类型，只能为 `contact` 或 `bending` 。")

    def check_mesh(self, mesh_idx=0, auto_refresh=True):
        module2 = self.mesh_with[mesh_idx].module
        if module2 != self.module:
            module = max(self.module, module2)
            self.module = module
            self.mesh_with[mesh_idx].module = module
            if auto_refresh:
                self.process_features()
                self.mesh_with[mesh_idx].process_features()

    def speed_error(self, target):
        err = (self.speed - target) / target
        if err > 0.005:
            return f'转速误差：{err: .2f}%'
        return f'转速误差：{err: .4f}%'
