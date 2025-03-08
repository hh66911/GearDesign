from dataclasses import dataclass
from enum import Enum
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
        return np.deg2rad(self._degrees)

    def sin(self):
        # 计算正弦值
        return np.sin(self.to_radians())

    def cos(self):
        # 计算余弦值
        return np.cos(self.to_radians())

    def tan(self):
        return np.tan(self.to_radians())


def try_parse_intstr(fval: float, decimals=2):
    if fval.is_integer():
        return str(int(fval))
    return f'{fval:.{decimals}f}'

@dataclass
class GearDataBase:
    gear_str: str


@dataclass
class GearKinematics(GearDataBase):
    speed: float  # 转速
    torque: float  # 扭矩
    power: float  # 功率

    @staticmethod
    def pack_table(kinematics: list['GearKinematics']):
        data = []
        for kin in kinematics:
            temp_dict = {
                r'转速 ($rpm$)': kin.speed,
                r'扭矩 ($N \cdot mm$)': kin.torque,
                r'功率 ($\text{kW}$)': kin.power
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
    stress_safe: dict[str, float]
    z_elastic: float

    @staticmethod
    def pack_table(materials: list['GearMaterial']):
        data = []
        for mat in materials:
            temp_dict = {
                r'$\sigma_{H_{lim}}$ ($MPa$)': mat.stress_lim['contact'],
                r'$\sigma_{FE}$ ($MPa$)': mat.stress_lim['bending'],
                r'$S_{H_{min}}$': mat.safe_factor['contact'],
                r'$S_{F_{min}}$': mat.safe_factor['bending'],
                r'$\sigma_H$ ($MPa$)': mat.stress_safe['contact'],
                r'$\sigma_{F}$ ($MPa$)': mat.stress_safe['bending'],
                r'$\Z_E$ ($MPa$)': mat.z_elastic
            }
            for key, value in temp_dict.items():
                temp_dict[key] = try_parse_intstr(value, 2)
            data.append(temp_dict)
        df = pd.DataFrame(data)
        df.index = [f'材料 ${mat.gear_str}$' for mat in materials]
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


def pack_table(datalist):
    ele_sample = datalist[0]
    for cls in GearDataBase.__subclasses__():
        if isinstance(ele_sample, cls):
            return cls.pack_table(datalist)


class CalcType(Enum):
    ALL = 'all'
    KINEMATICS = 'kinematics'
    FORM_FACTOR = 'form_factor'
    MATERIAL = 'material'


class GearDraft:
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
                   item=CalcType.ALL):
        match item:
            case CalcType.ALL:
                for gear in gears:
                    gear.process_features()
                return gears
            case CalcType.KINEMATICS:
                result: list[GearDataBase] = []
                for gear in gears:
                    gear.process_features()
                    result.append(gear.get_kinematics())
                return result
            case CalcType.FORM_FACTOR:
                result: list[GearDataBase] = []
                for gear in gears:
                    gear.process_features()
                    result.append(gear.form_factor())
                return result
            case CalcType.MATERIAL:
                result: list[GearDataBase] = []
                for gear in gears:
                    gear.process_features()
                    result.append(gear.get_material())
                return result

    def __init__(self, idx: str, hours: float):
        self.name = idx  # 齿轮名称
        self.time = hours  # 使用时间
        self.z = None  # 齿数
        self.module = None  # 模数
        self.d = None  # 分度圆直径
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
        self.material = None  # 材料
        self.stress_lim = {
            'contact': None,  # 接触应力极限
            'bending': None,  # 弯曲应力极限
        }
        self.safe_factor = {
            'contact': None,  # 接触应力安全系数
            'bending': None,  # 弯曲应力安全系数
        }
        self.stress_safe = {
            'contact': None,  # 接触应力许用值
            'bending': None,  # 弯曲应力许用值
        }
        self.z_elastic = None  # 弹性系数

    def process_features(self):
        '''
        计算齿轮的几何特征
        '''
        if self.power is not None and self.speed is not None:
            self.torque = (self.power * 60e6) / (2 * np.pi * self.speed)
        elif self.torque is not None and self.power is not None:
            self.speed = (60e6 * self.power) / (2 * np.pi * self.torque)
        elif self.torque is not None and self.speed is not None:
            self.power = (self.torque * 2 * np.pi * self.speed) / 60e6
        else:
            print("无法计算齿轮的功率、扭矩和转速")

        if self.speed is not None:
            self.n_round = self.time * self.speed
        else:
            print("无法计算齿轮的总转数")
            
            
        if self.module is not None and self.z is not None:
            self.d = self.module * self.z
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
        return gear2

    def __sub__(self, gear2: 'GearDraft') -> "GearDraft":
        '''
        将两个齿轮放在一个轴上，计算第二个齿轮上的运动学参数
        '''
        gear2.torque = self.torque
        gear2.speed = self.speed
        gear2.power = self.power
        return gear2

    def get_kinematics(self):
        return GearKinematics(
            self.name, self.speed,
            self.torque, self.power
        )
        
    def get_material(self):
        return GearMaterial(
            self.name, self.stress_lim,
            self.stress_safe, self.z_elastic
        )

    def gear_ratio(self, gear2: 'GearDraft'):
        return gear2.z / self.z

    def form_factor(self) -> float:
        '''
        计算齿形系数
        '''
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

        yfa = yfa(self.z, self.beta)
        ysa = ysa(self.z, self.beta)
        return GearFormFactor(
            self.name, ysa, yfa, ysa * yfa)

    def speed_error(self, target):
        err = (self.speed - target) / target
        if err > 0.005:
            return f'转速误差：{err: .2f}%'
        return f'转速误差：{err: .4f}%'
