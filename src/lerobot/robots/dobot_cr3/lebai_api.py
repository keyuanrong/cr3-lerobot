import time

try:
    from pymodbus.client.sync import ModbusSerialClient as ModbusClient
except ModuleNotFoundError:
    from pymodbus.client import ModbusSerialClient as ModbusClient

from pymodbus.exceptions import ModbusIOException


class lebai:

    def __init__(self):
        """
        初始化API
        """
        self.slave_address = 1
        self.verbose = False
        # 创建客户端实例
        self.port = '/dev/ttyACM0'
        self.baudrate = 115200
        self.timeout = 0.25
        self.retries = 0
        self.client = self._make_client(port=self.port, baudrate=self.baudrate, timeout=self.timeout, retries=self.retries)

    def configure(
        self,
        port: str | None = None,
        baudrate: int | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ):
        self.port = port or self.port
        self.baudrate = baudrate or self.baudrate
        self.timeout = timeout or self.timeout
        self.retries = self.retries if retries is None else retries
        self.client = self._make_client(port=self.port, baudrate=self.baudrate, timeout=self.timeout, retries=self.retries)

    def _make_client(self, port: str, baudrate: int, timeout: float, retries: int):
        try:
            return ModbusClient(method='rtu', port=port, baudrate=baudrate, timeout=timeout, retries=retries)
        except TypeError:
            try:
                return ModbusClient(port=port, baudrate=baudrate, timeout=timeout, retries=retries)
            except TypeError:
                return ModbusClient(port=port, baudrate=baudrate, timeout=timeout)

    def _check_response(self, response):
        if hasattr(response, 'isError') and response.isError():
            raise RuntimeError(f"Modbus通信错误: {str(response)}")
        if isinstance(response, ModbusIOException):
            raise RuntimeError(f"Modbus通信异常: {str(response)}")
        return response

    def _write_register(self, address: int, value: int):
        last_error = None
        for slave_kw in ("unit", "slave", "device_id"):
            try:
                return self._check_response(self.client.write_register(address, value, **{slave_kw: self.slave_address}))
            except TypeError as exc:
                last_error = exc
        raise last_error

    def _read_holding_registers(self, address: int, count: int):
        attempts = (
            ((address, count), {"unit": self.slave_address}),
            ((address,), {"count": count, "slave": self.slave_address}),
            ((address,), {"count": count, "device_id": self.slave_address}),
        )
        last_error = None
        for args, kwargs in attempts:
            try:
                return self._check_response(self.client.read_holding_registers(*args, **kwargs))
            except TypeError as exc:
                last_error = exc
        raise last_error

    def connect(self):
        """连接到夹爪"""
        if self.verbose:
            print(f"正在连接到夹爪 at {self.port}...")
        if not self.client.connect():
            raise ConnectionError("无法连接到夹爪")
        if self.verbose:
            print("夹爪连接成功！")

    def is_connected(self) -> bool:
        is_socket_open = getattr(self.client, "is_socket_open", None)
        if callable(is_socket_open):
            return bool(is_socket_open())

        connected = getattr(self.client, "connected", False)
        return bool(connected() if callable(connected) else connected)

    def disconnect(self):
        """断开与夹爪的连接"""
        if self.client:
            self.client.close()
            if self.verbose:
                print("夹爪连接已断开")

    # --- 核心控制方法 ---
    def set_width(self, width: int, wait: bool = True):
        """设置夹爪开合度 (0-100)"""
        # 寄存器地址 40000
        width = max(0, min(100, width))  # 确保值在0-100之间
        self._write_register(40000, width)
        if wait: self.wait_finish()

    def set_force(self, force: int):
        """设置夹爪力度 (0-100)"""
        # 寄存器地址 40001
        force = max(0, min(100, force))
        self._write_register(40001, force)

    """
    这里注意，经过测量，想要获得精确的值，必须要经过5秒，并且是第二次获取的值才是对的有±1的误差
    """

    def get_position(self) -> int:
        """获取夹爪当前的位置 (0-100)"""
        # 寄存器地址 40005
        response = self._read_holding_registers(40005, 1)
        return response.registers[0]

    def get_torque(self) -> int:
        """获取夹爪当前的力矩"""
        # 寄存器地址 40006
        response = self._read_holding_registers(40006, 1)
        return response.registers[0]

    def wait_finish(self):
        """等待指令完成"""
        # 寄存器地址 40007
        response = self._read_holding_registers(40007, 1)

        return response.registers[0]

    # 处理夹爪开合问题（角度不是0）
    def initialize(self, wait: bool = True):
        """执行找行程初始化"""
        # 寄存器地址 40008, 写入 1
        self._write_register(40008, 1)
        if wait: self.wait_finish()

    def open(self, wait: bool = True):
        """完全打开夹爪"""
        #self.set_width(100, wait)   #原本代码
        self.set_width(100, wait = False)   #新代码

    def close(self, wait: bool = True):
        """完全闭合夹爪"""
        #self.set_width(0, wait)   #原本代码
        self.set_width(0, wait = False)   #新代码

    # 实现抓取
    def catch(self, flag: int):
        """当力矩达到某个阈值，停止抓取"""
        # 设置夹爪力度
        self.set_force(50)

        """
            initiate_torque = 5 : 假设初始力矩 
            current_torque : 变化力矩 先赋值一个初始值 
            differ : 变化的差值 当前的 - 初始的
        """
        initiate_torque = 5
        current_torque = 0
        differ = current_torque - initiate_torque

        """
            现象 : 
                夹爪先开到最大，抓取到东西之后，停止抓取动作
                设置了小物品夹取的动作


            1 、 i : 设置一个最大角度,当前为待抓取的角度

            2 、 做一个循环对差值进行判断，经测试，夹爪夹住物品时，力矩会变大，20作为一个值进行判断
                当差值大于20时，夹爪停下

            3 、 在循环的过程中，不断地获取夹爪的力矩，从而进行差值的计算

            4 、 这里的 time.sleep(0.5) 是为了避免获取力矩时信息的堵塞，经测试，堵塞至少需要 5秒 才能顺畅获取当前有效力矩

            5 、 flag = 0 夹取范围支持0~10 否则就不是
        """
        i = 100
        while abs(differ) < 20:
            self.set_width(i)
            print(f"当前执行的角度是 -- {i}")
            i -= 10
            # i 取值区间 [0,100]
            if i < 0:
                print("-----------------------------夹爪没有抓到东西，当前已经合上了>_<-------------------------------")
                break
            time.sleep(0.5)
            current_torque = self.get_torque()
            # print(f"当前力矩是 {current_torque}%")
            differ = current_torque - initiate_torque
            # print(f"当前差值differ是 {abs(differ)}%")
            """
                对抓取角度小于10做的处理
            """
            if (i == 10) & (flag == 0):
                while abs(differ) < 20:
                    self.set_width(i)
                    print(f"当前执行的角度是 -- {i}")
                    i -= 1
                    # i 取值区间 [0,100]
                    if i < 1:
                        print(
                            "-----------------------------夹爪没有抓到东西，当前已经合上了>_<-------------------------------")
                        break
                    time.sleep(0.2)
                    current_torque = self.get_torque()
                    # print(f"当前力矩是 {current_torque}%")
                    differ = current_torque - initiate_torque
        if differ >= 20:
            print("-----------------------------夹爪已经抓住东西了>_<-------------------------------")
            """
                对当前的角度进行判断，如果在100~10之间的话可以加10，因为这是正在执行的角度
                如果是0~9的话就加1
            """
            if i >= 10:
                print(f"当前执行角度是 {i + 10}%")
            elif i < 10:
                print(f"当前执行角度是 {i + 1}%")
            print(f"当前力矩是 {current_torque}%")
            print(f"当前差值differ是 {abs(differ)}%")
