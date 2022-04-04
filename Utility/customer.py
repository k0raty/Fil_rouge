class Customer:
    CUSTOMER_CODE: str = ''
    LATITUDE: float = 0
    LONGITUDE: float = 0
    CUSTOMER_TIME_WINDOW_FROM_MIN: int = 8
    CUSTOMER_TIME_WINDOW_TO_MIN: int = 15
    TOTAL_WEIGHT_KG: float = 1
    TOTAL_VOLUME_M3: float = 1
    CUSTOMER_DELIVERY_SERVICE_TIME_MIN: float = 1
    POS: tuple = (0, 0)

    def __init__(self, row, index):
        self.INDEX = index
        self.CUSTOMER_CODE = row[0]
        self.LATITUDE = float(row[1])
        self.LONGITUDE = float(row[2])
        self.CUSTOMER_TIME_WINDOW_FROM_MIN = int(row[3])
        self.CUSTOMER_TIME_WINDOW_TO_MIN = int(row[4])
        self.CUSTOMER_DELIVERY_SERVICE_TIME_MIN = float(row[5])
        self.TOTAL_WEIGHT_KG = float(row[6])
        self.TOTAL_VOLUME_M3 = float(row[7])
        self.POS = str(row[8])
