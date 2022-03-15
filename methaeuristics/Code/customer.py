class Customer:
    CUSTOMER_ID: str = ''
    ROUTE_ID: str = ''
    CUSTOMER_NUMBER: str = ''
    CUSTOMER_CODE: str = ''
    CUSTOMER_LATITUDE: float = 0
    CUSTOMER_LONGITUDE: float = 0
    CUSTOMER_TIME_WINDOW_FROM_MIN: int = 8
    CUSTOMER_TIME_WINDOW_TO_MIN: int = 15
    NUMBER_OF_ARTICLES: int = 1
    TOTAL_WEIGHT_KG: float = 1
    TOTAL_VOLUME_M3: float = 1
    CUSTOMER_DELIVERY_SERVICE_TIME_MIN: float = 1

    def __init__(self, row):
        self.CUSTOMER_ID = row[0]
        self.ROUTE_ID = row[1]
        self.CUSTOMER_NUMBER = row[2]
        self.CUSTOMER_CODE = row[3]
        self.CUSTOMER_LATITUDE = float(row[4])
        self.CUSTOMER_LONGITUDE = float(row[5])
        self.CUSTOMER_TIME_WINDOW_FROM_MIN = int(row[6])
        self.CUSTOMER_TIME_WINDOW_TO_MIN = int(row[7])
        self.NUMBER_OF_ARTICLES = int(row[8])
        self.TOTAL_WEIGHT_KG = float(row[9])
        self.TOTAL_VOLUME_M3 = float(row[10])
        self.CUSTOMER_DELIVERY_SERVICE_TIME_MIN = float(row[11])
