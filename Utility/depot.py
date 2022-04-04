class Depot:
    ROUTE_ID: str = ''
    DEPOT_CODE: str = ''
    LATITUDE: float = 0
    LONGITUDE: float = 0
    DEPOT_AVAILABLE_TIME_FROM_MIN: int = 5
    DEPOT_AVAILABLE_TIME_TO_MIN: int = 15

    def __init__(self, row):
        self.INDEX = 0
        self.ROUTE_ID = row[0]
        self.DEPOT_CODE = row[1]
        self.LATITUDE = float(row[2])
        self.LONGITUDE = float(row[3])
        self.DEPOT_AVAILABLE_TIME_FROM_MIN = int(row[4])
        self.DEPOT_AVAILABLE_TIME_TO_MIN = int(row[5])
