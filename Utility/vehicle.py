class Vehicle:
    KEYS = [
        'VEHICLE_NUMBER',
        'VEHICLE_CODE',
        'VEHICLE_TOTAL_WEIGHT_KG',
        'VEHICLE_TOTAL_VOLUME_M3',
        'VEHICLE_VARIABLE_COST_KM',
        'VEHICLE_AVAILABLE_TIME_FROM_MIN',
        'VEHICLE_AVAILABLE_TIME_TO_MIN',
    ]

    def __init__(self, row):
        self.VEHICLE_NUMBER = row[0]
        self.VEHICLE_CODE = row[1]
        self.VEHICLE_TOTAL_WEIGHT_KG = row[2]
        self.VEHICLE_TOTAL_VOLUME_M3 = row[3]
        self.VEHICLE_VARIABLE_COST_KM = row[4]
        self.VEHICLE_AVAILABLE_TIME_FROM_MIN = row[5]
        self.VEHICLE_AVAILABLE_TIME_TO_MIN = row[6]
