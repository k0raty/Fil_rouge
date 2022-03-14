class Vehicle:
    KEYS = [
        'ROUTE_ID, VEHICLE_NUMBER',
        'VEHICLE_CODE',
        'VEHICLE_TOTAL_WEIGHT_KG',
        'VEHICLE_TOTAL_VOLUME_M3',
        'VEHICLE_FIXED_COST_KM',
        'VEHICLE_VARIABLE_COST_KM',
        'VEHICLE_AVAILABLE_TIME_FROM_MIN',
        'VEHICLE_AVAILABLE_TIME_TO_MIN',
        'RESULT_VEHICLE_TOTAL_DRIVING_TIME_MIN',
        'RESULT_VEHICLE_TOTAL_DELIVERY_TIME_MIN',
        'RESULT_VEHICLE_TOTAL_ACTIVE_TIME_MIN',
        'RESULT_VEHICLE_DRIVING_WEIGHT_KG',
        'RESULT_VEHICLE_DRIVING_VOLUME_M3',
        'RESULT_VEHICLE_FINAL_COST_KM',
    ]

    def __init__(self, row):
        for index in range(len(self.KEYS)):
            key = self.KEYS[index]
            value = row[index]
            setattr(self, key, value)
