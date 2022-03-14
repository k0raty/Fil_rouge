class Depot:
    KEYS = [
        'ROUTE_ID',
        'DEPOT_NUMBER',
        'DEPOT_CODE',
        'DEPOT_LATITUDE',
        'DEPOT_LONGITUDE',
        'DEPOT_AVAILABLE_TIME_FROM_MIN',
        'DEPOT_AVAILABLE_TIME_TO_MIN',
    ]

    def __init__(self, row):
        for index in range(len(self.KEYS)):
            key = self.KEYS[index]
            value = row[index]
            setattr(self, key, value)
