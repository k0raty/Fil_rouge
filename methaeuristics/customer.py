class Customer:
    KEYS = [
        'CUSTOMER_ID',
        'ROUTE_ID',
        'CUSTOMER_NUMBER',
        'CUSTOMER_CODE',
        'CUSTOMER_LATITUDE',
        'CUSTOMER_LONGITUDE',
        'CUSTOMER_TIME_WINDOW_FROM_MIN',
        'CUSTOMER_TIME_WINDOW_TO_MIN',
        'NUMBER_OF_ARTICLES',
        'TOTAL_WEIGHT_KG',
        'TOTAL_VOLUME_M3',
        'CUSTOMER_DELIVERY_SERVICE_TIME_MIN',
    ]

    def __init__(self, row):
        for index in range(len(self.KEYS)):
            key = self.KEYS[index]
            value = row[index]
            setattr(self, key, value)
