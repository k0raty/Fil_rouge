import math


# ALGORITHME DE CALCUL DE DISTANCES CLIENTS<-->CLIENTS
def distance(lat1, lon1, lat2, lon2, alt1=100, alt2=100):
    rEquat = 6378137
    rPole = 6356752
    rLat = rEquat - ((rEquat - rPole) * abs(lat1 / 90)) + alt1
    distParallele = abs(rLat * math.cos(((lat1 + lat2) / 2) * math.pi / 180) * ((lon2 - lon1) * math.pi / 180))
    distMeridien = abs(rLat * (lat2 - lat1) * math.pi / 180)
    distVerticale = abs(alt2 - alt1)
    distTotale = math.sqrt(
        (distParallele * distParallele) + (distMeridien * distMeridien) + (distVerticale * distVerticale))
    return distTotale
