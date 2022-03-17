import math


#ALGORITHME DE CALCUL DE DISTANCES CLIENTS<-->CLIENTS
def distance(lat1,lon1,lat2,lon2,alt1=100,alt2=100):
    rEquat=6378137
    rPole=6356752
    rLat=rEquat-((rEquat-rPole)*abs(lat1/90))+alt1
    distParallele=abs(rLat*math.cos(((lat1+lat2)/2)*math.pi/180)*((lon2-lon1)*math.pi/180))
    distMeridien=abs(rLat*(lat2-lat1)*math.pi/180)
    distVerticale=abs(alt2-alt1)
    distTotale=math.sqrt((distParallele*distParallele)+(distMeridien*distMeridien)+(distVerticale*distVerticale))
    return(distTotale)

def client_le_plus_proche(act_line,df,treated_lines): #actual_line est un numéro entre 0 et 1159
    act_lat=act_line['CUSTOMER_LATITUDE']
    act_lon=act_line['CUSTOMER_LONGITUDE']
    act_customer_code=act_line['CUSTOMER_CODE']
    first_client_found=False
    j=0
    dist_min=99999999999999999999999999
    """
    #INITIALISATION
    while not first_client_found:   #ON CHERCHE LE PREMIER CLIENT QUI N'EST PAS TABOU
        j_line=df.iloc[j]
        j_customer_code=j_line['CUSTOMER_CODE']
        
        if j not in treated_lines and act_customer_code!=j_customer_code:
            
            j_lat=j_line['CUSTOMER_LATITUDE']
            j_lon=j_line['CUSTOMER_LONGITUDE']
            
            dist_min=distance(act_lat,act_lon,j_lat,j_lon)
            customer_code_min=j
            first_client_found=True
            
        else:
            j+=1
    """
    #ON CHERCHE LE CLIENT LE PLUS PROCHE
    for i in range(0,df.shape[0]):
        i_line=df.iloc[i]
        i_customer_code=i_line['CUSTOMER_CODE']
       
        if i not in treated_lines and act_customer_code!=i_customer_code:
           
            i_lat=i_line['CUSTOMER_LATITUDE']
            i_lon=i_line['CUSTOMER_LONGITUDE']
               
            dist=distance(act_lat,act_lon,i_lat,i_lon)
            if dist<=dist_min:
                dist_min=dist
                customer_code_min=i_customer_code
            
            
    return(customer_code_min)