from math import cos,sin,asin,acos,atan


# ecef to local coordinate transfromation matrix
# input:
#   pos : geodetic position {lat,lon} (rad)
# return:
#   E   : ecef to local coord transformation matrix (3x3)
def xyz_to_enu(pos):
    E = [0.0 for i in range(9)]
    E[0] = -sin(pos[1])
    E[3] = cos(pos[1])
    E[1] = -sin(pos[0])*cos(pos[1])
    E[4] = -sin(pos[0])*sin(pos[1])
    E[7] = cos(pos[0])
    E[2] = cos(pos[0])*cos(pos[1])
    E[5] = cos(pos[0])*sin(pos[1])
    E[8] = sin(pos[0])
    return E


def ecef_to_enu(pos):
    return 1


#
def pos_to_ecef(pos):
    return 1