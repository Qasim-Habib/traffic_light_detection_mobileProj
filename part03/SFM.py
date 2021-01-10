import numpy as np


WAIGHT_X = 0.55
WAIGHT_Y = 0.45

def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container

def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ

def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec

def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    return np.array( [ ( (p - pp)  / focal ) for p in pts ] )
       
def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    return np.array( [ ( ( p[:2] * focal ) + pp ) for p in pts ] )

def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    tz = [ EM[0,3], EM[1,3], EM[2,3] ]
    foe = ( tz[0] / tz[2] , tz[1] / tz[2] ) if abs( tz[2] ) > 10e-6 else []
    return R, foe, tz[2]
    
def rotate(pts, R):
    # pts are x,y of traffic lights
    # rotate the points - pts using R
    pr = np.array( [ R.dot( [p[0], p[1], 1] ) for p in pts  ] )
    pr = np.array( [ [ pri[0] / pri[2] , pri[1] / pri[2] ] for pri in pr ] )
    return pr

def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    m = ( foe[1] - p[1] ) / ( foe[0] - p[0] )
    n = ( p[1] * foe[0] - foe[1] * p[0] ) / ( foe[0] - p[0] )
    # run over all norm_pts_rot and find the one closest to the epipolar line
    min_p_c = float('inf') 
    min_p = None
    min_ind = -1
    for i, p_r in enumerate(norm_pts_rot):
        dist_ = abs( ( m * p_r[0] + n - p_r[1] ) / ( ( m ** 2 + 1 ) ** 0.5 ) )
        if dist_ < min_p_c:
            min_p_c = dist_
            min_p = p_r
            min_ind = i
    # return the closest point and its index
    return min_ind, min_p

def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    zx = ( tZ * ( foe[0] - p_rot[0] ) ) / ( p_curr[0] - p_rot[0] ) 
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    zy = ( tZ * ( foe[1] - p_rot[1] ) ) / ( p_curr[1] - p_rot[1] )
    
    # z_x_w = np.sqrt( abs( p_curr[0] - p_rot[0] ) )
    # z_y_w = np.sqrt( abs( p_curr[1] - p_rot[1] ) )
    z_x_w = abs( p_curr[0] - p_rot[0] )
    z_y_w = abs( p_curr[1] - p_rot[1] )
    # sum_w = WAIGHT_X * z_x_w + WAIGHT_Y * z_y_w
    sum_w = z_x_w + z_y_w 
    if ( z_x_w + z_y_w ) == 0:
        return 0
    z_x_w /= sum_w
    z_y_w /= sum_w
    # combine the two estimations and return estimated Z
    return z_x_w * zx + z_y_w * zy
