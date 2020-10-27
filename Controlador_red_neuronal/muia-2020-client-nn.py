#!/usr/bin/python3

# --------------------------------------------------------------------------

print('### Script:', __file__)

# --------------------------------------------------------------------------

import math
import sys
import time

#import cv2 as cv
import numpy as np
import sim

import pickle

# --------------------------------------------------------------------------

def getRobotHandles(clientID):
    # Motor handles
    _,lmh = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_leftMotor',
                                     sim.simx_opmode_blocking)
    _,rmh = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_rightMotor',
                                     sim.simx_opmode_blocking)

    # Sonar handles
    str = 'Pioneer_p3dx_ultrasonicSensor%d'
    sonar = [0] * 16
    for i in range(16):
        _,h = sim.simxGetObjectHandle(clientID, str % (i+1),
                                       sim.simx_opmode_blocking)
        sonar[i] = h
        sim.simxReadProximitySensor(clientID, h, sim.simx_opmode_streaming)

    # Camera handles
    _,cam = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_camera',
                                        sim.simx_opmode_oneshot_wait)
    sim.simxGetVisionSensorImage(clientID, cam, 0, sim.simx_opmode_streaming)
    sim.simxReadVisionSensor(clientID, cam, sim.simx_opmode_streaming)

    return [lmh, rmh], sonar, cam

# --------------------------------------------------------------------------

def setSpeed(clientID, hRobot, lspeed, rspeed):
    sim.simxSetJointTargetVelocity(clientID, hRobot[0][0], lspeed,
                                    sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(clientID, hRobot[0][1], rspeed,
                                    sim.simx_opmode_oneshot)

# --------------------------------------------------------------------------

def getSonar(clientID, hRobot):
    r = [1.0] * 16
    for i in range(16):
        handle = hRobot[1][i]
        e,s,p,_,_ = sim.simxReadProximitySensor(clientID, handle,
                                                 sim.simx_opmode_buffer)
        if e == sim.simx_return_ok and s:
            r[i] = math.sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2])

    return r

# --------------------------------------------------------------------------

# def getImage(clientID, hRobot):
#     img = []
#     err,r,i = sim.simxGetVisionSensorImage(clientID, hRobot[2], 0,
#                                             sim.simx_opmode_buffer)

#     if err == sim.simx_return_ok:
#         img = np.array(i, dtype=np.uint8)
#         img.resize([r[1],r[0],3])
#         img = np.flipud(img)
#         img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

#     return err, img

# --------------------------------------------------------------------------

def getImageBlob(clientID, hRobot):
    rc,ds,pk = sim.simxReadVisionSensor(clientID, hRobot[2],
                                         sim.simx_opmode_buffer)
    blobs = 0
    coord = []
    if rc == sim.simx_return_ok and pk[1][0]:
        blobs = int(pk[1][0])
        offset = int(pk[1][1])
        for i in range(blobs):
            coord.append(pk[1][4+offset*i])
            coord.append(pk[1][5+offset*i])

    return blobs, coord

def is_crashing(sonar):
    print(sonar)
    return min(sonar) < 0.25

# --------------------------------------------------------------------------
# if (sonar[3] < 0.1) or (sonar[4] < 0.1) original
def avoid(sonar):

    lateral_izq = 0.0
    lateral_der = 0.0

    lateral_izq = sonar[2]+sonar[1]+sonar[0]
    lateral_der = sonar[5]+sonar[6]+sonar[7]
    

    if (sonar[3] < 0.3) or (sonar[4] < 0.3) or (sonar[5] < 0.18) or (sonar[2] < 0.18):
        if lateral_der > lateral_izq:
             lspeed, rspeed = +0.9, -0.2 # Gira derecha
             print("derecha")
        else: 
            lspeed, rspeed = -0.2, +0.9 # Gira izquierda
            print("izquierda")
    elif (sonar[14] < 0.18) or (sonar[12] < 0.3) or (sonar[11] < 0.3) or (sonar[9] < 0.18):
        lspeed, rspeed = +2.0, +2.0 # Recto
    else:
        lspeed, rspeed = +2.0, +2.0 # Recto

    return lspeed, rspeed

# --------------------------------------------------------------------------

def follow(coord, last_coord):

    if coord[0] <= 0.4: # Pelota hacia la izquierda
        lspeed, rspeed = +0.5, +1.3 # Gira izquierda
    elif coord[0] >= 0.6: #Pelota hacia la derecha
        lspeed, rspeed = +1.3, +0.5 # Gira derecha
    else:
        lspeed, rspeed = 1.5, 1.5

    return lspeed, rspeed

def search(last_coord):
    if last_coord[0] <= 0.5: # Pelota hacia la izquierda
        lspeed, rspeed = -0.5, +0.5 # Gira izquierda
    else: #Pelota hacia la derecha
        lspeed, rspeed = +0.5, -0.5 # Gira derecha

    return lspeed, rspeed
# --------------------------------------------------------------------------
def main():
    print('### Program started')

    print('### Number of arguments:', len(sys.argv), 'arguments.')
    print('### Argument List:', str(sys.argv))

    sim.simxFinish(-1) # just in case, close all opened connections

    port = int(sys.argv[1])
    clientID = sim.simxStart('127.0.0.1', port, True, True, 2000, 5)

    if clientID == -1:
        print('### Failed connecting to remote API server')

    else:
        print('### Connected to remote API server')
        hRobot = getRobotHandles(clientID)
        last_coord = [1,1]

        clf = pickle.load(open('../clf.sav', 'rb'))

        while sim.simxGetConnectionId(clientID) != -1:
            # Perception
            sonar = getSonar(clientID, hRobot)
            # print('### s', sonar)

            blobs, coord = getImageBlob(clientID, hRobot)
            # print('###  ', blobs, coord)
            if coord == []:
                coord = [0, 0]

            if blobs == 1:
                last_coord = coord

            if is_crashing(sonar):
                # lspeed, rspeed = avoid(sonar)
                # print('Me voy a chocar')

                if blobs == 1 and coord[1] > 0.7:
                    lspeed, rspeed = 0.0, 0.0
                    print('Me voy a chocar con la pelota')
                else:
                    lspeed, rspeed = avoid(sonar)
                    print('Me voy a chocar')

            else:

                y = np.array([[blobs, coord[0], coord[1], last_coord[0]]])
                speeds = clf.predict(y)
                lspeed = speeds[0][0]
                rspeed = speeds[0][1]
                # if blobs == 1: # Ve la pelota
                #     lspeed, rspeed = follow(coord, last_coord)
                    
                # else: # No la ve
                #     # Planning 
                #     lspeed, rspeed = search(last_coord)
                #     print('No la veo')
            
            # print(lspeed, rspeed)
            # if blobs == 0:
            #     lspeed, rspeed = 0.0, 0.0

            # Action
            setSpeed(clientID, hRobot, lspeed, rspeed)
            time.sleep(0.1)

        print('### Finishing...')
        sim.simxFinish(clientID)

    print('### Program ended')

# --------------------------------------------------------------------------

if __name__ == '__main__':
    main()
