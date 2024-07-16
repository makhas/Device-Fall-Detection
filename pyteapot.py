"""
PyTeapot module for drawing rotating cube using OpenGL as per
quaternion or yaw, pitch, roll angles received over serial port.
"""
import pygame
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *


useQuat = True   # set true for using quaternions, false for using y,p,r angles
q = [ 0.48416448,  0.50199875,  0.6958132 , -0.17154008] #initial attitude for delta calculations
degs_per_rad = 180/np.pi

def main():
    video_flags = OPENGL | DOUBLEBUF
    pygame.init()
    screen = pygame.display.set_mode((640, 480), video_flags)
    clock = pygame.time.Clock()
    pygame.display.set_caption("PyTeapot IMU orientation visualization")
    resizewin(640, 480)
    init()
    fps=2
    frames = 0
    n = 900
    
    pitch_from_q =[euler_to_quaternion(np.array(quaternion_to_euler(q)) + np.array([0,np.pi*(i+1)/20,np.pi*(i+1)/45])) for i in range(n)] #np.pi*(i+1)/45
    quats = pitch_from_q
    ticks = pygame.time.get_ticks()
    while 1:
        clock.tick(fps)
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break
        if(useQuat):
            i = frames % n
            [w, nx, ny, nz] = quats[i]
        else:
            [yaw, pitch, roll] = read_data()
        if(useQuat):
            draw(w, nx, ny, nz)
        else:
            draw(1, yaw, pitch, roll)
        pygame.display.flip()
        frames += 1
    print("fps: %d" % ((frames*1000)/(pygame.time.get_ticks()-ticks)))

def sample_sphere(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /=np.linalg.norm(vec, axis=0)
    #vec = vec.flatten()
    return vec.T

def resizewin(width, height):
    """
    For resizing window
    """
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0*width/height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)


def cleanSerialBegin():
    if(useQuat):
        try:
            line = ser.readline().decode('UTF-8').replace('\n', '')
            w = float(line.split('w')[1])
            nx = float(line.split('a')[1])
            ny = float(line.split('b')[1])
            nz = float(line.split('c')[1])
        except Exception:
            pass
    else:
        try:
            line = ser.readline().decode('UTF-8').replace('\n', '')
            yaw = float(line.split('y')[1])
            pitch = float(line.split('p')[1])
            roll = float(line.split('r')[1])
        except Exception:
            pass


def read_data():
    if(useSerial):
        ser.reset_input_buffer()
        cleanSerialBegin()
        line = ser.readline().decode('UTF-8').replace('\n', '')
        print(line)
    else:
        # Waiting for data from udp port 5005
        data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        line = data.decode('UTF-8').replace('\n', '')
        print(line)
                
    if(useQuat):
        w = float(line.split('w')[1])
        nx = float(line.split('a')[1])
        ny = float(line.split('b')[1])
        nz = float(line.split('c')[1])
        return [w, nx, ny, nz]
    else:
        yaw = float(line.split('y')[1])
        pitch = float(line.split('p')[1])
        roll = float(line.split('r')[1])
        
        return [yaw, pitch, roll]


def draw(w, nx, ny, nz):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0.0, -7.0)

    drawText((-2.6, 1.8, 2), "PyTeapot", 18)
    drawText((-2.6, 1.6, 2), "Module to visualize quaternion or Euler angles data", 16)
    drawText((-2.6, -2, 2), "Press Escape to exit.", 16)

    if(useQuat):
        [yaw, pitch , roll] = np.array(quaternion_to_euler([w, nx, ny, nz])) - np.array(quaternion_to_euler(q))
        modulo = np.pi
        if yaw < 0:
            yaw += modulo
        if pitch<0:
            pitch += modulo
        if roll<0:
            pitch += modulo

        if (pitch*degs_per_rad>50) or (roll*degs_per_rad>50):

            drawText((-2.6, -1.8, 2), "Yaw: %f, Pitch: %f, Roll: %f" %(yaw*degs_per_rad, pitch*degs_per_rad, roll*degs_per_rad), 16, (255,0,0,0))
        else:
            drawText((-2.6, -1.8, 2), "Yaw: %f, Pitch: %f, Roll: %f" %(yaw*degs_per_rad, pitch*degs_per_rad, roll*degs_per_rad), 16)
        glRotatef(2 * math.acos(w) * 180.00/math.pi, -1 * nx, nz, ny)
    else:
        yaw = nx
        pitch = ny
        roll = nz
        modulo = np.pi
        if yaw < 0:
            yaw += modulo
        if pitch<0:
            pitch += modulo
        if roll<0:
            pitch += modulo
        drawText((-2.6, -1.8, 2), "Yaw: %f, Pitch: %f, Roll: %f" %(yaw, pitch, roll), 16)
        glRotatef(-roll, 0.00, 0.00, 1.00)
        glRotatef(pitch, 1.00, 0.00, 0.00)
        glRotatef(yaw, 0.00, 1.00, 0.00)

    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(1.0, 0.2, 1.0)

    glColor3f(1.0, 0.5, 0.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(1.0, -0.2, -1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, -1.0)
    glEnd()


def drawText(position, textString, size, color= (255, 255, 255, 255)):
    font = pygame.font.SysFont("Courier", size, True)
    textSurface = font.render(textString, True, color, (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def quat_to_ypr(q): #[w, x, y, z]
    yaw   = math.atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
    pitch = -math.asin(2.0 * (q[1] * q[3] - q[0] * q[2]))
    roll  = math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])
    pitch *= 180.0 / math.pi
    yaw   *= 180.0 / math.pi
    #yaw   -= -0.13  # Declination at Chandrapur, Maharashtra is - 0 degress 13 min
    roll  *= 180.0 / math.pi
    return [yaw, pitch, roll]

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    q= [qw, qx, qy, qz]
    return np.array(q)

def quaternion_to_euler(q):
    (w, x, y, z) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.array([yaw, pitch, roll])

if __name__ == '__main__':
    main()