import numpy as np
from math import *
from matplotlib.pyplot import *
from numba import jit

#parameters
nx = 2000
ny = 3

cfl = 0.45
freq_output = 10

Lx = 1.
Ly = 1.

gamma = 2.
cv = 1.5
nt = 400

#grid
dx = Lx/nx
dy = Ly/ny

xf = np.linspace(-dx,Lx+dx,nx+3)
yf = np.linspace(-dy,Ly+dy,ny+3)

xc = np.linspace(-0.5*dx,Lx+0.5*dx,nx+2)
yc = np.linspace(-0.5*dy,Ly+0.5*dy,ny+2)

#data structure
nvar = 6
Uold = np.zeros((nx+2,ny+2,nvar))
Unew = np.zeros((nx+2,ny+2,nvar))

ID = 0
IU = 1
IV = 2
IE = 3
IBx = 4
IBy = 5

x_interface = 0.5  

for i in range(nx+2):
    for j in range(ny+2):
        x = xc[i]
        if x < x_interface:  
            P = 1.0
            Uold[i,j,ID] = 1.0  
            Uold[i,j,IU] = 0.0 
            Uold[i,j,IV] = 0.0 
            Uold[i,j,IBx] = 0.65
            Uold[i,j,IBy] = 1.  
            ekinl = 0
            emagl = 0.5*(Uold[i,j,IBx]**2 + Uold[i,j,IBy]**2)
            Uold[i,j,IE] = P/(gamma-1) + ekinl + emagl
            Unew[i,j,:] = Uold[i,j,:]
        else:  
            P = 0.1
            Uold[i,j,ID] = 0.125  
            Uold[i,j,IU] = 0.0  
            Uold[i,j,IV] = 0.0  
            Uold[i,j,IBx] = 0.65
            Uold[i,j,IBy] = -1.
            ekinr = 0
            emagr = 0.5*(Uold[i,j,IBx]**2 + Uold[i,j,IBy]**2)
            Uold[i,j,IE] = P/(gamma-1) + ekinr + emagr
            Unew[i,j,:] = Uold[i,j,:]

int_rho = sum(sum(Uold[1:nx+1,1:ny+1,ID],0),0)
int_E   = sum(sum(Uold[1:nx+1,1:ny+1,IE],0),0)

@jit(nopython=True)
def compute_timestep(Uold):
    dt = 1E20

    for i in range(1,nx+1,1):
        for j in range(1,ny+1,1):

            #x direction left flux
            rhoc = Uold[i,j,ID]
            uc = Uold[i,j,IU]/rhoc
            vc = Uold[i,j,IV]/rhoc
            ekinc = 0.5*(uc**2+vc**2)*rhoc
            emag = 0.5*(Uold[i,j,IBx]**2 + Uold[i,j,IBy]**2)
            pc = (Uold[i,j,IE]-ekinc-emag)*(gamma-1.)
            c02 = (gamma*pc/rhoc)
            ca2 = 2*emag / rhoc 
            cap2x = Uold[i,j,IBx]**2 / rhoc
            cmfx = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2x))

            cap2y = Uold[i,j,IBy]**2 / rhoc
            cmfy = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2y))

            dt_loc = cfl*dx/max(abs(uc)+cmfx,abs(vc)+cmfy)
            dt = min(dt,dt_loc)

    return dt


@jit(nopython=True)
def compute_kernel(Uold,Unew,dt):

    for i in range(1,nx+1,1):
        for j in range(1,ny+1,1):

            #x direction left flux
            rhol = Uold[i-1,j,ID]
            ul = Uold[i-1,j,IU]/rhol
            vl = Uold[i-1,j,IV]/rhol
            ekinl = 0.5*(ul**2+vl**2)*rhol
            Bxl = Uold[i-1,j,IBx]
            Byl = Uold[i-1,j,IBy]
            eBl = 0.5*(Bxl**2 + Byl**2)
            pth = (Uold[i-1,j,IE]-ekinl-eBl)*(gamma-1.)
            pl = pth + eBl - Bxl*Bxl
            ql = -Bxl*Byl
            c02 = (gamma*pth/rhol)
            ca2 = 2*eBl / rhol
            cap2x = Uold[i-1,j,IBx]**2 / rhol
            cmfl = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2x))
            al = rhol * cmfl

            rhor = Uold[i,j,ID]
            ur = Uold[i,j,IU]/rhor
            vr = Uold[i,j,IV]/rhor
            ekinr = 0.5*(ur**2+vr**2)*rhor
            Bxr = Uold[i,j,IBx]
            Byr = Uold[i,j,IBy]
            eBr = 0.5*(Bxr**2 + Byr**2)
            pth = (Uold[i,j,IE]-ekinr-eBr)*(gamma-1.)
            pr = pth + eBr - Bxr*Bxr
            qr = -Bxr*Byr
            c02 = (gamma*pth/rhor)
            ca2 = 2*eBr / rhor
            cap2x = Uold[i,j,IBx]**2 / rhor
            cmfr = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2x))
            ar = rhor * cmfr

            aface = 1.01*max(al,ar)

            ustar = 0.5*(ul+ur)-0.5*(pr-pl)/aface
            theta = min(abs(ustar)/max(al/rhol,ar/rhor),1)
            pstar = 0.5*(pl+pr)-0.5*(ur-ul)*aface*theta

            vstar = 0.5*(vl+vr)-0.5*(qr-ql)/aface
            qstar = 0.5*(ql+qr)-0.5*(vr-vl)*aface

            flux = np.zeros(nvar)

            if (ustar>0):
                flux[ID] = ustar*Uold[i-1,j,ID]
                flux[IU] = ustar*Uold[i-1,j,IU] + pstar
                flux[IV] = ustar*Uold[i-1,j,IV] + qstar
                flux[IE] = ustar*Uold[i-1,j,IE] + pstar*ustar + qstar*vstar
                flux[IBx] = ustar * Uold[i-1,j,IBx] - ustar * Uold[i,j,IBx]
                flux[IBy] = ustar * Uold[i-1,j,IBy] - vstar * Uold[i,j,IBx]
            else:
                flux[ID] = ustar*Uold[i,j,ID]
                flux[IU] = ustar*Uold[i,j,IU] + pstar
                flux[IV] = ustar*Uold[i,j,IV] + qstar
                flux[IE] = ustar*Uold[i,j,IE] + pstar*ustar + qstar*vstar
                flux[IBx] = ustar * Uold[i,j,IBx] - ustar * Uold[i-1,j,IBx]
                flux[IBy] = ustar * Uold[i,j,IBy] - vstar * Uold[i-1,j,IBx]

            for ivar in range(nvar):
                Unew[i,j,ivar] += (dt/dx)*flux[ivar]

            #x direction right flux
            rhol = Uold[i,j,ID]
            ul = Uold[i,j,IU]/rhol
            vl = Uold[i,j,IV]/rhol
            ekinl = 0.5*(ul**2+vl**2)*rhol
            Bxl = Uold[i,j,IBx]
            Byl = Uold[i,j,IBy]
            eBl = 0.5*(Bxl**2 + Byl**2)
            pth = (Uold[i,j,IE]-ekinl-eBl)*(gamma-1.)
            pl = pth + eBl - Bxl*Bxl
            ql = -Bxl*Byl
            c02 = (gamma*pth/rhol)
            ca2 = 2*eBl / rhol
            cap2x = Uold[i,j,IBx]**2 / rhol
            cmfl = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2x))
            al = rhol * cmfl

            rhor = Uold[i+1,j,ID]
            ur = Uold[i+1,j,IU]/rhor
            vr = Uold[i+1,j,IV]/rhor
            ekinr = 0.5*(ur**2+vr**2)*rhor
            Bxr = Uold[i+1,j,IBx]
            Byr = Uold[i+1,j,IBy]
            eBr = 0.5*(Bxr**2 + Byr**2)
            pth = (Uold[i+1,j,IE]-ekinr-eBr)*(gamma-1.)
            pr = pth + eBr - Bxr*Bxr
            qr = -Bxr*Byr
            c02 = (gamma*pth/rhor)
            ca2 = 2*eBr / rhor
            cap2x = Uold[i+1,j,IBx]**2 / rhor
            cmfr = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2x))
            ar = rhor * cmfr

            aface = 1.1*max(al,ar)

            ustar = 0.5*(ul+ur)-0.5*(pr-pl)/aface
            theta = min(abs(ustar)/max(al/rhol,ar/rhor),1)
            pstar = 0.5*(pl+pr)-0.5*(ur-ul)*aface*theta

            vstar = 0.5*(vl+vr)-0.5*(qr-ql)/aface
            qstar = 0.5*(ql+qr)-0.5*(vr-vl)*aface

            flux = np.zeros(nvar)

            if (ustar>0):
                flux[ID] = ustar*Uold[i,j,ID]
                flux[IU] = ustar*Uold[i,j,IU] + pstar
                flux[IV] = ustar*Uold[i,j,IV] + qstar
                flux[IE] = ustar*Uold[i,j,IE] + pstar*ustar + qstar*vstar
                flux[IBx] = ustar * Uold[i,j,IBx] - ustar * Uold[i+1,j,IBx]
                flux[IBy] = ustar * Uold[i,j,IBy] - vstar * Uold[i+1,j,IBx]
            else:
                flux[ID] = ustar*Uold[i+1,j,ID]
                flux[IU] = ustar*Uold[i+1,j,IU] + pstar
                flux[IV] = ustar*Uold[i+1,j,IV] + qstar
                flux[IE] = ustar*Uold[i+1,j,IE] + pstar*ustar + qstar*vstar
                flux[IBx] = ustar * Uold[i+1,j,IBx] - ustar * Uold[i,j,IBx]
                flux[IBy] = ustar * Uold[i+1,j,IBy] - vstar * Uold[i,j,IBx]

            for ivar in range(nvar):
                Unew[i,j,ivar] -= (dt/dx)*flux[ivar]

            #y direction left flux
            rhol = Uold[i,j-1,ID]
            ul = Uold[i,j-1,IU]/rhol
            vl = Uold[i,j-1,IV]/rhol
            ekinl = 0.5*(ul**2+vl**2)*rhol
            Bxl = Uold[i,j-1,IBx]
            Byl = Uold[i,j-1,IBy]
            eBl = 0.5*(Bxl**2 + Byl**2)
            pl = -Bxl*Byl
            pth = (Uold[i,j-1,IE]-ekinl-eBl)*(gamma-1.)
            ql = pth + eBl - Byl*Byl
            c02 = (gamma*pth/rhol)
            ca2 = 2*eBl / rhol
            cap2y = Uold[i,j-1,IBy]**2 / rhol
            cmfl = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2y))
            al = rhol * cmfl

            rhor = Uold[i,j,ID]
            ur = Uold[i,j,IU]/rhor
            vr = Uold[i,j,IV]/rhor
            ekinr = 0.5*(ur**2+vr**2)*rhor
            Bxr = Uold[i,j,IBx]
            Byr = Uold[i,j,IBy]
            eBr = 0.5*(Bxr**2 + Byr**2)
            pr = -Byr*Bxr
            pth = (Uold[i,j,IE]-ekinr-eBr)*(gamma-1.)
            qr = pth + eBr - Byr*Byr
            c02 = (gamma*pth/rhor)
            ca2 = 2*eBr / rhor
            cap2y = Uold[i,j,IBy]**2 / rhor
            cmfr = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2y))
            ar = rhor * cmfr

            aface = 1.01*max(al,ar)

            #normale
            ustar = 0.5*(vl+vr)-0.5*(qr-ql)/aface
            theta = min(abs(ustar)/max(al/rhol,ar/rhor),1)
            pstar = 0.5*(ql+qr)-0.5*(vr-vl)*aface*theta

            #tangentielle
            vstar = 0.5*(ul+ur)-0.5*(pr-pl)/aface
            qstar = 0.5*(pl+pr)-0.5*(ur-ul)*aface

            flux = np.zeros(nvar)

            if (ustar>0):
                flux[ID] = ustar*Uold[i,j-1,ID]
                flux[IU] = ustar*Uold[i,j-1,IU] + qstar
                flux[IV] = ustar*Uold[i,j-1,IV] + pstar
                flux[IE] = ustar*Uold[i,j-1,IE] + pstar*ustar + qstar*vstar
                flux[IBx] = ustar * Uold[i,j-1,IBx] - vstar * Uold[i,j,IBy] 
                flux[IBy] = ustar * Uold[i,j-1,IBy] - ustar * Uold[i,j,IBy]
            else:
                flux[ID] = ustar*Uold[i,j,ID]
                flux[IU] = ustar*Uold[i,j,IU] + qstar
                flux[IV] = ustar*Uold[i,j,IV] + pstar
                flux[IE] = ustar*Uold[i,j,IE] + pstar*ustar + qstar*vstar
                flux[IBx] = ustar * Uold[i,j,IBx] - vstar * Uold[i,j-1,IBy]
                flux[IBy] = ustar * Uold[i,j,IBy] - ustar * Uold[i,j-1,IBy]

            for ivar in range(nvar):
               Unew[i,j,ivar] += (dt/dy)*flux[ivar]

            #y direction right flux
            rhol = Uold[i,j,ID]
            ul = Uold[i,j,IU]/rhol
            vl = Uold[i,j,IV]/rhol
            ekinl = 0.5*(ul**2+vl**2)*rhol
            Bxl = Uold[i,j,IBx]
            Byl = Uold[i,j,IBy]
            eBl = 0.5*(Bxl**2 + Byl**2)
            pl = -Bxl*Byl
            pth = (Uold[i,j,IE]-ekinl-eBl)*(gamma-1.)
            ql = pth + eBl - Byl*Byl
            c02 = (gamma*pth/rhol)
            ca2 = 2*eBl / rhol
            cap2y = Uold[i,j,IBy]**2 / rhol
            cmfl = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2y))
            al = rhol * cmfl

            rhor = Uold[i,j+1,ID]
            ur = Uold[i,j+1,IU]/rhor
            vr = Uold[i,j+1,IV]/rhor
            ekinr = 0.5*(ur**2+vr**2)*rhor
            Bxr = Uold[i,j+1,IBx]
            Byr = Uold[i,j+1,IBy]
            eBr = 0.5*(Bxr**2 + Byr**2)
            pr = -Byr*Bxr
            pth = (Uold[i,j+1,IE]-ekinr-eBr)*(gamma-1.)
            qr = pth + eBr - Byr*Byr
            c02 = (gamma*pth/rhor)
            ca2 = 2*eBr / rhor
            cap2y = Uold[i,j+1,IBy]**2 / rhor
            cmfr = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2y))
            ar = rhor * cmfr

            aface = 1.01*max(al,ar)

            #normale
            ustar = 0.5*(vl+vr)-0.5*(qr-ql)/aface
            theta = min(abs(ustar)/max(al/rhol,ar/rhor),1)
            pstar = 0.5*(ql+qr)-0.5*(vr-vl)*aface*theta

            #tangentielle
            vstar = 0.5*(ul+ur)-0.5*(pr-pl)/aface
            qstar = 0.5*(pl+pr)-0.5*(ur-ul)*aface

            flux = np.zeros(nvar)

            if (ustar>0):
                flux[ID] = ustar*Uold[i,j,ID]
                flux[IU] = ustar*Uold[i,j,IU] + qstar
                flux[IV] = ustar*Uold[i,j,IV] + pstar
                flux[IE] = ustar*Uold[i,j,IE] + pstar*ustar + qstar*vstar
                flux[IBx] = ustar * Uold[i,j,IBx] - vstar * Uold[i,j+1,IBy] 
                flux[IBy] = ustar * Uold[i,j,IBy] - ustar * Uold[i,j+1,IBy]
            else:
                flux[ID] = ustar*Uold[i,j+1,ID]
                flux[IU] = ustar*Uold[i,j+1,IU] + qstar
                flux[IV] = ustar*Uold[i,j+1,IV] + pstar
                flux[IE] = ustar*Uold[i,j+1,IE] + pstar*ustar + qstar*vstar
                flux[IBx] = ustar * Uold[i,j+1,IBx] - vstar * Uold[i,j,IBy]
                flux[IBy] = ustar * Uold[i,j+1,IBy] - ustar * Uold[i,j,IBy]

            for ivar in range(nvar):
                Unew[i,j,ivar] -= (dt/dy)*flux[ivar]

#time loop
iout = 0
time = 0.
tend = 0.2
it = 0.

figure(1)
clf()
imshow(Unew[:,:,IBy], origin='lower')
colorbar()
savefig('output_blast'+str(iout).zfill(3)+'.png')
iout += 1
t_values = [time]
Bx_values = [0.65]

while time < tend:
    it+=1
    print("timestep: ",it)

    #compute time step
    dt = compute_timestep(Uold)
    time +=dt
    t_values.append(time)

    #advection equation
    compute_kernel(Uold,Unew,dt)
    Bx_values.append(Unew[i,j,IBx])

    #copy Unew in Uold
    Uold = Unew.copy()

    # Bords gauche et droit, Dirichlet
    Uold[0, :, ID] = 1.0      
    Uold[nx+1, :, ID] = 0.125   
    Uold[0, :, IU] = 0.0      
    Uold[nx+1, :, IU] = 0.0  
    Uold[0, :, IV] = 0.0      
    Uold[nx+1, :, IV] = 0.0
    Uold[0, :, IBx] = 0.65     
    Uold[nx+1, :, IBx] = 0.65  
    Uold[0, :, IBy] = 1.    
    Uold[nx+1, :, IBy] = -1.
    ekinl = 0
    emagl = 0.5*(Uold[0,:,IBx]**2 + Uold[0,:,IBy]**2)
    P = 1.
    Uold[0,:,IE] = P/(gamma-1) + ekinl + emagl 
    P = 0.1
    Uold[nx+1,:,IE] = P/(gamma-1) + ekinl + emagl 


    # Bords haut et bas, Dirichlet
    for i in range(nx+2):
        x = xc[i]
        if x < x_interface:  
            P = 1.0
            Uold[i,0,ID] = 1.0  
            Uold[i,0,IU] = 0.0 
            Uold[i,0,IV] = 0.0 
            Uold[i,0,IBx] = 0.65
            Uold[i,0,IBy] = 1.  
            ekinl = 0
            emagl = 0.5*(Uold[i,0,IBx]**2 + Uold[i,0,IBy]**2)
            Uold[i,0,IE] = P/(gamma-1) + ekinl + emagl 
            Uold[i,ny+1,:] =  Uold[i,0,:]
        else:  
            P = 0.1
            Uold[i,0,ID] = 0.125  
            Uold[i,0,IU] = 0.0  
            Uold[i,0,IV] = 0.0  
            Uold[i,0,IBx] = 0.65
            Uold[i,0,IBy] = -1.
            ekinr = 0
            emagr = 0.5*(Uold[i,0,IBx]**2 + Uold[i,0,IBy]**2)
            Uold[i,0,IE] = P/(gamma-1) + ekinr + emagr
            Uold[i,ny+1,:] = Uold[i,0,:]

#final output
clf()
print("time: ",time," rho, E conservation: ",abs(int_rho-sum(sum(Uold[1:nx+1,1:ny+1,ID],0),0))/int_rho, abs(int_E-sum(sum(Uold[1:nx+1,1:ny+1,IE],0),0))/int_E)
#compute error

j_mid = ny // 2 
rho_values = Unew[1:nx+1, j_mid, ID] 
By_values = Unew[1:nx+1, j_mid, IBy] 
x_values = xc[1:nx+1] 

figure(1)
plot(t_values, Bx_values, label="Constance de Bx tout au long de la simu?")
xlabel("t")
ylabel("BX")
title("Bx en fonction de t".format(time))
grid(True)
legend()
savefig("Bx.png")
show()

figure(2)
plot(x_values, By_values, label="Densité By (au milieu de la grille)")
xlabel("x")
ylabel("Densité By")
title("Densité By en fonction de x à t = {:.2f}".format(time))
grid(True)
legend()
savefig("By.png")
show()

figure(3)
plot(x_values, rho_values, label="Densité rho (au milieu de la grille)")
xlabel("x")
ylabel("Densité rho")
title("Densité rho en fonction de x à t = {:.2f}".format(time))
grid(True)
legend()
savefig("rho.png")
show()