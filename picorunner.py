#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:38:02 2017

@author: cordell
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from astropy.io import fits
import astropy.cosmology as co
from matplotlib.mlab import normpdf
import pypico

cosmodir="/home/cordell/Research/Cosmology/"


font = {'family' : 'normal',
        'weight' : 'medium',
        'size'   : 12}

mpl.rc('font', **font)






def omh2(omega,H0):
    return omega*(H0**2/10000)

def theta(initialtheta,initialH0,newH0):
    return initialtheta*newH0 / initialH0


#planckdat=np.loadtxt(cosmodir+"plc_2.0/hi_l/plik_lite/plik_lite_v18_TT.clik/clik/lkl_0/_external/base_plikHM_TT_lowTEB.dat")
#plkl=planckdat[:,0].astype('int')


# data from
# http://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/previews/ps_index.html
planckfits=fits.open(cosmodir+"COM_PowerSpect_CMB_R2.02.fits")

planckELL=planckfits[8].data['ELL']
planckTT=planckfits[8].data['D_ELL']
planckERR=planckfits[8].data['ERR']

planckbELL=planckfits[7].data['ELL']
planckbTT=planckfits[7].data['D_ELL']
planckbERR=planckfits[7].data['ERR']



initialNeff = co.parameters.Planck15['Neff']
initialH0 = co.parameters.Planck15['H0']
initialomb = co.parameters.Planck15['Ob0']
initialomc = co.parameters.Planck15['Oc0']
initialtau = co.parameters.Planck15['tau']
initialtheta = 1.04085 /100   #Planck15


oldcolor='blue'
newcoloracc='lime'
newcolorrej='r'


pico = pypico.load_pico(cosmodir+"pico3_tailmonty_v34.dat")

inp = pico.example_inputs()

inp['massive_neutrinos'] = initialNeff
inp['ombh2'] = omh2(initialomb,initialH0)
inp['omch2'] = omh2(initialomc,initialH0)
inp['theta'] = theta(initialtheta,initialH0,initialH0)
inp['re_optical_depth']=initialtau

# sigmas for picking new values
sigmaNeff = .5 #1
sigmaH0= 2

sigmatarget=0.23  # statistically ideal value
sigmarate = 0.01  # adjust this
#sigmascaler=.5
#sigmincount=4

bincount=50#20

mcn=500
burnin=400

oldNeff = initialNeff
oldH0 = initialH0
oldx2 = 1e9
#oldlik = 1e-9

Nefflist=[]
H0list=[]

Nlim = (0,6)
Hlim = (64,76)

fig=plt.figure(1,figsize=(12,12))
fig.clf()
#ax1=plt.subplot(2,2,3)
ax1 = plt.subplot2grid([3,3],[1,0],rowspan=2,colspan=2)
#ax1.set_title('Scatter')
ax1.set_xlabel(r'$H_0 \left( \frac{km/s}{Mpc}\right) $')
ax1.set_ylabel(r'$N_{eff}$')
plotdata, =ax1.plot([],[],'ko',markersize=3)
#ax1.set_autoscaley_on(True)
#ax1.set_autoscalex_on(True)
ax1.set_ylim(Nlim)
ax1.set_xlim(Hlim)

greenpercent = 23
gpax=plt.text(2000,5000,str(greenpercent))


reissH=73.24
reissHerr=1.74
reissrange=reissH + reissHerr* np.array([1,-1])

Nup = Nlim[1]+np.zeros(2)
Ndown = Nlim[0]+np.zeros(2)

ax1.fill_between(reissrange,Nup,Ndown,alpha=0.2,color='grey')
ax1.plot(reissH*np.ones(2),Nlim,'k-',linewidth=3)


ax1.plot(Hlim,initialNeff*np.ones(2),':k',linewidth=3)


# top distribution
axH = plt.subplot2grid([3,3],[0,0],rowspan=1,colspan=2)
histH,=axH.plot([],[])
axH.set_ylim((0,1))
#axH.set_autoscalex_on(True)
axH.set_xlim(Hlim)
axN = plt.subplot2grid([3,3],[1,2],rowspan=2,colspan=1)
histN,=axN.plot([],[])
axN.set_xlim((0,1))
#axN.set_autoscaley_on(True)
axN.set_ylim(Nlim)

# side distribution
axH.plot(reissH*np.ones(2),(0,1),'k-',linewidth=3)
axH.fill_between(reissrange,(1,1),(0,0),alpha=0.2,color='grey')

axN.plot((0,1),initialNeff*np.ones(2),':k',linewidth=3)

axH.xaxis.tick_top()
axN.yaxis.tick_right()
axH.set_yticklabels([])
axN.set_xticklabels([])


newspotH,= axH.plot([0,0],[0,1],linewidth=5)
oldspotH,= axH.plot([0,0],[0,1],linewidth=5,color=oldcolor)

newspotN,= axN.plot([0,1],[0,0],linewidth=5)
oldspotN,= axN.plot([0,1],[0,0],linewidth=5,color=oldcolor)

olddistHx = np.linspace(Hlim[0],Hlim[1],100)
olddistNx = np.linspace(Nlim[0],Nlim[1],100)

olddistH,=axH.plot(olddistHx,olddistHx*0,color=oldcolor)
olddistN,=axN.plot(olddistNx*0,olddistNx,color=oldcolor)


newspot, =ax1.plot([],[],'o',markersize=20)
oldspot, =ax1.plot([],[],'o',markersize=20,color=oldcolor)

ax2=plt.subplot2grid([3,3],[0,2],rowspan=1,colspan=1)
ax2.errorbar(planckbELL,planckbTT,planckbERR,fmt='.')
psdat,=ax2.plot([],[])
#ax2.set_title('TT Power Spectrum')
ax2.set_ylabel(r'$C_l$')
ax2.set_xlabel(r'$l$')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')





greencount = 1

for i in range(0,burnin):
    
    # generate new sigmas

#    if np.size(Nefflist) > sigmincount:
#        sigmaNeff = np.std(Nefflist)*sigmascaler
#        sigmaH0= np.std(H0list)*sigmascaler

    #print greencount/((i+1)*sigmatarget),
    if greencount < i*sigmatarget:
        sigmaNeff = sigmaNeff*(1-sigmarate)
        sigmaH0 = sigmaH0*(1-sigmarate)
        #print "-",greencount
    else:
        sigmaNeff = sigmaNeff*(1+sigmarate)
        sigmaH0 = sigmaH0*(1+sigmarate)
        #print "+",greencount


    
    # generate new Neff,H0
    newNeff = np.random.normal(oldNeff,sigmaNeff)
    newH0   = np.random.normal(oldH0,sigmaH0)
    
    inp['massive_neutrinos'] = newNeff
    inp['ombh2'] = omh2(initialomb,newH0)
    inp['omch2'] = omh2(initialomc,newH0)
    inp['theta'] = theta(initialtheta,initialH0,newH0)

    
    # Run through pico
    result = pico.get(force=True,**inp)

    # Test with MH
    newx2 = np.sum((result['cl_TT'][planckELL] - planckTT)**2/planckERR**2)
    #newlik=np.exp(-x2/2.)
    cutoff = np.random.rand()
    
    #print newx2, ", ",oldx2,", ",cutoff

    
    if oldx2/newx2 > cutoff:  
        
    # Refactoring Lnew/Lold > cutoff
    #if newx2-oldx2 < -2*np.log( cutoff):  
        
        
        greencount=greencount+1
        
        Nefflist.append(newNeff)
        H0list.append(newH0)

        oldx2=newx2
        oldNeff=newNeff
        oldH0=newH0



Nefflist=[]
H0list=[]

normH=1/(2*normpdf(0,0,sigmaH0))
normN=1/(2*normpdf(0,0,sigmaNeff))

greencount=1

for i in range(burnin,mcn):
    
    
    # generate new Neff,H0
    newNeff = np.random.normal(oldNeff,sigmaNeff)
    newH0   = np.random.normal(oldH0,sigmaH0)
    
    inp['massive_neutrinos'] = newNeff
    inp['ombh2'] = omh2(initialomb,newH0)
    inp['omch2'] = omh2(initialomc,newH0)
    inp['theta'] = theta(initialtheta,initialH0,newH0)

    
    # Run through pico
    result = pico.get(force=True,**inp)

    # Test with MH
    newx2 = np.sum((result['cl_TT'][planckELL] - planckTT)**2/planckERR**2)
    #newlik=np.exp(-x2/2.)
    cutoff = np.random.rand()
    
    #greenpercent=100*greencount/float(i)
    #gpax.set_text(str(round(greenpercent)))
    
    
    plotdata.set_xdata(H0list)
    plotdata.set_ydata(Nefflist)
        
    newspot.set_xdata([newH0])
    newspot.set_ydata([newNeff])
    newspot.set_markerfacecolor(newcolorrej)
    
    newspotH.set_xdata(newH0*np.ones(2))
    newspotN.set_ydata(newNeff*np.ones(2))
    newspotH.set_markerfacecolor(newcolorrej)
    newspotN.set_markerfacecolor(newcolorrej)

    


    Hbins,Hedges=np.histogram(H0list,bincount,range=Hlim,normed=True)
    histH.set_xdata(np.array([Hedges[:-1],Hedges[1:]]).T.flatten())
    histH.set_ydata(np.array([Hbins,Hbins]).T.flatten()/(1.1*Hbins.max()))
    Nbins,Nedges=np.histogram(Nefflist,bincount,range=Nlim,normed=True)
    histN.set_ydata(np.array([Nedges[:-1],Nedges[1:]]).T.flatten())
    histN.set_xdata(np.array([Nbins,Nbins]).T.flatten()/(1.1*Nbins.max()))
    
    psdat.set_ydata(result['cl_TT'])
    psdat.set_xdata(np.arange(result['cl_TT'].size))
    psdat.set_color(newcolorrej)

    
    oldspot.set_xdata([oldH0])
    oldspot.set_ydata([oldNeff])

    oldspotH.set_xdata(oldH0*np.ones(2))
    oldspotN.set_ydata(oldNeff*np.ones(2))
    newspotH.set_color(newcolorrej)
    newspotN.set_color(newcolorrej)

    
    olddistH.set_ydata(normH*normpdf(olddistHx,oldH0,sigmaH0))
    olddistN.set_xdata(normN*normpdf(olddistNx,oldNeff,sigmaNeff))


    if oldx2/newx2 > cutoff:  
    #if newx2-oldx2 < -2*np.log( cutoff):  
        
        greencount=greencount+1
        
        # Record into distribution
        Nefflist.append(newNeff)
        H0list.append(newH0)

        
        newspot.set_markerfacecolor(newcoloracc)
        psdat.set_color(newcoloracc)
        newspotH.set_color(newcoloracc)
        newspotN.set_color(newcoloracc)
        
        
        oldx2=newx2
        oldNeff=newNeff
        oldH0=newH0
    else:
        # Record into distribution
        Nefflist.append(oldNeff)
        H0list.append(oldH0)




#    ax1.relim()
#    ax1.autoscale_view()
#    axH.relim()
#    axH.autoscale_view()
#    axN.relim()
#    axN.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


if True: 
    for i in range(mcn,mcn+100):
        
        
        # generate new Neff,H0
        newNeff = np.random.normal(oldNeff,sigmaNeff)
        newH0   = np.random.normal(oldH0,sigmaH0)
        
        inp['massive_neutrinos'] = newNeff
        inp['ombh2'] = omh2(initialomb,newH0)
        inp['omch2'] = omh2(initialomc,newH0)
        inp['theta'] = theta(initialtheta,initialH0,newH0)
        
        
        # Run through pico
        result = pico.get(force=True,**inp)
        
        # Test with MH
        newx2 = np.sum((result['cl_TT'][planckELL] - planckTT)**2/planckERR**2)
        #newlik=np.exp(-x2/2.)
        cutoff = np.random.rand()
        
        
        
        
        plotdata.set_xdata(H0list)
        plotdata.set_ydata(Nefflist)
            
        newspot.set_xdata([newH0])
        newspot.set_ydata([newNeff])
        newspot.set_markerfacecolor(newcolorrej)
        
        newspotH.set_xdata(newH0*np.ones(2))
        newspotN.set_ydata(newNeff*np.ones(2))
        newspotH.set_markerfacecolor(newcolorrej)
        newspotN.set_markerfacecolor(newcolorrej)
        
        
        
        
        Hbins,Hedges=np.histogram(H0list,bincount,range=Hlim,normed=True)
        histH.set_xdata(np.array([Hedges[:-1],Hedges[1:]]).T.flatten())
        histH.set_ydata(np.array([Hbins,Hbins]).T.flatten()/(1.1*Hbins.max()))
        Nbins,Nedges=np.histogram(Nefflist,bincount,range=Nlim,normed=True)
        histN.set_ydata(np.array([Nedges[:-1],Nedges[1:]]).T.flatten())
        histN.set_xdata(np.array([Nbins,Nbins]).T.flatten()/(1.1*Nbins.max()))
        
        psdat.set_ydata(result['cl_TT'])
        psdat.set_xdata(np.arange(result['cl_TT'].size))
        psdat.set_color(newcolorrej)
        
        
        oldspot.set_xdata([oldH0])
        oldspot.set_ydata([oldNeff])
        
        oldspotH.set_xdata(oldH0*np.ones(2))
        oldspotN.set_ydata(oldNeff*np.ones(2))
        newspotH.set_color(newcolorrej)
        newspotN.set_color(newcolorrej)
        
        
        olddistH.set_ydata(normH*normpdf(olddistHx,oldH0,sigmaH0))
        olddistN.set_xdata(normN*normpdf(olddistNx,oldNeff,sigmaNeff))
        
        if oldx2/newx2 > cutoff:  
        #if newx2-oldx2 < -2*np.log( cutoff):  
            
            # Record into distribution
            Nefflist.append(newNeff)
            H0list.append(newH0)
    
            
            newspot.set_markerfacecolor(newcoloracc)
            psdat.set_color(newcoloracc)
            newspotH.set_color(newcoloracc)
            newspotN.set_color(newcoloracc)
            
            
            oldx2=newx2
            oldNeff=newNeff
            oldH0=newH0
        else:
            # Record into distribution
            Nefflist.append(oldNeff)
            H0list.append(oldH0)
        
        
        
        #    ax1.relim()
        #    ax1.autoscale_view()
        #    axH.relim()
        #    axH.autoscale_view()
        #    axN.relim()
        #    axN.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        raw_input('Press Enter to Step Forward:')
    





#clTE=result['cl_TE']


#for key in result.keys():
#    plt.plot(result[key])
#    plt.title(key)
#    plt.show()

#plt.plot(result['cl_TT'][planckELL] - planckTT)
#plt.title('cl_TT - Planck')
#plt.show()

#plt.show()
