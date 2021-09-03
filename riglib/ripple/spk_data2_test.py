import xipppy as xp
import numpy as np
import time

tcp_flag=False
numElec = 16 #number of electrodes to rec from
max_spk=1 #number of spikes to grab
iter_avg=[]

with xp.xipppy_open(): 
    
    recCh = xp.list_elec('nano')+xp.list_elec('micro')
    print(recCh)
    rec_ch = recCh[0:numElec]
    print('Channel data type:', type(rec_ch[0]))
    spk,c=xp.spk_data(rec_ch[0]) #clear spike buffer
    #spk,c=xp.spk_data2(rec_ch) #clear spike buffer
    print(rec_ch)
    
    iMAX=50
    i=0
    
    while i < iMAX:
        tic=time.time() #start time
        spk_list,c = xp.spk_data(rec_ch[0]) #have to limit max spikes else takes forever
        #spk_list,c = xp.spk_data2(rec_ch,max_spk) #have to limit max spikes else takes forever
        toc=time.time()-tic #end
        #print(spk_list)
        #print(len(c))
        if len(c)>0:
            i=i+1 #if sensed, iterate
            print('elapsed time: ', toc, 'count: ', c)
            iter_avg.append(toc)
    print('\n Mean Time: ',np.mean(iter_avg),'\n Std Dev: ', np.std(iter_avg))