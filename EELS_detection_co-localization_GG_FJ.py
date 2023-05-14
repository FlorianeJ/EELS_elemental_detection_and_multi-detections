
#%%  EELS_elemental_detection_GG_FJ.py : This script compiled the elemental detection and multi-detections for each pixel of EELS data.
#Firstly, the code locate the elemental thresholds of C K-edge, O K-edge, Fe L-edges, Al K-edge and Si K-edge.
#Then a power law function fits and removes the background per elementary threshold, as defined by Egerton 2011.
#The energy range of the power law fits are here after referred as "Background_fit".
#Then, a determination of an elementary detection is made by computing the signal to noise ratio (SNR) adapted from Egerton (1982, 2011).
#The SNR detection threshold is hereafter referred to as "SNR_detection.


#%% Title         : Elemental detection of C (K-edge), O (K-edge), Fe (L-edge), Al (K-edge) and Si (K-edge) in each pixel of EELS data
#%% Project       : nanoSoilC ANR Project
#%%
#%% File          : EELS_elemental_detection_GG_FJ.py
#%% Author        : GASSIER ghislain <gassier@cerege.fr>, JAMOTEAU floriane <floriane.jamoteau@unil.ch>
#%%
#%% Created       : 2023/03/01
#%% Last checkin  : $Date: 2023/03/20 15:28:47 $ by $Author: Gassier and Jamoteau
#%% Revision      : $Revision:  $
#%%---------------------------------------------------------------------------


from os.path import isfile

import dm4reader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import integrate
import re
from itertools import combinations
repData="Data/"

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx  # array[idx]
def find_nearest_values(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def Ib_Ik_computation (popt, pcov, ParameterElement, index, Signal, Model, ligne, colonne, elt):
    #Shift the background fit (model) and the data (signal) into a ln.ln frame
    Model=np.log(Model)
    Signal=np.log(Signal)
    Model = Model[:, 0]
    Signal = Signal[:, 0]

    #Definition of background fit and integration boundaries (in energy) for Ik and Ib areas
    IEk = find_nearest(index, ParameterElement[elt]['aire'][0])
    IEn = find_nearest(index, ParameterElement[elt]['aire'][1])
    IEj = find_nearest(index, ParameterElement[elt]['fit'][0])
    Ek = find_nearest_values(index, ParameterElement[elt]['aire'][0])
    En = find_nearest_values(index, ParameterElement[elt]['aire'][1])
    Ej = find_nearest_values(index, ParameterElement[elt]['fit'][0])

    #Recall of the power law parameters (a=np.log(A * Em ** (-r))) to fit the background
    A = popt[0]
    r = popt[1]
    Em = (Ej * Ek) ** (1 / 2)
    a = np.log(A * Em ** (-r))
    b = -r

    # Translation of the En and Ek integration boundaries in a ln frame
    xn = np.log(En / Em)
    xk = np.log(Ek / Em)

    #Integration of the Ib area (background air) using the background fit (model)
    f1 = interpolate.interp1d(np.log(index[IEk:IEn]), np.log(Model[IEk:IEn]), fill_value="extrapolate")
    y1, err1 = integrate.quad(f1, np.log(Ek), np.log(En))
    Ib = y1

    #Integration of the Ik area (elemental threshold) using the signal fit (data)
    f2 = interpolate.interp1d(np.log(index[IEk:IEn]), np.log(Signal[IEk:IEn]), fill_value="extrapolate")
    y2, err2 = integrate.quad(f2, np.log(Ek), np.log(En))
    Ik = y2 - Ib

    #Choice of an elementary detection according to the SNR
    SNR_criteria = Ik / ((Ik + h * Ib) ** (1 / 2))
    if (np.exp(SNR_criteria) >= ParameterElement[elt]["SNR_detection"]):
        detection = True
    else:
        detection = False

    #visualization of the fit and the elementary threshold in a ln.ln frame
    plt.plot(np.log(index), np.log(Signal), 'b-')
    plt.plot(np.log(index), np.log(Model), 'g-')
    #(disabled to avoid memory saturation problems, to disable it, remove the  # in front of the line)
    #fig.savefig("figure/" + str(elt) + "(" + str(ligne) + "," + str(colonne) + ")" + str(np.exp(Ib_Ik_computation)) + ".png")
    return (detection)


#Creation of a list of elemental thresholds for Carbon, Oxygen, Aluminium, Silicium K-edges and Iron L-edge
#This list contains several entries including:
#-Background_fit stand for energy boundaries where the lower low function is fitted
#-Intregation_boundaries stand for energy boundaries where Ib and Ik areas are computed (see def Ib_Ik_computation)
#-A counter is created to count each elemental detection if the SNR is higher than the "SNR_detection" threshold
#-SNR_dectetion stand for the SNR detection threshold
ListeElement = ['Carbon', 'Oxygen', 'Iron', 'Aluminium', 'Silicium']
ParameterElement = {}
ParameterElement['Carbon'] = {"Background_fit": (253, 278), "Intregation_boundaries": (280, 305), "counter": 0,"SNR_detection":1.0009}
ParameterElement['Oxygen'] = {"Background_fit": (465, 515), "Intregation_boundaries": (525, 570), "counter": 0,"SSNR_detection":1.004}
ParameterElement['Iron'] = {"Background_fit": (628,690), "Intregation_boundaries": (693, 720), "counter": 0,"SNR_detection":1.001}
ParameterElement['Aluminium'] = {"Background_fit": (1432, 1532), "Intregation_boundaries": (1545, 1674), "counter": 0,"SNR_detection":1.0019}
ParameterElement['Silicium'] = {"Background_fit": (1654, 1815), "Intregation_boundaries": (1830, 1970), "counter": 0,"SNR_detection":1.0012}
CombinaisonElement = []
for j in range(1, len(ListeElement)):
    temp = combinations(ListeElement, j)
    l = list(temp)
    for i in range(len(l)):
        # print("".join(l[i]))
        CombinaisonElement.append("".join(l[i]))
CombinaisonElement.append("".join(ListeElement))
Compteur = {}
for e in CombinaisonElement:
    Compteur[e] = 0
dfComptageElement=pd.DataFrame(index=Compteur.keys())


#Search for data in .dM4 in a "Data" folder and open-it using the jamesra library "#https://github.com/jamesra/dm4reader"
fichiers = [f for f in listdir(repData) if isfile(join(repData, f))]
print(fichiers)
for fic in fichiers:
    dm4data = dm4reader.DM4File.open(repData+fic)
    tags = dm4data.read_directory()
    image_data_tag = tags.named_subdirs['ImageList'].unnamed_subdirs[1].named_subdirs['ImageData']
    image_tag = image_data_tag.named_tags['Data']
    XDim = dm4data.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[0])
    YDim = dm4data.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[1])
    ZDim = dm4data.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[2])
    np_array = np.array(dm4data.read_tag_data(image_tag), dtype=np.uint16)
    Cube = np.reshape(np_array, (XDim,YDim, ZDim), order='F')
    Dim=np_array.shape
    Image=Cube[:,:,1021].T
        Present=[]
    #Definition of the energies measured with EELS
    Energie=np.linspace(150,2074,num=Cube.shape[2])


    #Per pixel and for each elemental threshold ('Carbon', 'Oxygen', 'Iron', 'Aluminium', 'Silicium'):
    #-fit of the background
    #-recall of the "Ib_Ik_computation" function,
    #-and attribute an elementary detection according to the threshold "SNR_detection"
    for ligne in range(Image.shape[1]): #collection of the column pixel
        for colonne in range(Image.shape[0]): #collection of the line pixel
            print("\n pixel :(", line, ",", column, ")")

            Spectre = Cube[line, column, :] #data formatting
            dfSpectre = pd.DataFrame(Spectre, columns=['Spectre'], index=Energie)
            Present=[]

            for elt in ParameterElement: #for a definite elemental threshold
                print("element traité : ", elt)
                dfSpectreElt=dfSpectre.loc[ParameterElement[elt]['fit'][0]:
                                                  ParameterElement[elt]['aire'][1]]
                dfSpectreFit=dfSpectre.loc[ParameterElement[elt]['fit'][0]:
                                                  ParameterElement[elt]['fit'][1]]
                idx = np.asarray(dfSpectreFit.index.tolist())
                val= np.asarray(dfSpectreFit.values.tolist())
                val=val[:,0]

                #Background fit using a power law function
                def func(x, a, r):
                    return a * x**(-r )
                popt, pcov = curve_fit(f=func,p0=[0,0],xdata=idx,ydata=val, maxfev=1000000, ftol=1e-10)
                idx2 = np.asarray(dfSpectreElt.index.tolist())
                val2 = np.asarray(dfSpectreElt.values.tolist())
                Model=func(idx2, *popt)
                Model=Model[:,np.newaxis]
                Signal=val2-Model

                #Recall the function 'Ib_Ik_computation' which calculates the Ib and Ik areas and the SNR
                #This function also calculates if the SNR threshold is higher than the detection threshold as defined for each element in the parameterList ""SNR_detection"
                #Dectection returns to the final decision of an elemental detection (e.g. truth or false)
                detection = Ib_Ik_computation(popt, pcov, ParameterElement, idx2, val2, Model,ligne,colonne,elt)
                print("Detection :", detection)
                if(detection):
                    ParameterElement[elt]['counter']=ParameterElement[elt]['counter']+1
                    Present.append(elt)
                    Counter[elt] = Counter[elt] + 1
            print(Present)
            CombinaisonPresent = ''.join(Present)
            print(CombinaisonPresent)
            l=str(CombinaisonPresent)
            try :
                if (elt !=l):
                    Counter[l]=Counter[l]+1
            except:
                print('no elemental detection')
            print(Counter)

#compilation of counters for each element and pixel
print(ParameterElement)
for cle, valeur in Compteur.items():
    #print(cle, valeur)
    dfComptageElement.loc[cle, fic] = valeur

#formatting in .cvs format
dfComptageElement.to_csv('compteur.csv')
