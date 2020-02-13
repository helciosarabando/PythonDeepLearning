# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:36:57 2020

@author: hsarabando & lgodói

Arquivo contendo alguns métodos "utilitários", como imprimir o tempo
transcorrido entre dois pontos do código fonte...

"""

def printtime(tempo):
    hora = tempo/3600
    minuto = tempo/60    
    segundo = tempo%60
    ms = segundo - int(segundo)
    if int(hora) != 0:
        prhora = str(int(hora)) + ' hora(s)'
    else:
        prhora = ''
    if int(minuto) != 0:
        prminuto = str(int(minuto)) + ' minuto(s),'
    else:
        prminuto = ''
    if int(segundo) != 0:
        prsegundo = str(int(segundo)) + ' segundo(s)'
    else:
        prsegundo = ''
    print ('\nTempo de execução:',
           prhora, 
           prminuto,
           prsegundo,
           'e', int((ms*1000)), 'milissegundo(s)')    
    return

# if __name__ == "__main__":
#     tempo = 1246.93
#     printtime(tempo)