#import mahotas as mh
import numpy as np
import cv2
import PIL
import math
import argparse
import imutils


from picamera.array import PiRGBArray
from picamera import PiCamera
import time

"""
    | eixoY
    |
    |
    |
    |
    |               eixoX
  ----------------------
    |
    
"""





def busca_cruzamento(eixoY):
    eixoX=1
    saida=0
    eixoX2=1
    img=0
    while(eixoX<comprimento):
        if saida==0:
            (b)=thresholding[eixoY,eixoX]
            if b==255:
                eixoX2=eixoX
                while(eixoX2<comprimento):
                    (b1)=thresholding[eixoY,eixoX2]
                    if b1==255:
                        eixoX2=eixoX2+1
                        img=img+1
                        saida=1
                    if b1==0:
                        saida=0
                        return 0
                        break
                if eixoX2==comprimento:
                    break
            eixoX=eixoX+1
    if saida==1 and img>comprimento:
        return 1
    if saida==0:
        return 0


def busca_direita(eixoY):
    eixoX=1
    saida=0
    eixoX2=1
    while(eixoX<comprimento):
        if saida==0:
            (b)=thresholding[eixoY,eixoX]
            if b==255:
                eixoX2=eixoX
                while(eixoX2<comprimento):
                    (b1)=thresholding[eixoY,eixoX2]
                    if b1==255:
                        saida=1
                        #cv2.line(frame,(a0,eixoY),(eixoX2,eixoY),(0,0,255),5)
                        break
                    eixoX2=eixoX2+1
            eixoX=eixoX+1
        if saida==1:
            return (eixoX2,eixoY,1)
            break
    if saida==0:
        return(1000,1000,0)
    
def busca_esquerda(eixoY):
    eixoX=comprimento-1
    saida=0
    eixoX2=comprimento-1
    while(eixoX>0):
        if saida==0:
            (b)=thresholding[eixoY,eixoX]
            if b==255 :
                eixoX2=eixoX
                while(eixoX2>0):
                    (b1)=thresholding[eixoY,eixoX2]
                    if b1==255 :
                        saida=1
                        #cv2.line(frame,(a0,eixoY),(img1,eixoY),(0,0,255),5)
                        break
                    eixoX2=eixoX2-1
            eixoX=eixoX-1
        if saida==1:
            return (eixoX2,eixoY,1)
            break
    if saida==0:
        return (1000,1000,0)


#############################
def busca_direita_esquerda_teste(eixoY):
    eixoX=comprimento
    saida=0
    eixoX2=comprimento
    contador=1
    estadoArmazenado=0
    while(eixoX>0):
        (b)=thresholding[eixoY,eixoX]
        if b==255:
            if estadoArmazenado==0:
                eixoX2=eixoX
            contador=contador+1
            eixoX=eixoX-1
            estadoArmazenado=1
        else:
            eixoX=eixoX-1
            contador=0
            estadoArmazenado=0
        if(contador>65):
            saida=1
            cv2.line(frame,(eixoX2,eixoY),(eixoX2+1,eixoY),(0,0,255),5)
            return (eixoX2,eixoY,1)
            break
            
    if saida==1:
        return (eixoX2,eixoY,1)
    if saida==0:
        return (1000,1000,0)

#############################
#############################
def busca_esquerda_direita_teste(eixoY):
    eixoX=1
    saida=0
    eixoX2=1
    contador=1
    estadoArmazenado=0
    while(eixoX<comprimento):
        (b)=thresholding[eixoY,eixoX]
        if b==255:
            if estadoArmazenado==0:
                eixoX2=eixoX
            contador=contador+1
            eixoX=eixoX+1
            estadoArmazenado=1
        else:
            eixoX=eixoX+1
            contador=0
            estadoArmazenado=0
        if(contador>65):
            saida=1
            cv2.line(frame,(eixoX2,eixoY),(eixoX2+1,eixoY),(255,0,0),5)
            return (eixoX2,eixoY,1)
            break
            
    if saida==1:
        return (eixoX2,eixoY,1)
    if saida==0:
        return (1000,1000,0)

#############################
    
def busca_centro(eixoY):
    bola0=busca_esquerda_direita_teste(eixoY)
    bola1=busca_direita_esquerda_teste(eixoY)
    if bola0[2]==1 or bola1[2]==1:
        #cv2.line(frame,(eixoX+2,eixoY),(eixoX2-2,eixoY),(255,0,0),2)
        #cordenadas das duas extremidades da linha preta
        dx2 = (bola0[0]-bola1[0])**2
        dy2 = (bola0[1]-bola1[1])**2 
        distanceFloat=math.sqrt(dx2 + dy2)
        distanceInt=int((distanceFloat)/2)
        #a cordenada da esquerda acrescida a metade da distancia entre as duas
        #extremidades da linha preta, logo, resulta no centro da linha
        centro=bola0[0]+distanceInt
        cv2.circle(frame,(centro,eixoY),5,(120,50,120),-1)
        return (centro,eixoY,bola0[1],eixoY)
    if bola0[2]==0 or bola1[2]==0:
        return (1000,1000,1000,1000)

def distance_roxo_centro(bola,centro):
    retornoLado=0
    #condição inibidora
    if(bola[0]!=1000 and bola[1]!=1000 and centro[0]!=1000 and centro[1]!=1000):
        #distancia do centro a bola
        dx2 = (bola[0]-centro[0])**2
        dy2 = (bola[1]-centro[1])**2
        #direita
        if meio<bola[0]:
            zi=int((math.sqrt(dx2 + dy2))*-1)
            cv2.putText(frame,'>>',(meio,centro[1]+40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,100,255),2,cv2.LINE_AA)
            cv2.putText(frame,str(zi),(meio,centro[1]+23),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
        #esquerda
        else:
            zi=int(math.sqrt(dx2 + dy2))
            cv2.putText(frame,'<<',(meio,centro[1]+40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,0,255),2,cv2.LINE_AA)
            cv2.putText(frame,str(zi),(meio,centro[1]+23),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
        return (zi)
    else:
        cv2.putText(frame,'0',(meio,centro[1]+40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(55,100,55),2,cv2.LINE_AA)
        return (1000)

def busca_branco_direita(bola):
    eixoX=bola[0]+3
    saida=0
    img1=bola[0]+3
    eixoY=bola[1]
    while(eixoX<comprimento):
        if saida==0:
            eixoX2=eixoX
            while(eixoX2<comprimento):
                (b1,g1,r1)=frame[eixoY,eixoX2]
                if b1==255 and g1==255 and r1==255:
                    saida=1
                    #print('Branco',eixoY,'yes')
                    break
                eixoX2=eixoX2+1
        eixoX=eixoX+1
        if saida==1:
            return 1
            break
        else:
            return 0
            break
    
def busca_branco_esquerda(bola):
    eixoX=bola[0]-3
    saida=0
    eixoX2=bola[0]-3
    eixoY=bola[1]
    while(eixoX<comprimento):
        if saida==0:
            eixoX2=eixoX
            while(eixoX2>0):
                (b1,g1,r1)=frame[eixoY,eixoX2]
                if b1==255 and g1==255 and r1==255:
                    saida=1
                    #print('Branco',eixoY,'yes')
                    break
                eixoX2=eixoX2-1
        eixoX=eixoX-1
        if saida==1:
            return 1
            break
        else:
            return 0
            break

#biblioteca da lei dos cossenos, calcula o angulo do triangulo, utilizando as
#cordenadas já definidas 
def leiCos(bola3,bola4,bola5):

    #calculo das 3 distancias, 3 lados do triangulo
    dx2 = (bola4[0]-bola5[0])**2
    dy2 = (bola4[1]-bola5[1])**2 
    zi=int(math.sqrt(dx2 + dy2))
    distance45 = str(zi)
    
    dx2 = (bola4[0]-bola3[0])**2
    dy2 = (bola4[1]-bola3[1])**2 
    xi=int((math.sqrt(dx2 + dy2)))
    distancever = str(xi)
    
    dx2 = (bola5[0]-bola3[0])**2
    dy2 = (bola5[1]-bola3[1])**2 
    yi=int(math.sqrt(dx2 + dy2))
    distanceponta= str(yi)

    #calculo dos 3 angulos do triangulo
    if zi!=0 and xi!=0 and yi!=0:
        cosi=(((yi**2)-(zi**2)-(xi**2))/(2*xi*zi))*-1
        cof = np.cos(cosi)
        coffe = np.arccos(cof)
        coffe=cosi*coffe
        a=int(180+(coffe*180))

    if zi!=0 and xi!=0 and yi!=0:
        cosi=(((zi**2)-(yi**2)-(xi**2))/(2*xi*yi))*-1
        cof = np.cos(cosi)
        coffe = np.arccos(cof)
        coffe=cosi*coffe
        b=int(180-(coffe*180))

    if zi!=0 and xi!=0 and yi!=0:
        cosi=(((xi**2)-(yi**2)-(zi**2))/(2*zi*yi))*-1
        cof = np.cos(cosi)
        coffe = np.arccos(cof)
        coffe=cosi*coffe
        c=int(180-(coffe*180))
        
    #lugar onde é exibido o texto
    local=(parte2,30)
    if a>90 and b>30 and busca_branco_direita(bola4)==1:
        cv2.putText(frame,'curva de 90 direita',local,cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,100,255),3,cv2.LINE_AA)

    elif a>90 and b>30 and busca_branco_esquerda(bola4)==1:
        cv2.putText(frame,'curva de 90 esquerda',local,cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,0,255),3,cv2.LINE_AA)

    elif a>10 and busca_branco_esquerda(bola4)==1:
        cv2.putText(frame,'leve curva a esquerda',local,cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,0,255),2,cv2.LINE_AA)
    elif a>4 and busca_branco_direita(bola4)==1:
        cv2.putText(frame,'leve curva a direita',local,cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,100,255),2,cv2.LINE_AA)
    else:
        cv2.putText(frame,'reta',local,cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,100,0),2,cv2.LINE_AA)


#variáveis pré definidas
altura=479
comprimento=639
bola5=bola4=bola3=bola2=bola1=(0,90)
roxo=[0,0,0,0,0]
partex=int(altura/6)
x=35
parte0=int(partex*1)+x
parte1=int(partex*2)+x
parte2=int(partex*3)+x
parte3=int(partex*4)+x
parte4=int(partex*5)+x
meio=320
isto=(0,0)
somatorioErro=0

#configurações da camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
camera.sharpness=0
camera.contrast=40
camera.brightness=60
camera.saturation=0
camera.ISO=100
camera.exposure_compensation=0
camera.exposure_mode='auto'
camera.meter_mode='average'
camera.awb_mode='tungsten'
camera.image_effect='none'
camera.color_effects= None
camera.rotation= 10
camera.hflip=False
camera.vflip=False
camera.crop=(0.0,0.0,1.0,1.0)

rawCapture = PiRGBArray(camera, size = (640, 480))
time.sleep(0.1)

for frame0 in camera.capture_continuous(rawCapture, format = 'bgr',
                                       use_video_port = True):

    # Captura frame por frame
    frame2 = frame0.array
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    
    #filtros e tratamento da imagem
    frame=cv2.GaussianBlur(frame2,(5,5),0)
    frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #ret,thresholding=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    #thresholding=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                                   cv2.THRESH_BINARY_INV,3,1)
    ret,thresholding = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    thresholding = cv2.erode( thresholding, kernel, iterations=2)
    thresholding = cv2.dilate( thresholding, kernel, iterations=1)
    im2,contours,hierarchy = cv2.findContours(thresholding,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )


                ##############################################
                ###############FUNÇÃO DAS BOXES###############
                ##############################################
    
    #Cria lista/vetor conforme o num de boxes
    #Acha a maior boxe e elimina todas as outras boxes menores
    area=[None]*len(contours)
    vetor=[None]*len(contours)
    num=0
    while (num<len(contours)):
        cnt = contours[num]
        a = cv2.contourArea(cnt)
        if a>10:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #cv2.drawContours(frame,[box],0,(255,0,0),1)
            area[num]=(a)
            vetor[num]=([box])
        num=num+1

    #print(area.count(None))
    while int(area.count(None))!=0 :
        area.remove(None)
        vetor.remove(None)

    if(len(contours)>1):
        lek=area.index(max(area))
        area.remove(max(area))
        del(vetor[lek])
        #print("0",area,"0")
        #print([box])
        num=0
        while (num<len(area)):
            cv2.drawContours(thresholding,vetor[num],0,(0,0,0),-1)
            #cv2.drawContours(frame,vetor[num],0,(255,255,255),-1)
            num=num+1

                    #################FIM#################
                
    #thresholding = cv2.erode( thresholding, kernel, iterations=1)
    

    #busca centro da imagem e traça linha horizontal
    bola5=busca_centro(parte0)
    bolav1=(bola5[2],bola5[3])
    bola4=busca_centro(parte1)
    bolav2=(bola4[2],bola4[3])
    bola3=busca_centro(parte2)
    bola2=busca_centro(parte3)
    bola1=busca_centro(parte4)
    
    #testa as cordenadas válidas, e vai mudando o triangulo de posição
    if (bola3[0]!=1000 and bola4[0]!=1000 and bola5[0]!=1000):
        cv2.line(frame,(bola3[0],bola3[1]),(bola5[0],bola5[1]),(255,255,255),2)
        leiCos((bola3[0],bola3[1]),(bola4[0],bola4[1]),(bola5[0],bola5[1]))
        cv2.line(frame,(bola4[0],bola4[1]),(bola5[0],bola5[1]),(255,255,255),2)
        cv2.line(frame,(bola3[0],bola3[1]),(bola4[0],bola4[1]),(255,255,255),2)
        isto=bola5
        
    elif (bola2[0]!=1000 and bola3[0]!=1000 and bola4[0]!=1000):
        cv2.line(frame,(bola2[0],bola2[1]),(bola4[0],bola4[1]),(255,255,255),2)
        leiCos((bola2[0],bola2[1]),(bola3[0],bola3[1]),(bola4[0],bola4[1]))
        cv2.line(frame,(bola2[0],bola2[1]),(bola3[0],bola3[1]),(255,255,255),2)
        cv2.line(frame,(bola3[0],bola3[1]),(bola4[0],bola4[1]),(255,255,255),2)
        isto=bola4

    elif (bola1[0]!=1000 and bola2[0]!=1000 and bola3[0]!=1000):
        cv2.line(frame,(bola1[0],bola1[1]),(bola3[0],bola3[1]),(255,255,255),2)
        leiCos((bola1[0],bola1[1]),(bola2[0],bola2[1]),(bola3[0],bola3[1]))
        cv2.line(frame,(bola1[0],bola1[1]),(bola2[0],bola2[1]),(255,255,255),2)
        cv2.line(frame,(bola2[0],bola2[1]),(bola3[0],bola3[1]),(255,255,255),2)
        isto=bola3

    #quando todas as cordenadas são invalidas    
    elif (bola1[0]==1000 and bola2[0]==1000 and bola3[0]==1000 and bola4[0]==1000 and bola5[0]==1000):
        cv2.putText(frame,"gap",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1,cv2.LINE_AA)

    elif bola2[0]!=1000:
        isto=bola2
    elif bola1[0]!=1000:
        isto=bola1
    

            #######################################################
            ##################CALCULO DO CONTROLE##################
            #######################################################

    #distancia do roxo da linha e do roxo central
    roxo[0]=distance_roxo_centro(bola5,(meio,parte0))
    roxo[1]=distance_roxo_centro(bola4,(meio,parte1))
    roxo[2]=distance_roxo_centro(bola3,(meio,parte2))
    roxo[3]=distance_roxo_centro(bola2,(meio,parte3))
    roxo[4]=distance_roxo_centro(bola1,(meio,parte4))

    
    #conta as bolas que estão para a direita e para esquerda do centro da img
    """i=0
    esquerda=0
    direita=0
    while i<5:
        if roxo[i]==2:
           direita=direita+1 
        elif roxo[i]==1:
            esquerda=esquerda+1
        i=i+1
    if direita>esquerda:
        cv2.putText(frame,'direita',(400,parte4),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,100,255),2,cv2.LINE_AA)
    else:
        cv2.putText(frame,'esquerda',(400,parte4),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,0,255),2,cv2.LINE_AA)
    """
    passa=1
    if(roxo[3]!=0 and roxo[3]!=1000):
        erro=roxo[3]
    elif(roxo[2]!=0 and roxo[2]!=1000):
        erro=roxo[2]
    elif(roxo[4]!=0 and roxo[4]!=1000):
        erro=roxo[4]
    elif(roxo[1]!=0 and roxo[1]!=1000):
        erro=roxo[1]
    elif(roxo[0]!=0 and roxo[0]!=1000):
        erro=roxo[0]
    else:
        erro=1
        
    kp=2.15
    proporcional=kp*erro
    pwm=int((255*(proporcional+(320*kp)))/(640*kp))
    #cv2.putText(frame,str(pwm),(50,100),cv2.FONT_HERSHEY_SIMPLEX,\
     #           0.7,(100,0,255),2,cv2.LINE_AA)

    ti=1/(5)
    erroAnterior=erro
    somatorioErro=somatorioErro+erro
    if somatorioErro>2000:
        somatorioErro=2000
    if somatorioErro<-2000:
        somatorioErro=-2000
    integral=int((kp*erro)+(kp*(1/ti)*somatorioErro)+127)
    pwm=int((255*(integral+6627)/(13254)))
    if(pwm>255):
        pwm=255
    if(pwm<-255):
        pwm=-255
    
    cv2.putText(frame,str(pwm),(50,100),cv2.FONT_HERSHEY_SIMPLEX,\
                0.7,(100,0,255),2,cv2.LINE_AA)

                    #################FIM#################

    '''
    #circulos centrais
    cv2.circle(frame,(meio,parte0),2,(120,50,120),-1)
    cv2.circle(frame,(meio,parte1),2,(120,50,120),-1)
    cv2.circle(frame,(meio,parte2),2,(120,50,120),-1)
    cv2.circle(frame,(meio,parte3),2,(120,50,120),-1)
    cv2.circle(frame,(meio,parte4),2,(120,50,120),-1)
    '''
                #############################################
                ###############FUNÇÃO DO VERDE###############
                #############################################
    
    #define a mascara a ser considerada para o verde
    lower=np.array([40,60,50])
    upper=np.array([80,255,255])  
    mask=cv2.inRange(frame2,lower,upper)
    mask2 = cv2.erode( mask, kernel, iterations=1)
    _,contours,_=cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    NumeroContornos = str(len(contours))
    #cv2.drawContours(frame,contours,-1,(0,0,255),2)
    if int(NumeroContornos)>0:
        #mascara verde1
        mascara=np.zeros(frame.shape[:2],dtype="uint8")
        (cX,cY)=(frame.shape[1] // 2, frame.shape[0] // 2)
        cv2.circle(mascara,(isto[0]-55,isto[1]-5),20,255,-1)
        frame_mascara= cv2.bitwise_and(frame2,frame2,mask=mascara)
        
        cv2.circle(frame,(isto[0]-55,isto[1]-5),20,255,1)
        mask=cv2.inRange(frame_mascara,lower,upper)
        _,contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        NumeroPXSBrancos = int(cv2.countNonZero(mask))
        #print(NumeroPXSBrancos)
        if int(NumeroPXSBrancos)>500:
            cv2.putText(frame,"verde lado esquerdo",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3,cv2.LINE_AA)
            esquerda=1
            direita=0
        #mascara verde2
        mascara=np.zeros(frame.shape[:2],dtype="uint8")
        (cX,cY)=(frame.shape[1] // 2, frame.shape[0] // 2)
        cv2.circle(mascara,(isto[0]+55,isto[1]-5),20,255,-1)
        frame_mascara= cv2.bitwise_and(frame2,frame2,mask=mascara)
        cv2.circle(frame,(isto[0]+55,isto[1]-5),20,255,1)
        
        mask=cv2.inRange(frame_mascara,lower,upper)
        _,contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        NumeroPXSBrancos = int(cv2.countNonZero(mask))
        #print(NumeroPXSBrancos)
        if int(NumeroPXSBrancos)>500:
            cv2.putText(frame,"verde lado direito",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3,cv2.LINE_AA)
            direita=1
            esquerda=0

                    #################FIM#################

    #exibe o resultado
    cv2.imshow('frame',frame)
    cv2.imshow('thresholding',thresholding)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

