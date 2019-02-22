
# coding: utf-8

# In[ ]:
import math
from math import pi
import copy
import cv2
import numpy as np
img = cv2.imread("task2.jpg", 0)
image=list(img)

#Function to form the Gaussian kernel
def GaussianKernel(height,width,sigma):
    total=0
    Gkernel1=[]
    #Creating an empty list to store the Gaussian kernel
    for i in range(height):
        Gkernel1.append([0]*width)
    #Calculating each element value for the given sigma
    for cols,i in zip(range(0,width),range(-3,4)):
        for rows,j in zip(range(0,height),range(3,-4,-1)):
            ival=0
            val=0
            ival=math.exp((-1*((i**2)+(j**2)))/(2*(sigma**2)))
            val=(1/(2*pi*sigma**2))*ival
            Gkernel1[rows][cols]=val
            total=total+val
    Gkernel1=[[x/total for x in lst] for lst in Gkernel1]
    return Gkernel1

#Convolution Function to perform Convolution of the image with the gaussian kernel
def conv_func(image,kernel):
  rows=len(image)
  cols=len(image[0])
  krows=len(kernel)
  kcols=len(kernel[0])
  gx = copy.deepcopy(image)
  # find center position of kernel (half of kernel size)
  kCenterX = kcols // 2
  kCenterY = krows // 2
  for i in range(rows):
    for j in range(cols):
        sum=0
        for m in range(krows):
          mm = krows - 1 - m;
          for n in range(kcols):
            nn = kcols - 1 - n
            ii = i + (kCenterY - mm)
            jj = j + (kCenterX - nn)
            # ignore input samples which are out of bound
            if( ii >= 0 and ii < rows and jj >= 0 and jj < cols ):
              sum += image[ii][jj] * kernel[mm][nn]
        if sum > 255:
          sum = 255
        if sum < 0:
          sum = 0
        gx[i][j]=sum
  return gx

########Forming the first octave#####
gkernel_1Oct=[]  #initialising empty list to hold the different gaussian kernel for 1st octave
BlurImg_1stOct=[] #initialising empty list to hold the different blurred image for 1st octave
gkernel_1Oct.append(GaussianKernel(7,7,0.707))
BlurImg_1stOct.append(conv_func(image, gkernel_1Oct[0]))
Bimage1_1=np.asarray(BlurImg_1stOct[0])
gkernel_1Oct.append(GaussianKernel(7,7,1))
BlurImg_1stOct.append(conv_func(image, gkernel_1Oct[1]))
Bimage1_2=np.asarray(BlurImg_1stOct[1])
gkernel_1Oct.append(GaussianKernel(7,7,1.414))
BlurImg_1stOct.append(conv_func(image,gkernel_1Oct[2]))
Bimage1_3=np.asarray(BlurImg_1stOct[2])
gkernel_1Oct.append(GaussianKernel(7,7,2))
BlurImg_1stOct.append(conv_func(image,gkernel_1Oct[3]))
Bimage1_4=np.asarray(BlurImg_1stOct[3])
gkernel_1Oct.append(GaussianKernel(7,7,2.828))
BlurImg_1stOct.append(conv_func(image,gkernel_1Oct[4]))
Bimage1_5=np.asarray(BlurImg_1stOct[4])

########Forming the 2nd octave#####
#Scaling the image by half
image2col=[row[0::2] for row in image]
image2=image2col[0::2]
image2_s=np.asarray(image2)
cv2.imwrite('2nd_Octave_Scaled_Image.png',image2_s)
print('2nd Octave Scaled Image resolution is: ',len(image2),',',len(image2[0]))
gkernel_2Oct=[]  #initialising empty list to hold the different gaussian kernel for 2ns octave
BlurImg_2Oct=[] #initialising empty list to hold the different blurred image for 2nd octave
gkernel_2Oct.append(GaussianKernel(7,7,1.414))
BlurImg_2Oct.append(conv_func(image2,gkernel_2Oct[0]))
Bimage2_1=np.asarray(BlurImg_2Oct[0])

gkernel_2Oct.append(GaussianKernel(7,7,2))
BlurImg_2Oct.append(conv_func(image2,gkernel_2Oct[1]))
Bimage2_2=np.asarray(BlurImg_2Oct[1])

gkernel_2Oct.append(GaussianKernel(7,7,2.828))
BlurImg_2Oct.append(conv_func(image2,gkernel_2Oct[2]))
Bimage2_3=np.asarray(BlurImg_2Oct[2])

gkernel_2Oct.append(GaussianKernel(7,7,4))
BlurImg_2Oct.append(conv_func(image2,gkernel_2Oct[3]))
Bimage2_4=np.asarray(BlurImg_2Oct[3])

gkernel_2Oct.append(GaussianKernel(7,7,5.656))
BlurImg_2Oct.append(conv_func(image2,gkernel_2Oct[4]))
Bimage2_5=np.asarray(BlurImg_2Oct[4])



########Forming the 3rd octave#####
#Scaling the image by half
image3col=[row[0::2] for row in image2]
image3=image3col[0::2]
image3_s=np.asarray(image3)
cv2.imwrite('3rd_Octave_Scaled_Image.png',image3_s)
print('3rd Octave Scaled Image resolution is: ',len(image3),',',len(image3[0]))
gkernel_3Oct=[]  #initialising empty list to hold the different gaussian kernel for 2ns octave
BlurImg_3Oct=[] #initialising empty list to hold the different blurred image for 2nd octave
gkernel_3Oct.append(GaussianKernel(7,7,2.828))
BlurImg_3Oct.append(conv_func(image3,gkernel_3Oct[0]))
Bimage3_1=np.asarray(BlurImg_3Oct[0])

gkernel_3Oct.append(GaussianKernel(7,7,4))
BlurImg_3Oct.append(conv_func(image3,gkernel_3Oct[1]))
Bimage3_2=np.asarray(BlurImg_3Oct[1])

gkernel_3Oct.append(GaussianKernel(7,7,5.656))
BlurImg_3Oct.append(conv_func(image3,gkernel_3Oct[2]))
Bimage3_3=np.asarray(BlurImg_3Oct[2])

gkernel_3Oct.append(GaussianKernel(7,7,8))
BlurImg_3Oct.append(conv_func(image3,gkernel_3Oct[3]))
Bimage3_4=np.asarray(BlurImg_3Oct[3])

gkernel_3Oct.append(GaussianKernel(7,7,11.313))
BlurImg_3Oct.append(conv_func(image3,gkernel_3Oct[4]))
Bimage3_5=np.asarray(BlurImg_3Oct[4])


########Forming the 4th octave#####
#Scaling the image by half
image4col=[row[0::2] for row in image3]
image4=image4col[0::2]
gkernel_4Oct=[]  #initialising empty list to hold the different gaussian kernel for 2ns octave
BlurImg_4Oct=[] #initialising empty list to hold the different blurred image for 2nd octave
gkernel_4Oct.append(GaussianKernel(7,7,5.656))
BlurImg_4Oct.append(conv_func(image4,gkernel_4Oct[0]))
Bimage4_1=np.asarray(BlurImg_4Oct[0])
gkernel_4Oct.append(GaussianKernel(7,7,8))
BlurImg_4Oct.append(conv_func(image4,gkernel_4Oct[1]))
Bimage4_2=np.asarray(BlurImg_4Oct[1])
gkernel_4Oct.append(GaussianKernel(7,7,11.313))
BlurImg_4Oct.append(conv_func(image4,gkernel_4Oct[2]))
Bimage4_3=np.asarray(BlurImg_4Oct[2])
gkernel_4Oct.append(GaussianKernel(7,7,16))
BlurImg_4Oct.append(conv_func(image4,gkernel_4Oct[3]))
Bimage4_4=np.asarray(BlurImg_4Oct[3])
gkernel_4Oct.append(GaussianKernel(7,7,22.627))
BlurImg_4Oct.append(conv_func(image4,gkernel_4Oct[4]))
Bimage4_5=np.asarray(BlurImg_4Oct[4])

######Creating DoGs for first octave#####

DoG1_1=Bimage1_1 - Bimage1_2
DoG1_2=Bimage1_2 - Bimage1_3
DoG1_3=Bimage1_3 - Bimage1_4
DoG1_4=Bimage1_4 - Bimage1_5

DoG2_1=Bimage2_1 - Bimage2_2
DoG2_2=Bimage2_2 - Bimage2_3
DoG2_3=Bimage2_3 - Bimage2_4
DoG2_4=Bimage2_4 - Bimage2_5
cv2.imwrite('2nd_Octave_1st_DoG.png',DoG2_1)
cv2.imwrite('2nd_Octave_2nd_DoG.png',DoG2_2)
cv2.imwrite('2nd_Octave_3rd_DoG.png',DoG2_3)
cv2.imwrite('2nd_Octave_4th_DoG.png',DoG2_4)

DoG3_1=Bimage3_1 - Bimage3_2
DoG3_2=Bimage3_2 - Bimage3_3
DoG3_3=Bimage3_3 - Bimage3_4
DoG3_4=Bimage3_4 - Bimage3_5
cv2.imwrite('3rd_Octave_1st_DoG.png',DoG3_1)
cv2.imwrite('3rd_Octave_2nd_DoG.png',DoG3_2)
cv2.imwrite('3rd_Octave_3rd_DoG.png',DoG3_3)
cv2.imwrite('3rd_Octave_4th_DoG.png',DoG3_4)

DoG4_1=Bimage4_1 - Bimage4_2
DoG4_2=Bimage4_2 - Bimage4_3
DoG4_3=Bimage4_3 - Bimage4_4
DoG4_4=Bimage4_4 - Bimage4_5

#####Finding points of maxima and minima in each Octave
DoG1_1_lst=list(DoG1_1)
DoG1_2_lst=list(DoG1_2)
DoG1_3_lst=list(DoG1_3)
DoG1_4_lst=list(DoG1_4)
DoG1_11=[]
DoG1_11.append(DoG1_1_lst)
DoG1_11.append(DoG1_2_lst)
DoG1_11.append(DoG1_3_lst)
DoG1_11.append(DoG1_4_lst)
#finding points of minima/maxima for 1st octave
for i in range(1,len(DoG1_11[0])-1):
    for j in range(1,len(DoG1_11[0][0])-1):
        for k in range(0,len(DoG1_11)-1):
            if(DoG1_11[1][i][j]>=DoG1_11[k][i][j] or DoG1_11[1][i][j]<DoG1_11[k][i][j]):
                DoG1_11[1][i][k]=255

for i in range(1,len(DoG1_11[2])-1):
    for j in range(1,len(DoG1_11[2][0])-1):
        for k in range(1,len(DoG1_11)):
            if(DoG1_11[2][i][j]>=DoG1_11[k][i][j] or DoG1_11[2][i][j]<DoG1_11[k][i][j]):
                DoG1_11[2][i][k]=255


#2nd octave
DoG2_1_lst=list(DoG2_1)
DoG2_2_lst=list(DoG2_2)
DoG2_3_lst=list(DoG2_3)
DoG2_4_lst=list(DoG2_4)
DoG2_11=[]
DoG2_11.append(DoG2_1_lst)
DoG2_11.append(DoG2_2_lst)
DoG2_11.append(DoG2_3_lst)
DoG2_11.append(DoG2_4_lst)
#finding points of minima/maxima for 2nd octave
for i in range(1,len(DoG2_11[0])-1):
    for j in range(1,len(DoG2_11[0][0])-1):
        for k in range(0,len(DoG2_11)-1):
            if(DoG2_11[1][i][j]>=DoG2_11[k][i][j] or DoG2_11[1][i][j]<DoG2_11[k][i][j]):
                DoG2_11[1][i][k]=255

for i in range(1,len(DoG2_11[2])-1):
    for j in range(1,len(DoG2_11[2][0])-1):
        for k in range(1,len(DoG2_11)):
            if(DoG2_11[2][i][j]>=DoG2_11[k][i][j] or DoG2_11[2][i][j]<DoG2_11[k][i][j]):
                DoG2_11[2][i][k]=255

for i in range(len(DoG2_11[1])):
    for j in range(len(DoG2_11[1][0])):
        if(DoG2_11[1][i][j]<255):
            DoG2_11[1][i][j]=0
        if(DoG2_11[1][i][j]==255):
            image[i*2][j*2]=255
image_detected=np.asarray(image)
#cv2.imshow('first detection',image_detected)
cv2.imwrite('1st_image_showing_detected_Keypoints_in_white_from_second_octave.png',image_detected)
###Co-ordinates of the five lefmost detected points
cntr=0
for i in range(len(image)//2):
    for j in range(len(image[0]//2)):
        if (image[i][j]==255):
            cntr=cntr+1
            print('The five leftmost detected points in second octave 1st image are: ',i,',',j)
        if(cntr==5):
            break
    else:
        continue  # only executed if the inner loop did NOT break
    break  # only executed if the inner loop DID break

img = cv2.imread("task2.jpg", 0)
image=list(img)
for i in range(len(DoG2_11[2])):
    for j in range(len(DoG2_11[2][0])):
        if(DoG2_11[2][i][j]<255):
            DoG2_11[2][i][j]=0
        if(DoG2_11[2][i][j]==255):
            image[i*2][j*2]=255
image_detected=np.asarray(image)
cv2.imwrite('2nd_image_showing_detected_Keypoints_in_white_from_second_octave.png',image_detected)
cntr=0
for i in range(len(image)//2):
    for j in range(len(image[0]//2)):
        if (image[i][j]==255):
            cntr=cntr+1
            print('The five leftmost detected points in second octave 2nd image are: ',i,',',j)
        if(cntr==5):
            break
    else:
        continue  # only executed if the inner loop did NOT break
    break  # only executed if the inner loop DID break
#3rd octave
DoG3_1_lst=list(DoG3_1)
DoG3_2_lst=list(DoG3_2)
DoG3_3_lst=list(DoG3_3)
DoG3_4_lst=list(DoG3_4)
DoG3_11=[]
DoG3_11.append(DoG3_1_lst)
DoG3_11.append(DoG3_2_lst)
DoG3_11.append(DoG3_3_lst)
DoG3_11.append(DoG3_4_lst)
#finding points of minima/maxima for 3rd octave
for i in range(1,len(DoG3_11[0])-1):
    for j in range(1,len(DoG3_11[0][0])-1):
        for k in range(0,len(DoG3_11)-1):
            if(DoG3_11[1][i][j]>=DoG3_11[k][i][j] or DoG3_11[1][i][j]<DoG3_11[k][i][j]):
                DoG3_11[1][i][k]=255

for i in range(1,len(DoG3_11[2])-1):
    for j in range(1,len(DoG3_11[2][0])-1):
        for k in range(1,len(DoG3_11)):
            if(DoG3_11[2][i][j]>=DoG3_11[k][i][j] or DoG3_11[2][i][j]<DoG3_11[k][i][j]):
                DoG3_11[2][i][k]=255

img = cv2.imread("task2.jpg", 0)
image=list(img)
for i in range(len(DoG3_11[1])):
    for j in range(len(DoG3_11[1][0])):
        if(DoG3_11[1][i][j]<255):
            DoG3_11[1][i][j]=0
        if(DoG3_11[1][i][j]==255):
            image[i*4][j*4]=255
image_detected=np.asarray(image)
cv2.imwrite('1st_image_showing_detected_Keypoints_in_white_from_third_octave.png',image_detected)
cntr=0
for i in range(len(image)//2):
    for j in range(len(image[0]//2)):
        if (image[i][j]==255):
            cntr=cntr+1
            print('The five leftmost detected points in 3rd octave 1st image are: ',i,',',j )
        if(cntr==5):
            break
    else:
        continue  # only executed if the inner loop did NOT break
    break  # only executed if the inner loop DID break


img = cv2.imread("task2.jpg", 0)
image=list(img)
for i in range(len(DoG3_11[2])):
    for j in range(len(DoG3_11[2][0])):
        if(DoG3_11[2][i][j]<255):
            DoG3_11[2][i][j]=0
        if(DoG3_11[2][i][j]==255):
            image[i*4][j*4]=255
image_detected=np.asarray(image)
cv2.imwrite('2nd_image_showing_detected_Keypoints_in_white_from_third_octave.png',image_detected)
cntr=0
for i in range(len(image)//2):
    for j in range(len(image[0]//2)):
        if (image[i][j]==255):
            cntr=cntr+1
            print('The five leftmost detected points in 3rd octave 2nd image are: ',i,',',j )
        if(cntr==5):
            break
    else:
        continue  # only executed if the inner loop did NOT break
    break  # only executed if the inner loop DID break
#4th octave
DoG4_1_lst=list(DoG4_1)
DoG4_2_lst=list(DoG4_2)
DoG4_3_lst=list(DoG4_3)
DoG4_4_lst=list(DoG4_4)
DoG4_11=[]
DoG4_11.append(DoG4_1_lst)
DoG4_11.append(DoG4_2_lst)
DoG4_11.append(DoG4_3_lst)
DoG4_11.append(DoG4_4_lst)
#finding points of minima/maxima for 1st octave
for i in range(1,len(DoG4_11[0])-1):
    for j in range(1,len(DoG4_11[0][0])-1):
        for k in range(0,len(DoG4_11)-1):
            if(DoG4_11[1][i][j]>=DoG4_11[k][i][j] or DoG4_11[1][i][j]<DoG4_11[k][i][j]):
                DoG4_11[1][i][k]=255

for i in range(1,len(DoG4_11[2])-1):
    for j in range(1,len(DoG4_11[2][0])-1):
        for k in range(1,len(DoG4_11)):
            if(DoG4_11[2][i][j]>=DoG4_11[k][i][j] or DoG4_11[2][i][j]<DoG4_11[k][i][j]):
                DoG4_11[2][i][k]=255

img = cv2.imread("task2.jpg", 0)
image=list(img)
for i in range(len(DoG4_11[1])):
    for j in range(len(DoG4_11[1][0])):
        if(DoG4_11[1][i][j]<255):
            DoG4_11[1][i][j]=0
        if(DoG4_11[1][i][j]==255):
            image[i*8][j*8]=255

img = cv2.imread("task2.jpg", 0)
image=list(img)
for i in range(len(DoG4_11[2])):
    for j in range(len(DoG4_11[2][0])):
        if(DoG4_11[2][i][j]<255):
            DoG4_11[2][i][j]=0
        if(DoG4_11[2][i][j]==255):
            image[i*8][j*8]=255


cv2.waitKey(0)
cv2.destroyAllWindows()
