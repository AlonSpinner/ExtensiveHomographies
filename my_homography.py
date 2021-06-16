import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import matplotlib
from matplotlib import pyplot as plt
from skimage import io
from skimage import color

#Add imports if needed:
#allows to use ginput in jupyter:
#https://stackoverflow.com/questions/41403406/matplotlib-use-of-ginput-on-jupyter-matplotlib-notebook

from PIL import Image
import scipy
from scipy import interpolate
from scipy import sparse as spa
import scipy.ndimage
from scipy.sparse import linalg

import os
import glob
import sys
import pysift
#end imports

#Add extra functions here:
def draw_ginput_points(im,p):
    '''
    Inputs:
    im is an np array image
    p is an np array mx2 
    
    draws points on the image
    '''
    if not isinstance(im,np.ndarray):
        im=np.array(im)
    else:
        im=np.copy(im)
        
    colors = matplotlib.cm.viridis(np.linspace(0, 1, p.shape[1]))
                                 
    p=p.astype(np.float32) #cant handle np.float64
    for i in range(p.shape[1]):
        im = cv2.circle(im, (p[0,i],p[1,i]), radius=10, color=255*colors[i], thickness=2)
    
    plt.figure(figsize=(16,5))
    plt.imshow(im)
    
def homogenize_coordinates(p):
    #accepts mx2 and returns mx3 where last column is ones
    OnesVec=np.ones((1,p.shape[1]))
    homog_p=np.vstack((p,OnesVec))
    return homog_p
def hetrogenize_coordinates(homog_p):
    #accepts mx3, divides each row by its third element, and returns matrix of first two colunmns
    
    #np.newaxis adds a dimension to the numpy array.
    #https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
    homog_p=homog_p/homog_p[2,:] #divides by third element
    hetro_p=homog_p[:2,:]
    return hetro_p
def testH(im1,im2,H2to1,Npoints):
    im1=np.array(im1) if not isinstance(im1,np.ndarray) else np.copy(im1)
    im2=np.array(im2) if not isinstance(im1,np.ndarray) else np.copy(im2)
    
    #obtain points
    np_im2=np.array(im2)
    p2_test=np.random.random((2,Npoints))
    p2_test[0,:]*=np_im2.shape[1]
    p2_test[1,:]*=np_im2.shape[0]
    p2_test=p2_test.astype(np.float32)

    homog_p2_test=homogenize_coordinates(p2_test)
    H_homog_p2_test=(H2to1@homog_p2_test)
    H_p2=hetrogenize_coordinates(H_homog_p2_test)
    H_p2=H_p2.astype(np.int)
    
    #draw points
    fig,axes = plt.subplots(1,2,figsize = (16,5))
    colors = matplotlib.cm.viridis(np.linspace(0, 1, Npoints))                                 
       
    for i in range(Npoints):
        im1 = cv2.circle(im1, (H_p2[0,i],H_p2[1,i]), radius=10, color=255*colors[i], thickness=2)
    axes[0].imshow(im1)
    
    for i in range(Npoints):
        im2 = cv2.circle(im2, (p2_test[0,i],p2_test[1,i]), radius=10, color=255*colors[i], thickness=2)
    axes[1].imshow(im2)
    
def compute_warp_output_size(im1,H): 
    #input: 
    #im1 - np.array or PIL
    # H - 3x3 homography matrix
    
    #output:
    #[h,w] - required image size to show complete image after warping
    
    #validate input:
    if not isinstance(im1,np.ndarray):
        im1=np.array(im1) 
    
    #extract corners: buttom left, buttom right, top right, top left
    x=np.array([0,im1.shape[1],im1.shape[1],0])
    y=np.array([im1.shape[0],im1.shape[0],0,0])
    
    #compute H*corners
    p=np.vstack((x,y))
    homog_p=homogenize_coordinates(p)
    H_homog_p=(H@homog_p)
    H_p=hetrogenize_coordinates(H_homog_p)
    
    #find bounding box
    x_max=np.max(H_p[0,:])
    x_min=np.min(H_p[0,:])
    y_max=np.max(H_p[1,:])
    y_min=np.min(H_p[1,:])
    w=(np.ceil(x_max-x_min)).astype(int)
    h=(np.ceil(y_max-y_min)).astype(int)

    return [h,w]    
def compute_translation(im1,H):
    #input: 
    #im1 - np.array
    # H - 3x3 homography matrix
    
    #output:
    #translation - [x,y] bias to add to template image pixels computed in function warpH
    
    #extract corners: buttom left, buttom right, top right, top left
    x=np.array([0,im1.shape[1],im1.shape[1],0])
    y=np.array([im1.shape[0],im1.shape[0],0,0])
    
    #compute H*corners
    p=np.vstack((x,y))
    homog_p=homogenize_coordinates(p)
    H_homog_p=(H@homog_p)
    H_p=hetrogenize_coordinates(H_homog_p)
    
    #find translation
    x_min=np.min(H_p[0,:])
    y_min=np.min(H_p[1,:])
    x_translation=(np.ceil(x_min)).astype(int)
    y_translation=(np.ceil(y_min)).astype(int)
    translation=np.array([x_translation,y_translation])
    
    return translation
def crop_image(img,tol=0):
    # outputs image with no 'extra' black padding in it
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    #from https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
def plotMatches(im1, im2, p1, p2, N):
    """
    input:
    im1 and im2 are gray images used for calculating p1,p2 in fcn 'getPoints_SIFT'
    
    p1 - features in im1
    p2 - features in im2
    p1[j] matches p2[j]
    p1[j]=[[x[j],y[j]]. to extract x[j] out of p1 do p1[j][0][0]
    industry standard to represent the points as it can be an input to cv2.findhomography
    
    N - amount of matches to show. bounded by p1.shape[0]
    """
    fig = plt.figure(figsize=(20,8))
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = im1
    im[0:im2.shape[0], im1.shape[1]:] = im2
    plt.imshow(im, cmap='gray')
    
    if N=='max':
        m=range(p1.shape[0])
    else:
        m=range(N[0],N[1]) #amount of matches to show
        
    colors = matplotlib.cm.prism(np.linspace(0, 1, len(m)))
    for i, c in enumerate(colors):
        pt1 = p1[m[i]][0].copy() #.copy() is here because the variable name is a pointer to the array!
        pt2 = p2[m[i]][0].copy()
        pt2[0] += im1.shape[1] #if we dont add .copy() changing pt2[0] will change the input!!
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,color=c,linewidth=0.4)
        plt.scatter(x,y,color=c,s=3)
    plt.show()
    
def PreProcessFeaturePoints(p):
    '''
    input:
    p.shape=[m,1,2]
    p[j]=[[x[j],y[j]]. to extract x[j] out of p do p[j][0][0]
    
    output:
    pout.shape=(2,m) where first row is x and second row is y
    '''
    pout=np.zeros((2,p.shape[0])).astype(np.float32)
    for m in range(p.shape[0]):
        pout[:,m]=p[m][0]
    return pout
def plotMatches_pp(im1, im2, p1, p2, N):
    """
    input:
    im1 and im2 are gray images used for calculating p1,p2 in fcn 'getPoints_SIFT'
    
    p1 - features in im1
    p2 - features in im2
    
    p1,p2 are of size 2xm 
    
    N - amount of matches to show. bounded by p1.shape[0]
    """
    fig = plt.figure(figsize=(20,8))
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = im1
    im[0:im2.shape[0], im1.shape[1]:] = im2
    plt.imshow(im, cmap='gray')
    
    if N=='max':
        m=range(p1.shape[1])
    else:
        m=range(N[0],N[1]) #amount of matches to show
        
    colors = matplotlib.cm.prism(np.linspace(0, 1, len(m)))
    for i, c in enumerate(colors):
        pt1 = p1[:,m[i]].copy() #.copy() is here because the variable name is a pointer to the array!
        pt2 = p2[:,m[i]].copy()
        pt2[0] += im1.shape[1] #if we dont add .copy() changing pt2[0] will change the input!!
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,color=c,linewidth=0.4)
        plt.scatter(x,y,color=c,s=3)
    plt.show()
    
def Load_Images(ImageNameList,sz_width):
    images=[]
    images_g=[]
    for ImageName in ImageNameList:
        im=cv2.cvtColor(cv2.imread(ImageName),cv2.COLOR_BGR2RGB)
        im_tb=image_resize(im,width=sz_width,inter = cv2.INTER_AREA)        
        images.append(im_tb)
    return images
def Load_Images_FullSize(ImageNameList):
    images=[]
    images_g=[]
    for ImageName in ImageNameList:
        im=cv2.cvtColor(cv2.imread(ImageName),cv2.COLOR_BGR2RGB)
        images.append(im)
    return images
def ComputeH_Sift(im1,im2, method='flann',Nbest=20,plot=0):
    '''
    input:
    im1,im2 images np.array uint8
    method - siftmethod for feature matching: 'flann','brute_force_crosscheck','brute_force_knn' also options for 'orb'
    Nbest - if method == 'brute_force_crosscheck' take Nbest matches (decided by norm2 distance)
    plot - if plot == 1, plot matches
    
    output:
    homography matrix im2->im1
    '''

    im1_g = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im2_g = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    
    [p1,p2] = getPoints_SIFT(im1_g,im2_g, method = method, Nbest=Nbest)
    pp_p1=PreProcessFeaturePoints(p1)
    pp_p2=PreProcessFeaturePoints(p2)
    
    if plot:
        plotMatches(im1_g, im2_g, p1, p2, 'max')
        plt.show()

    return computeH(pp_p1,pp_p2)
def ComputeH_Manual(im1,im2, plot=0):
    '''
    input:
    im1,im2 images np.array uint8
    plot - if plot == 1, plot matches
    
    output:
    homography matrix im2->im1
    '''
    p1, p2 = getPoints(im1, im2, 4)
    if plot:
        im1_g = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2_g = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        plotMatches_pp(im1_g, im2_g, p1, p2, 'max')
        plt.show()
    
    return computeH(p1,p2)
def imageStitching4Pano(img1, warp_img2,t):
    """
    note: only difference from imageStitching is that we return more values
    
    Inputs: 
    im1 , warp_img2 are two colored images
        warp_img2 is promised to be float32 [0,1]
    t is a translation vector -  [x,y] = O(warp_img2)-O(im1), where O(im1)==[0,0]. computed in warpH
    
    Output: 
    panoImg is the output gathered panorama.
    """
    if not isinstance(img1,np.ndarray):
        img1=np.array(img1) 
    
    xmin=int(np.floor(min(0,t[0])))
    xmax=int(np.ceil(max(img1.shape[1],warp_img2.shape[1]+t[0])))
    ymin=int(np.floor(min(0,t[1])))
    ymax=int(np.ceil(max(img1.shape[0],warp_img2.shape[0]+t[1])))
    
    w=xmax-xmin
    h=ymax-ymin
    panoImg=np.zeros((h,w,3), dtype=np.uint8)
    
    dx_img2=t[0]-0-xmin #O(img2)-O(Img1)-O(panoImg)
    dy_img2=t[1]-0-ymin
    panoImg[dy_img2:dy_img2+warp_img2.shape[0],dx_img2:dx_img2+warp_img2.shape[1],:] = warp_img2
    
    #not just broadcast - dont paint black pixels
    dx_img1=0-xmin #O(img1)-O(panoImg)
    dy_img1=0-ymin
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if np.any(img1[i,j]): #at least 1 channel has color (not black pixel)
                panoImg[dy_img1+i,dx_img1+j] = img1[i,j]

    #panoImg[dy_img1:dy_img1+img1.shape[0],dx_img1:dx_img1+img1.shape[1],:] = img1
    s1=np.array([dx_img1,dy_img1])
    s2=np.array([dx_img2,dy_img2])
    return panoImg,s1,s2
def Panorama5(images,method = 'sift', siftmethod='brute_force_crosscheck', Nbest=20, plot=0):
    '''
    input:
    images - list of images to be stitched . image[2] will be counted as middle image
    method - 'manual' or 'sift'
    siftmethod - 'flann','brute_force_crosscheck','brute_force_knn' also options for 'orb'
                    all methods except for 'brute_force_crosscheck' use Lowe's ratio test.
                    'brute_force_crosscheck' selected the nbest matches by norm2 distance
    Nbest - only used if siftmethod == 'brute_force_crosscheck'. amount of matches to include. chooses by best norm2 distance
    plot - if plot, prints warped images height and width, images explaining the process
    
    output:
    panoramic image of scene
    '''
    
    if method.lower() == 'manual':            
        #bring image 1 to image 2
        H1t2 = ComputeH_Manual(images[2],images[1], plot = 1) 
        [h1t2, w1t2] = compute_warp_output_size(images[1],H1t2)
        if plot:
            print('h1t2,w1t2:')
            print(h1t2, w1t2)
            sys.stdout.flush()
        im1t2, t1t2 = warpH(images[1],H1t2,(h1t2,w1t2),method='linear',colorspace='LAB')

        #bring image 0 to image 2
        H0t1 = ComputeH_Manual(images[1],images[0], plot = 1)
        H0t2 = H1t2@H0t1
        [h0t2, w0t2] = compute_warp_output_size(images[0],H0t2)
        if plot:
            print('h0t2, w0t2:')
            print(h0t2, w0t2)
            sys.stdout.flush()
        im0t2, t0t2 = warpH(images[0],H0t2,(h0t2,w0t2),method='linear',colorspace='LAB')

        #bring image 3 to 2
        H3t2 = ComputeH_Manual(images[2],images[3], plot = 1) 
        [h3t2, w3t2] = compute_warp_output_size(images[3],H3t2)
        if plot:
            print('h3t2, w3t2:')
            print(h3t2, w3t2)
            sys.stdout.flush()
        im3t2, t3t2 = warpH(images[3],H3t2,(h3t2,w3t2),method='linear',colorspace='LAB')

        #bring image 4 to image 2
        H4t3 = ComputeH_Manual(images[3],images[4], plot = 1)
        H4t2 = H3t2@H4t3
        [h4t2, w4t2] = compute_warp_output_size(images[4],H4t2)
        if plot:
            print('h4t2, w4t2:')
            print(h4t2, w4t2)
            sys.stdout.flush()
        im4t2, t4t2 = warpH(images[4],H4t2,(h4t2,w4t2),method='linear',colorspace='LAB')

        PanoImage,sfrm1,_ = imageStitching4Pano(images[2], im1t2, t1t2)
        PanoImage,sfrm0,_ = imageStitching4Pano(PanoImage, im0t2, t0t2+sfrm1)
        PanoImage,sfrm3,_ = imageStitching4Pano(PanoImage, im3t2, t3t2+sfrm1+sfrm0)
        PanoImage,_,_ = imageStitching4Pano(PanoImage, im4t2, t4t2+sfrm1+sfrm0+sfrm3)

    elif method.lower() == 'sift':
        #bring image 1 to image 2
        H1t2 = ComputeH_Sift(images[2],images[1], method = siftmethod, Nbest = Nbest, plot = plot) 
        [h1t2, w1t2] = compute_warp_output_size(images[1],H1t2)
        if plot:
            print('h1t2,w1t2:')
            print(h1t2, w1t2)
            sys.stdout.flush()
        im1t2, t1t2 = warpH(images[1],H1t2,(h1t2,w1t2),method='linear',colorspace='LAB')

        #bring image 0 to image 2
        H0t1 = ComputeH_Sift(images[1],images[0], method = siftmethod, Nbest = Nbest, plot = plot)
        H0t2 = H1t2@H0t1
        [h0t2, w0t2] = compute_warp_output_size(images[0],H0t2)
        if plot:
            print('h0t2, w0t2:')
            print(h0t2, w0t2)
            sys.stdout.flush()
        im0t2, t0t2 = warpH(images[0],H0t2,(h0t2,w0t2),method='linear',colorspace='LAB')

        #bring image 3 to 2
        H3t2 = ComputeH_Sift(images[2],images[3], method = siftmethod, Nbest = Nbest, plot = plot) 
        [h3t2, w3t2] = compute_warp_output_size(images[3],H3t2)
        if plot:
            print('h3t2, w3t2:')
            print(h3t2, w3t2)
            sys.stdout.flush()
        im3t2, t3t2 = warpH(images[3],H3t2,(h3t2,w3t2),method='linear',colorspace='LAB')

        #bring image 4 to image 2
        H4t3 = ComputeH_Sift(images[3],images[4], method = siftmethod, Nbest = Nbest, plot = plot)
        H4t2 = H3t2@H4t3
        [h4t2, w4t2] = compute_warp_output_size(images[4],H4t2)
        if plot:
            print('h4t2, w4t2:')
            print(h4t2, w4t2)
            sys.stdout.flush()
        im4t2, t4t2 = warpH(images[4],H4t2,(h4t2,w4t2),method='linear',colorspace='LAB')

        PanoImage,sfrm1,_ = imageStitching4Pano(images[2], im1t2, t1t2)
        PanoImage,sfrm0,_ = imageStitching4Pano(PanoImage, im0t2, t0t2+sfrm1)
        PanoImage,sfrm3,_ = imageStitching4Pano(PanoImage, im3t2, t3t2+sfrm1+sfrm0)
        PanoImage,_,_ = imageStitching4Pano(PanoImage, im4t2, t4t2+sfrm1+sfrm0+sfrm3)

    return PanoImage
def ComputeH_Sift_RANSAC(im1,im2, nIter, tol, method='flann', Nbest=20, plot=0):
    '''
    input: 
    im1,im2 - images (np.array uint8)
    nIter - iterations for RANSAC
    tol - tolerance for RANSAC
    plot - if plot == 1, plots matches
    
    output:
    H - homography matrix from im2->im1
    '''
    im1_g = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im2_g = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    
    [p1,p2] = getPoints_SIFT(im1_g,im2_g, method = method, Nbest=Nbest)
    
    if plot:
        plotMatches(im1_g, im2_g, p1, p2, 'max')
        plt.show()

    return ransacH(p1, p2, nIter, tol)
def Panorama5_RANSAC(images, nIter = 1000, tol = 1,siftmethod='flann', Nbest=20, plot=0):
    '''
    input:
    images - list of images to be stitched . image[2] will be counted as middle image
    nIter - iterations for RANSAC
    tol - tolerance for RANSAC
    siftmethod - 'flann','brute_force_crosscheck','brute_force_knn' also options for 'orb'
                    all methods except for 'brute_force_crosscheck' use Lowe's ratio test.
                    'brute_force_crosscheck' selected the nbest matches by norm2 distance
    nbest - only used if siftmethod == 'brute_force_crosscheck'. amount of matches to include. chooses by best norm2 distance
    plot - if plot, prints warped images height and width, images explaining the process
    
    #output:
    panoramic image of scene
    '''

    #bring image 1 to image 2
    H1t2 = ComputeH_Sift_RANSAC(images[2],images[1], nIter, tol , method = siftmethod, Nbest = Nbest, plot = plot) 
    [h1t2, w1t2] = compute_warp_output_size(images[1],H1t2)
    if plot:
        print('h1t2,w1t2:')
        print(h1t2, w1t2)
        sys.stdout.flush()
    im1t2, t1t2 = warpH(images[1],H1t2,(h1t2,w1t2),method='linear',colorspace='LAB')

    #bring image 0 to image 2
    H0t1 = ComputeH_Sift_RANSAC(images[1],images[0], nIter, tol ,method = siftmethod, Nbest = Nbest, plot = plot)
    H0t2 = H1t2@H0t1
    [h0t2, w0t2] = compute_warp_output_size(images[0],H0t2)
    if plot:
        print('h0t2, w0t2:')
        print(h0t2, w0t2)
        sys.stdout.flush()
    im0t2, t0t2 = warpH(images[0],H0t2,(h0t2,w0t2),method='linear',colorspace='LAB')

    #bring image 3 to 2
    H3t2 = ComputeH_Sift_RANSAC(images[2],images[3], nIter, tol , method = siftmethod, Nbest = Nbest, plot = plot) 
    [h3t2, w3t2] = compute_warp_output_size(images[3],H3t2)
    if plot:
        print('h3t2, w3t2:')
        print(h3t2, w3t2)
        sys.stdout.flush()
    im3t2, t3t2 = warpH(images[3],H3t2,(h3t2,w3t2),method='linear',colorspace='LAB')

    #bring image 4 to image 2
    H4t3 = ComputeH_Sift_RANSAC(images[3],images[4], nIter, tol , method = siftmethod, Nbest = Nbest, plot = plot)
    H4t2 = H3t2@H4t3
    [h4t2, w4t2] = compute_warp_output_size(images[4],H4t2)
    if plot:
        print('h4t2, w4t2:')
        print(h4t2, w4t2)
        sys.stdout.flush()
    im4t2, t4t2 = warpH(images[4],H4t2,(h4t2,w4t2),method='linear',colorspace='LAB')

    PanoImage,sfrm1,_ = imageStitching4Pano(images[2], im1t2, t1t2)
    PanoImage,sfrm0,_ = imageStitching4Pano(PanoImage, im0t2, t0t2+sfrm1)
    PanoImage,sfrm3,_ = imageStitching4Pano(PanoImage, im3t2, t3t2+sfrm1+sfrm0)
    PanoImage,_,_ = imageStitching4Pano(PanoImage, im4t2, t4t2+sfrm1+sfrm0+sfrm3)

    return PanoImage
def dist2D_pfroml(p,q1,q2):
    '''
    input:
    p,q1,q2 - points [x,y]
    q1,q2 represent the line
    
    output:
    d - distance of point to line
    '''
    v1=q2-q1
    v2=p-q1  
    d=abs(cross2D(v1,v2))/np.linalg.norm(v1)
    return d
def cross2D(v1,v2):
    return v1[0]*v2[1]-v2[0]*v1[1]
def compute_warp_corners_and_output_size(im,H): 
    #input: 
    #im - np.array or PIL
    #H - 3x3 homography matrix
    
    #output:
    #[h,w] - required image size to show complete image after warping
    
    #validate input:
    if not isinstance(im,np.ndarray):
        im=np.array(im) 
    
    #extract corners: buttom left, buttom right, top right, top left
    x=np.array([0,im.shape[1],im.shape[1],0])
    y=np.array([im.shape[0],im.shape[0],0,0])
    
    #compute H*corners
    p=np.vstack((x,y))
    homog_p=homogenize_coordinates(p)
    H_homog_p=(H@homog_p)
    H_p=hetrogenize_coordinates(H_homog_p)
    
    #find bounding box
    x_max=np.max(H_p[0,:])
    x_min=np.min(H_p[0,:])
    y_max=np.max(H_p[1,:])
    y_min=np.min(H_p[1,:])
    w=(np.ceil(x_max-x_min)).astype(int)
    h=(np.ceil(y_max-y_min)).astype(int)

    return h,w,H_p
def ContestPixel(i,j,dy_img1,dx_img1,img1,warp_img2,warp_img2_corners,threshold=10):
    '''
    see documentation in attached presentation.
    img1 is RGB np.array uint8 rectangle pic
    warp_img2 is RGB np.array uint8 not rectangle, but has black padding to create rectangle pic.
    warp_img2 is in the same coordiante system as img1.
    
    i,j are row,col indcies that fit img1
    dy_img1, dx_img1 are transaltions of img1 for panoramic image
    warp_img2_corners is a 2xm matrix of [x;y] of the 4 corners of warp_img2.
    
    threshold is a distance parameter from which we dont blend, but take img1 as it is.
    
    function returns the blended pixel value of pixel [i,j]
    pixel value determined by distance from the edges of img1,img2_warped
    '''
    h=img1.shape[0]
    w=img1.shape[1]
    s1=min([i,j,h-i,w-j])
    
    p=[j,i] #x,y
    s2=min([dist2D_pfroml(p,warp_img2_corners[:,0],warp_img2_corners[:,1]),
            dist2D_pfroml(p,warp_img2_corners[:,1],warp_img2_corners[:,2]),
            dist2D_pfroml(p,warp_img2_corners[:,2],warp_img2_corners[:,3]),
            dist2D_pfroml(p,warp_img2_corners[:,3],warp_img2_corners[:,0])])
    
    if s1>threshold:
        pixelVal=img1[i,j]     
    else:
        alpha_im1 = s1/(s1+s2)
        pixelVal = alpha_im1*img1[i,j] + (1-alpha_im1)*warp_img2[i+dy_img1,j+dx_img1]
    
    return pixelVal
def Stitch_DistanceBlend(img1,warp_img2,t,t_warp_img2_corners,threshold):
    """
    Inputs: 
    im1 , warp_img2 are two colored images uint8
    t is a translation vector -  [x,y] = O(warp_img2)-O(im1), where O(im1)==[0,0]. computed in warpH
    t_warp_img2_corners is a 2xm np array [-x-;-y-] of corner locations after warping AND translation
    threshold - threshold for ContestPixel. distance from which we just take img1 pixel val (no blending)
    
    Output: 
    panoImg is the output gathered panorama.
    """
    if not isinstance(img1,np.ndarray):
        img1=np.array(img1) 
    
    xmin=int(np.floor(min(0,t[0])))
    xmax=int(np.ceil(max(img1.shape[1],warp_img2.shape[1]+t[0])))
    ymin=int(np.floor(min(0,t[1])))
    ymax=int(np.ceil(max(img1.shape[0],warp_img2.shape[0]+t[1])))
    
    w=xmax-xmin
    h=ymax-ymin
    panoImg=np.zeros((h,w,3), dtype=np.uint8)
    
    dx_img2=t[0]-0-xmin #O(img2)-O(Img1)-O(panoImg)
    dy_img2=t[1]-0-ymin
    panoImg[dy_img2:dy_img2+warp_img2.shape[0],dx_img2:dx_img2+warp_img2.shape[1],:] = warp_img2
    
    #not just broadcast - dont paint black pixels
    dx_img1=0-xmin #O(img1)-O(panoImg)
    dy_img1=0-ymin
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if np.any(img1[i,j]): #at least 1 channel has color (not black pixel)
                if np.any(panoImg[dy_img1+i,dx_img1+j]): #warp_img2 exists in this place
                    panoImg[dy_img1+i,dx_img1+j] =ContestPixel(i,j,dy_img1,dx_img1,img1,panoImg,t_warp_img2_corners,threshold)
                else: #only img1 exists in [i,j]
                    panoImg[dy_img1+i,dx_img1+j] = img1[i,j]            
    return panoImg         
def createMask_fromAny(im):
    '''
    returns a mask same size of im such that any pixel that has some color (not completely black) will accept 255
    holes are filled incase there are completely black pixels in middle of shape
    '''
    mask=np.zeros((im.shape[0],im.shape[1]),dtype=np.bool)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if any(im[i,j]):
                mask[i,j]=1
    mask = scipy.ndimage.binary_fill_holes(mask).astype(np.uint8)*255
                  
    return mask
def findContours(im):
    '''
    input:
    im - RGB np.array uint8
    
    output:
    contours - some wierd shape. [x,y] of contour pixels. to get xi do: contours[0][i][0]
    BinaryMask - mask of filled contours
    imWithcontours - originial image with contours drawn in light green ontop
    '''
    im=im.copy() #.copy() inside!!!
    BinaryMask=createMask_fromAny(im)
    contours, hierarachy = cv2.findContours(BinaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    imWithcontours=cv2.drawContours(im, contours, 0, (0, 255, 0), 2) 
    return contours,BinaryMask,imWithcontours
def div(i,j,im1,im2):
    #used in poisson_blending function
    pixel_value = np.array([0,0,0])
    pixel_value[0] = im1[i,j,0] * 4 - im1[i+1,j,0] - im1[i-1,j,0] - im1[i,j+1,0] - im1[i,j-1,0] 
    pixel_value[1] = im1[i,j,1] * 4 - im1[i+1,j,1] - im1[i-1,j,1] - im1[i,j+1,1] - im1[i,j-1,1] 
    pixel_value[2] = im1[i,j,2] * 4 - im1[i+1,j,2] - im1[i-1,j,2] - im1[i,j+1,2] - im1[i,j-1,2]
    return pixel_value
def poisson_blending(im1,im2,ref, plot=0):
    '''
    input:
    im1 is image to blended with im2 (cat)
    im2 is base image (library)
    both images are uint8 RGB numpy arrays
    
    plot - if plot == 1, plots 3 images: image with contours, mask of filled contour and image of boundary starting conditions
    
    ref - tuple. translation of im2 inside im1 [y,x]
    
    outout:
    img_blend - changed im2 with im1 copied and blended on top.
    '''
    
    contours,_,imWithcontours= findContours(im1)
    mask = createMask_fromAny(im1)/255 #createMask_fromAny returns 255 or 0 and we want 1 or 0
    for c in contours[0]:
        x=c[0][0]
        y=c[0][1]
        
        mask[y,x]=0 #delete contour from mask
        
        if (np.sum(im2[y+ref[0],x+ref[1]])==0): #if im2 has a black pixel in [y,x], insert pixel from im1
            im2[y+ref[0],x+ref[1]] = im1[y,x]
    
    if plot:
        fig = plt.figure(figsize=(20,8)); plt.imshow(imWithcontours)
        fig = plt.figure(figsize=(20,8)); plt.imshow(mask,cmap = 'gray')
        fig = plt.figure(figsize=(20,8)); plt.imshow(im2) #boundary from im1 was added         
    
    #image shape
    r_, c_ = mask.shape[:2]
    
    pix_num = r_ * c_
    
    F = np.zeros((pix_num, 3))
    
    A = scipy.sparse.identity(pix_num, format='lil')
    
    gdiv = lambda ii,jj: div(ii,jj, im1, im2)
    num = lambda ii,jj: ii + jj*r_
            
    for i in range(r_):
            for j in range(c_):
                k = num(i,j)

                if mask[i, j] == 1:
                    pix_val = np.array([0.0, 0.0, 0.0])

                    if mask[i - 1, j] == 1:
                        A[k, k - 1] = -1
                    else:
                        pix_val += im2[i - 1 + ref[0], j + ref[1]]

                    if mask[i + 1, j] == 1:
                        A[k, k + 1] = -1
                    else:
                        pix_val += im2[i + 1 + ref[0], j + ref[1]]

                    if mask[i, j - 1] == 1:
                        A[k, k - r_] = -1
                    else:
                        pix_val += im2[i + ref[0], j - 1 + ref[1]]

                    if mask[i, j + 1] == 1:
                        A[k, k + r_] = -1
                    else:
                        pix_val += im2[i + ref[0], j + 1 + ref[1]]

                    A[k, k] = 4
                    F[k] = gdiv(i, j) + pix_val

                else:
                    F[k] = im2[i + ref[0], j + ref[1]]
    A = A.tocsr()

    img_blend = im2.astype(np.uint8).copy()

    for l in range(3):
        x = spa.linalg.spsolve(A, F[:, l])
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, img_blend.dtype)
        img_blend[ref[0]:ref[0] + r_, ref[1]:ref[1] + c_, l] = x.reshape(r_, c_,order = 'F')

    return img_blend
def Stitch4Pano_PoissonBlend(img1,warp_img2,t, plot=0 ,Addinfront=0):
    '''
    input: 
    img1(np.array uint8)
    warp_img2 - img2 in img1 coordiantes system (np.array uint8)
    t - [x,y] translation of coordiantes of warp_img2 relative to img1
    plot - if plot==1 poisson_blending function will plot the blending process
    Addinfront - if Addinfront == 1, put warp_img2 infront of img1 
    
    output:
    PanoImg_Blend - stiched and blended image (np.array uint8)
    s1 - important parameter: [x,y] shift in PanoImg_Blend from originial img1 coordiantes
    s2 - not important parameter: [x,y] shift in img2 coordinates from orginal img1 coordinates
    '''
    
    xmin=int(np.floor(min(0,t[0])))
    xmax=int(np.ceil(max(img1.shape[1],warp_img2.shape[1]+t[0])))
    ymin=int(np.floor(min(0,t[1])))
    ymax=int(np.ceil(max(img1.shape[0],warp_img2.shape[0]+t[1])))
    
    w=xmax-xmin
    h=ymax-ymin
    panoImg=np.zeros((h,w,3), dtype=np.uint8)
    
    dx_img1=0-xmin #O(img1)-O(panoImg)
    dy_img1=0-ymin
    
    dx_img2=t[0]-0-xmin #O(img2)-O(Img1)-O(panoImg)
    dy_img2=t[1]-0-ymin
    
    if Addinfront:
        panoImg[dy_img1:dy_img1+img1.shape[0],dx_img1:dx_img1+img1.shape[1],:] = img1
        panoImg_Blend = poisson_blending(warp_img2, panoImg, #left image is cat, right image is library
                                         ref=(dy_img2,dx_img2), plot=plot) #ref is (rows,cols)
    else:
        panoImg[dy_img2:dy_img2+warp_img2.shape[0],dx_img2:dx_img2+warp_img2.shape[1],:] = warp_img2
        panoImg_Blend = poisson_blending(img1,panoImg, #left image is cat, right image is library
                                         ref=(dy_img1,dx_img1), plot=plot) #ref is (rows,cols)
    
    s1=np.array([dx_img1,dy_img1])
    s2=np.array([dx_img2,dy_img2])
    return panoImg_Blend,s1,s2
def Panorama5_RANSAC_PoissonBlend(images, nIter = 1000, tol = 1, siftmethod='flann', Nbest=20, plot=0):
    '''
    input:
    images - list of images to be stitched . image[2] will be counted as middle image
    nIter - iterations for RANSAC
    tol - tolerance for RANSAC
    siftmethod - 'flann','brute_force_crosscheck','brute_force_knn' also options for 'orb'
                    all methods except for 'brute_force_crosscheck' use Lowe's ratio test.
                    'brute_force_crosscheck' selected the nbest matches by norm2 distance
    nbest - only used if siftmethod == 'brute_force_crosscheck'. amount of matches to include. chooses by best norm2 distance
    plot - if plot, prints warped images height and width, images explaining the process
    
    #output:
    panoramic image of scene
    '''
    siftmethod='flann' #default
    Nbest = 20 #default, value for siftmethod = brute_force_crosscheck
    plot = 0 #default dont plot
    for key, value in kwargs.items(): 
        if key.lower()=='siftmethod':
            siftmethod = value
        if key.lower()=='nbest':
            Nbest = value
        if key.lower()=='plot':
            plot = value

    #bring image 1 to image 2
    H1t2 = ComputeH_Sift_RANSAC(images[2],images[1], nIter, tol , method = siftmethod, Nbest = Nbest, plot = plot) 
    [h1t2, w1t2] = compute_warp_output_size(images[1],H1t2)
    if plot: print('h1t2,w1t2:'); print(h1t2, w1t2); sys.stdout.flush();
    im1t2, t1t2 = warpH(images[1],H1t2,(h1t2,w1t2),method='cubic',colorspace='LAB')

    #bring image 0 to image 2
    H0t1 = ComputeH_Sift_RANSAC(images[1],images[0], nIter, tol ,method = siftmethod, Nbest = Nbest, plot = plot)
    H0t2 = H1t2@H0t1
    [h0t2, w0t2] = compute_warp_output_size(images[0],H0t2)
    if plot: print('h0t2, w0t2:'); print(h0t2, w0t2); sys.stdout.flush();
    im0t2, t0t2 = warpH(images[0],H0t2,(h0t2,w0t2),method='cubic',colorspace='LAB')

    #bring image 3 to 2
    H3t2 = ComputeH_Sift_RANSAC(images[2],images[3], nIter, tol , method = siftmethod, Nbest = Nbest, plot = plot) 
    [h3t2, w3t2] = compute_warp_output_size(images[3],H3t2)
    if plot: print('h3t2, w3t2:'); print(h3t2, w3t2);sys.stdout.flush();
    im3t2, t3t2 = warpH(images[3],H3t2,(h3t2,w3t2),method='cubic',colorspace='LAB')

    #bring image 4 to image 2
    H4t3 = ComputeH_Sift_RANSAC(images[3],images[4], nIter, tol , method = siftmethod, Nbest = Nbest, plot = plot)
    H4t2 = H3t2@H4t3
    [h4t2, w4t2] = compute_warp_output_size(images[4],H4t2)
    if plot: print('h4t2, w4t2:'); print(h4t2, w4t2); sys.stdout.flush();
    im4t2, t4t2 = warpH(images[4],H4t2,(h4t2,w4t2),method='cubic',colorspace='LAB')
    
    #stich images togther using the poisson method
    PanoImage,sfrm1,_ = Stitch4Pano_PoissonBlend(images[2], im1t2, t1t2, plot = plot)
    if plot: fig = plt.figure(figsize=(20,8)); plt.imshow(PanoImage)
    
    PanoImage,sfrm0,_ = Stitch4Pano_PoissonBlend(PanoImage, im0t2, t0t2+sfrm1, plot = plot)
    if plot: fig = plt.figure(figsize=(20,8)); plt.imshow(PanoImage)
    
    PanoImage,sfrm3,_ = Stitch4Pano_PoissonBlend(PanoImage, im3t2, t3t2+sfrm1+sfrm0, plot = plot)
    if plot: fig = plt.figure(figsize=(20,8)); plt.imshow(PanoImage)
    
    PanoImage,_,_ = Stitch4Pano_PoissonBlend(PanoImage, im4t2, t4t2+sfrm1+sfrm0+sfrm3, plot = plot)

    return PanoImage #not in use
def Panorama_Nodd(images,nIter=1000,tol=0.3,siftmethod='flann',Nbest=20, plot=0, Addinfront=0):
    '''
    stitches images to panoramic using feature matching, ransac and poisson blending
    
    input:
    images - list of images to be stitched . image[len(images)/2] will be assumed to be the middle image
    nIter - iterations for RANSAC
    tol - tolerance for RANSAC
    siftmethod - 'flann','brute_force_crosscheck','brute_force_knn' also options for 'orb'
                    all methods except for 'brute_force_crosscheck' use Lowe's ratio test.
                    'brute_force_crosscheck' selected the nbest matches by norm2 distance
    nbest - only used if siftmethod == 'brute_force_crosscheck'. amount of matches to include. chooses by best norm2 distance
    plot - if plot, prints warped images height and width, images explaining the process
    Addinfront - if Addinfront == 1, put warp_img2 infront of img1 
    
    #output:
    panoramic image of scene
    '''
            
    #find index of middle picture
    Nmid=int(len(images)/2) #start index is 0, so dividing by 2 works. 
    PanoImage = images[Nmid]
    H_L_t_Mid=np.eye(3);
    H_R_t_Mid=np.eye(3);
    s=0 #shift in coordiantes from adding images to pano image
    
    for k in range(Nmid):
        #-----------------bring image Nmid-k-1 to mid (from left to mid)
        #compute H
        H_km1_t_k = ComputeH_Sift_RANSAC(images[Nmid-k],images[Nmid-k-1], nIter, tol,
                                         method = siftmethod, Nbest = Nbest, plot = plot)
        H_L_t_Mid = H_L_t_Mid@H_km1_t_k
        #compute warp    
        [h_L_t_Mid, w_L_t_Mid] = compute_warp_output_size(images[Nmid-k-1],H_L_t_Mid)
        if plot: print('h_warped_left,w_warped_left:'); print(h_L_t_Mid, w_L_t_Mid); sys.stdout.flush();
        im_L_t_Mid, t_L_t_Mid = warpH(images[Nmid-k-1],H_L_t_Mid,(h_L_t_Mid, w_L_t_Mid),method='cubic',colorspace='LAB')
        
        #-----------------bring image Nmid+k+1 to mid (from right to mid)
        #compute H
        H_kp1_t_k = ComputeH_Sift_RANSAC(images[Nmid+k],images[Nmid+k+1], nIter, tol,
                                         method = siftmethod, Nbest = Nbest, plot = plot)
        H_R_t_Mid = H_R_t_Mid@H_kp1_t_k
        #compute warp    
        [h_R_t_Mid, w_R_t_Mid] = compute_warp_output_size(images[Nmid+k+1],H_R_t_Mid)
        if plot: print('h_warped_right,w_warped_right:'); print(h_R_t_Mid, w_R_t_Mid); sys.stdout.flush();
        im_R_t_Mid, t_R_t_Mid = warpH(images[Nmid+k+1],H_R_t_Mid,(h_R_t_Mid, w_R_t_Mid),method='cubic',colorspace='LAB')
        
        #------------------stich images to middle image (creating pano)
        PanoImage,s_frm_L,_ = Stitch4Pano_PoissonBlend(PanoImage, im_L_t_Mid, t_L_t_Mid+s,
                                                       plot = plot, Addinfront=Addinfront) #left to mid
        s=s+s_frm_L
        PanoImage,s_frm_R,_ = Stitch4Pano_PoissonBlend(PanoImage, im_R_t_Mid, t_R_t_Mid+s,
                                                       plot = plot , Addinfront=Addinfront) #left to mid
        s=s+s_frm_R
        if plot: fig = plt.figure(figsize=(20,8)); plt.imshow(PanoImage)
            
    return PanoImage #includes RANSAC and blending
def computeAffine(p1,p2):
    '''
    Inputs: 
    p1 and p2 should be matrices of corresponding coordinates between two images. 
    
    Outputs: 
    H2to1 should be a matrix minimizing p1-H*p2
    '''
    #p1 = H*p2
    #[x,y,1].T=H[u,v,1].T
    
    if p1.shape!=p2.shape:
        return 0
    
    #construct A
    m=p1.shape[1] #amount of points
    A=np.zeros((2*m,6)) 
    b=np.zeros((2*m,1))
    for i in range(m):
        x=p1[0,i]
        y=p1[1,i]
        u=p2[0,i]
        v=p2[1,i]
        
        A[i*2,:]=[u,v,1,0,0,0]
        A[i*2+1,:]=[0,0,0,u,v,1]
        b[i*2,:]=x
        b[i*2+1,:]=y

    h=np.linalg.pinv(A)@b
    Hr12 = h.reshape(2,3)
    Hr3 = np.array([0,0,1])
    H2to1=np.vstack([Hr12,Hr3])
    
    return H2to1
def ransacH_Affine(p1, p2, nIter, tol):
    """
    Inputs: 
    p1 and p2 are matrices specifying point locations in each of the images a
    nd p1[j],p2[j] are matched points between two images.

    nIter is the number of iterations to run RANSAC for

    tol is the tolerance value for considering a point to be an inlier

    Outputs:
    bestH should be the homography model with the most inliers found during RANSAC
    """
    n = p1.shape[0]
    
    pp_p1=PreProcessFeaturePoints(p1) #turns to 2xm 
    pp_p2=PreProcessFeaturePoints(p2)
    
    homog_pp_p1=homogenize_coordinates(pp_p1) #turns to 3xm
    homog_pp_p2=homogenize_coordinates(pp_p2)
    
    max_inliers = 0
    for i in range(nIter):
        ind_pairs = np.random.randint(n, size=4)

        Ip_p1=pp_p1[:,ind_pairs] #chose from 2xm
        Ip_p2=pp_p2[:,ind_pairs]

        H_candidate = computeAffine(Ip_p1, Ip_p2) #accepts 2xm
        H_homog_pp_p2 = H_candidate @ homog_pp_p2 #outputs 3xm
        H_p2=hetrogenize_coordinates(H_homog_pp_p2) #turns to 2xm
        
        inliers_index = np.where((np.sum(np.abs(H_p2-pp_p1)**2,axis=0)**(1./2)<tol))[0]
        if (max_inliers < len(inliers_index)):
            max_inliers       = len(inliers_index)
            bestH = H_candidate   
            
    return bestH
def ComputeH_Sift_RANSAC_Affine(im1,im2, nIter, tol, method='flann', Nbest=20, plot=0):
    '''
    input: 
    im1,im2 - images (np.array uint8)
    nIter - iterations for RANSAC
    tol - tolerance for RANSAC
    plot - if plot == 1, plots matches
    
    output:
    H - homography matrix from im2->im1
    '''
    im1_g = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im2_g = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    
    [p1,p2] = getPoints_SIFT(im1_g,im2_g, method = method, Nbest=Nbest)
    
    if plot:
        plotMatches(im1_g, im2_g, p1, p2, 'max')
        plt.show()

    return ransacH_Affine(p1, p2, nIter, tol)
def Panorama_Nodd_Affine(images,nIter=1000,tol=0.3,siftmethod='flann',Nbest=20, plot=0, Addinfront=0):
    '''
    stitches images to panoramic using feature matching, ransac and poisson blending
    
    input:
    images - list of images to be stitched . image[len(images)/2] will be assumed to be the middle image
    nIter - iterations for RANSAC
    tol - tolerance for RANSAC
    siftmethod - 'flann','brute_force_crosscheck','brute_force_knn' also options for 'orb'
                    all methods except for 'brute_force_crosscheck' use Lowe's ratio test.
                    'brute_force_crosscheck' selected the nbest matches by norm2 distance
    nbest - only used if siftmethod == 'brute_force_crosscheck'. amount of matches to include. chooses by best norm2 distance
    plot - if plot, prints warped images height and width, images explaining the process
    Addinfront - if Addinfront == 1, put warp_img2 infront of img1 
    
    #output:
    panoramic image of scene
    '''
            
    #find index of middle picture
    Nmid=int(len(images)/2) #start index is 0, so dividing by 2 works. 
    PanoImage = images[Nmid]
    H_L_t_Mid=np.eye(3);
    H_R_t_Mid=np.eye(3);
    s=0 #shift in coordiantes from adding images to pano image
    
    for k in range(Nmid):
        #-----------------bring image Nmid-k-1 to mid (from left to mid)
        #compute H
        H_km1_t_k = ComputeH_Sift_RANSAC_Affine(images[Nmid-k],images[Nmid-k-1], nIter, tol,
                                         method = siftmethod, Nbest = Nbest, plot = plot)
        H_L_t_Mid = H_L_t_Mid@H_km1_t_k
        #compute warp    
        [h_L_t_Mid, w_L_t_Mid] = compute_warp_output_size(images[Nmid-k-1],H_L_t_Mid)
        if plot: print('h_warped_left,w_warped_left:'); print(h_L_t_Mid, w_L_t_Mid); sys.stdout.flush();
        im_L_t_Mid, t_L_t_Mid = warpH(images[Nmid-k-1],H_L_t_Mid,(h_L_t_Mid, w_L_t_Mid),method='cubic',colorspace='LAB')
        
        #-----------------bring image Nmid+k+1 to mid (from right to mid)
        #compute H
        H_kp1_t_k = ComputeH_Sift_RANSAC_Affine(images[Nmid+k],images[Nmid+k+1], nIter, tol,
                                         method = siftmethod, Nbest = Nbest, plot = plot)
        H_R_t_Mid = H_R_t_Mid@H_kp1_t_k
        #compute warp    
        [h_R_t_Mid, w_R_t_Mid] = compute_warp_output_size(images[Nmid+k+1],H_R_t_Mid)
        if plot: print('h_warped_right,w_warped_right:'); print(h_R_t_Mid, w_R_t_Mid); sys.stdout.flush();
        im_R_t_Mid, t_R_t_Mid = warpH(images[Nmid+k+1],H_R_t_Mid,(h_R_t_Mid, w_R_t_Mid),method='cubic',colorspace='LAB')
        
        #------------------stich images to middle image (creating pano)
        PanoImage,s_frm_L,_ = Stitch4Pano_PoissonBlend(PanoImage, im_L_t_Mid, t_L_t_Mid+s,
                                                       plot = plot, Addinfront=Addinfront) #left to mid
        s=s+s_frm_L
        PanoImage,s_frm_R,_ = Stitch4Pano_PoissonBlend(PanoImage, im_R_t_Mid, t_R_t_Mid+s,
                                                       plot = plot , Addinfront=Addinfront) #left to mid
        s=s+s_frm_R
        if plot: fig = plt.figure(figsize=(20,8)); plt.imshow(PanoImage)
            
    return PanoImage #includes RANSAC and blending
#Extra functions end

# HW functions:
def getPoints(im1,im2,N):
    '''
    Inputs: 
    im1 and im2 are two 2D grayscale images.
    N is the number of corrosponding points you want to extract. 

    Output:
    p1 and p2 should be
    matrices of corresponding coordinates between two images.
    '''
    fig,axes=plt.subplots(1,2,figsize=(16,10))
    axes[0].imshow(im1)
    axes[1].imshow(im2)
    fig.suptitle('{} points in im1(left) and then {} points in im2(right)'.format(N,N), fontsize=40)
    
    p = plt.ginput(2*N, timeout = 0)
    p1=p[0:N]
    p2=p[N:2*N]
    
    p1=np.array(p1)
    p2=np.array(p2)

    return p1.T,p2.T
def computeH(p1, p2):
    '''
    Inputs: 
    p1 and p2 should be matrices of corresponding coordinates between two images. 
    
    Outputs: 
    H2to1 should be a matrix minimizing p1-H*p2
    
    #p1 = H*p2
    #[x,y,1].T=H[u,v,1].T
    '''
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)

    #construct A
    m=p1.shape[1] #amount of points
    A=np.zeros((2*m,9)) 
    for i in range(m):
        x=p1[0,i]
        y=p1[1,i]
        u=p2[0,i]
        v=p2[1,i]
        
        A[i*2,:]=[u,v,1,0,0,0,-x*u,-x*v,-x]
        A[i*2+1,:]=[0,0,0,u,v,1,-y*u,-y*v,-y]
        
#     #SVD:
#     #solve for h
#     (U,D,Vh) = np.linalg.svd(A)
#     h = Vh.T[:,-1]
#     #reshape to homography matrix
#     h = h/h[8]
#     H2to1=h.reshape((3,3))  
    
    #eig method:
    # make inner product of the matrices to bound the size 9x9
    M = A.T @ A
    # calculate the eigen values and the eigen vectors
    w, v  = np.linalg.eig(M)
    # find the minimal eigen vector
    index = np.argmin(w)
    #rearrange the minimal eigen vector to the homogenous matrix
    v_norm = v[:, index]
    v_norm = v_norm/v_norm[8]
    H2to1 = v_norm.reshape(3,3)
    return H2to1
def warpH(im1, H, out_size, method='cubic', colorspace='rgb'):
    """
    Inputs: 
    im1 is a colored image. 
    H is a 3 × 3 matrix encoding the homography between im1 and im2. 
    out_size is the size of the wanted output (new_imH,new_imW).
    colorspace - 'rgb'/'lab'/'gray'
    method - interpolation method 'cubic'/'linear'

    Output: 
    warp_im1 - transposed warp image im1 include empty background (zeros).
    translation - [x,y] bias of warp_im1 coordinates. required to subtract for stiching
    """
    #---------------------------------preprocessing--------------------------
    #validate input:
    if not isinstance(im1,np.ndarray):
        im1=np.array(im1) 
    #preprocess
    if colorspace.lower()=='gray':
            im1 =color.rgb2gray(im1)
            im1 = np.expand_dims(im1, axis=2)
    if colorspace.lower()=='lab':
            im1 = color.rgb2lab(im1)
            
    #------------------------------------Main-------------------------------
    NM=out_size[0]*out_size[1] #amount of points in template image
    translation=compute_translation(im1,H)
    #build vector
    xtmp = np.arange(out_size[1])+translation[0]
    ytmp = np.arange(out_size[0])+translation[1]
    #create homogenize coordiantes of new image
    xxtmp, yytmp = np.meshgrid(xtmp, ytmp)
    xxtmp_ravel=np.expand_dims(xxtmp.ravel(),axis=0)
    yytmp_ravel=np.expand_dims(yytmp.ravel(),axis=0)
    ptmp=np.vstack((xxtmp_ravel,yytmp_ravel))
    homog_ptmp=homogenize_coordinates(ptmp)
    #apply H on homogenious coordiantes of new image and extract hetrogenous coordinates
    H_homog_ptmp=np.linalg.inv(H)@homog_ptmp
    H_ptmp=hetrogenize_coordinates(H_homog_ptmp)
    
    #create xy of im1 for color draining and interpolation
    xim1 = np.arange(im1.shape[1])
    yim1 = np.arange(im1.shape[0])
    
    #construct new image
    warp_im1=np.zeros((out_size[0],out_size[1],im1.shape[2]))
    for c in range(im1.shape[2]):
        #build interpolation fcn
        I = im1[:,:,c]
        f = interpolate.interp2d(xim1,yim1,I,kind=method,bounds_error = False, fill_value = 0)
        #interpolate
        Itmp_ravel=np.zeros((NM))
        for i in range(NM):
            xq=H_ptmp[0,i]
            yq=H_ptmp[1,i]
            Itmp_ravel[i]=f(xq,yq)
        warp_im1[:,:,c]=Itmp_ravel.reshape(out_size)
    
    #--------------------------------post processing----------------------------------
    if colorspace.lower()=='gray':
        warp_im1=warp_im1.squeeze()
    if colorspace.lower()=='lab':
        warp_im1 = 255*color.lab2rgb(warp_im1) #rgb2lab maps to [0,1] and converts to float
                                
    warp_im1=np.clip(warp_im1,0,255).astype(np.uint8)
    return warp_im1,translation
def imageStitching(img1, wrap_img2):
    '''
    function called "Stitch4Pano_PoissonBlend" in ipynb "HW4_2" found under Q2.9
    
    input: 
    img1(np.array uint8)
    warp_img2 - img2 in img1 coordiantes system (np.array uint8)
    t - [x,y] translation of coordiantes of warp_img2 relative to img1
    plot - if plot==1 poisson_blending function will plot the blending process
    
    output:
    PanoImg_Blend - stiched and blended image (np.array uint8)
    s1 - important parameter: [x,y] shift in PanoImg_Blend from originial img1 coordiantes
    s2 - not important parameter: [x,y] shift in img2 coordinates from orginal img1 coordinates
    '''
    
    xmin=int(np.floor(min(0,t[0])))
    xmax=int(np.ceil(max(img1.shape[1],warp_img2.shape[1]+t[0])))
    ymin=int(np.floor(min(0,t[1])))
    ymax=int(np.ceil(max(img1.shape[0],warp_img2.shape[0]+t[1])))
    
    w=xmax-xmin
    h=ymax-ymin
    panoImg=np.zeros((h,w,3), dtype=np.uint8)
    
    dx_img2=t[0]-0-xmin #O(img2)-O(Img1)-O(panoImg)
    dy_img2=t[1]-0-ymin
    panoImg[dy_img2:dy_img2+warp_img2.shape[0],dx_img2:dx_img2+warp_img2.shape[1],:] = warp_img2
       
    dx_img1=0-xmin #O(img1)-O(panoImg)
    dy_img1=0-ymin
    panoImg_Blend = poisson_blending(img1,panoImg, #left image is cat, right image is library
                                                    ref=(dy_img1,dx_img1), plot=plot) #ref is (rows,cols)
    
    s1=np.array([dx_img1,dy_img1])
    s2=np.array([dx_img2,dy_img2])
    return panoImg,s1,s2
def ransacH(p1, p2, nIter, tol):
    """
    ransacH in this file was defined to accept the following inputs: (matches, locs1, locs2, nIter, tol)
    It was not defined differently in the pdf ransacH(p1, p2, nIter, tol).
    We went with the PDF version
    
    Inputs: 
    p1 and p2 are matrices specifying point locations in each of the images a
    nd p1[j],p2[j] are matched points between two images.

    nIter is the number of iterations to run RANSAC for

    tol is the tolerance value for considering a point to be an inlier

    Outputs:
    bestH should be the homography model with the most inliers found during RANSAC
    """
    n = p1.shape[0]
    
    pp_p1=PreProcessFeaturePoints(p1) #turns to 2xm 
    pp_p2=PreProcessFeaturePoints(p2)
    
    homog_pp_p1=homogenize_coordinates(pp_p1) #turns to 3xm
    homog_pp_p2=homogenize_coordinates(pp_p2)
    
    max_inliers = 0
    for i in range(nIter):
        ind_pairs = np.random.randint(n, size=4)

        Ip_p1=pp_p1[:,ind_pairs] #chose from 2xm
        Ip_p2=pp_p2[:,ind_pairs]

        H_candidate = computeH(Ip_p1, Ip_p2) #accepts 2xm
        H_homog_pp_p2 = H_candidate @ homog_pp_p2 #outputs 3xm
        H_p2=hetrogenize_coordinates(H_homog_pp_p2) #turns to 2xm
        
        inliers_index = np.where((np.sum(np.abs(H_p2-pp_p1)**2,axis=0)**(1./2)<tol))[0]
        if (max_inliers < len(inliers_index)):
            max_inliers       = len(inliers_index)
            bestH = H_candidate   
            
    return bestH
def getPoints_SIFT(im1, im2,method='flann', Nbest=10, plot=0):
    """
    input:
    im1 and im2 are gray images np.arary uint8
    method - siftmethod for feature matching: 'flann','brute_force_crosscheck','brute_force_knn' also options for 'orb'
    Nbest - if method == 'brute_force_crosscheck' take Nbest matches (decided by norm2 distance)
    plot - if plot == 1, plots the matches
    
    output:
    p1 - features in im1
    p2 - features in im2
    p1[j] matches p2[j]
    p1[j]=[[x[j],y[j]]. to extract x[j] out of p1 do p1[j][0][0]
    industry standard to represent the points as it can be an input to cv2.findhomography
    
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    https://www.uio.no/studier/emner/matnat/its/TEK5030/v19/lect/lecture_4_2_feature_matching.pdf
    https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    """
    
    if method.lower() == 'orb':
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1,None)
        kp2, des2 = orb.detectAndCompute(im2,None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

    else:
        # Compute SIFT keypoints and descriptors
        kp1, des1 = pysift.computeKeypointsAndDescriptors(im1)
        kp2, des2 = pysift.computeKeypointsAndDescriptors(im2)

        if method.lower() == 'brute_force_crosscheck':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)
            N=min(Nbest,len(matches))
            good = matches[:N]

        if method.lower() == 'brute_force_knn':
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des1,des2, k=2)

            # Lowe's ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        elif method.lower() == 'flann':
            # Initialize and use FLANN
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # Lowe's ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
        
    p1 = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) #src points
    p2 = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) #dest points
    #p1[j]=[[x[j],y[j]]. to extract x[j] out of p1 do p1[j][0][0]
    
    if plot:
        plotMatches(im1, im2, p1, p2, 'max')
        
    return p1,p2

if __name__ == '__main__':
    print('my_homography')
    im1 = cv2.imread('data/incline_L.png')
    im2 = cv2.imread('data/incline_R.png')
    """
    Your code here
    """
