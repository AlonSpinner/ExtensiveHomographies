import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import os
from matplotlib import pyplot as plt
import my_homography as mh
#Add imports if needed:


#end imports

#Add functions here:
def warp2book(im1, H, im2, interpolation, colorspace):
    
    #check image colorspace space 
    if (colorspace == 'LAB'):
        im1 = color.rgb2lab(im1)
        im2 = color.rgb2lab(im2)
        im1 = np.asarray(im1)
        im2 = np.asarray(im2)
    elif(colorspace == 'RGB'):
        im1 = np.asarray(im1)
        im2 = np.asarray(im2)
    elif (colorspace != 'RGB'):
        print('error')
        return

    
    warp_im2 = []
    r_ , c_ = im1.shape[:2]
    
    x = np.arange(0, c_ , 1)
    y = np.arange(0, r_ , 1)
    
    im_run      = im1.transpose(2, 0, 1)
    
    
    r2_, c2_    = im2.shape[:2]
    
    pos_warp    = np.indices([c2_,r2_,1]).reshape((3,-1))
    pos    = pos_warp.astype(np.float32)
    pos[2] += 1
    pos = H @ pos
    pos = np.divide(pos ,pos[2])
    xx, yy = np.meshgrid(x, y)
    out_r, out_c, out_ch = im2.shape[:]
    im2_warp = im2.transpose(2, 1, 0)
    for j,im in enumerate(im_run, 0):
        colorval   = im[yy ,xx]
        f = interpolate.interp2d(x, y, colorval, kind=interpolation)
        for i,k in enumerate(pos_warp.T,0):
            if(pos[0,i]<c_ and pos[0,i]>= 0 and pos[1,i] < r_ and pos[1,i] > 0):
                im2_warp[j,k[0],k[1]] = f(pos[0,i], pos[1,i])
    
    im2_warp = im2_warp.transpose(2,1,0)
    if (colorspace == 'LAB'):
        im2_warp = color.lab2rgb(im2_warp)
    return im2_warp

def im_resize(img,scale_percent = 50):
    #image resize 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    #make resize with interpolation
    img = cv2.resize(img, dim, cv2.INTER_CUBIC)
    return img

def getPoints_SIFT(im1, im2):
    im1    = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    im2    = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    keypoints1, descriptors1 = pysift.computeKeypointsAndDescriptors(im1)
    keypoints2, descriptors2 = pysift.computeKeypointsAndDescriptors(im2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    p1 = []   
    p2 = []
    for m in good:
        p1.append(keypoints1[m.queryIdx].pt)
        p2.append(keypoints2[m.trainIdx].pt)
        
    image_matches = cv2.drawMatches((im1*255).astype(np.uint8),keypoints1,(im2*255).astype(np.uint8),keypoints2,good[:],None,
                                   flags=2)
    plt.imshow(image_matches),plt.show()

    p1 = np.array(p1)
    p2 = np.array(p2)
    return p1,p2

def pl_img_spot(img, points):
    #input: image and identification points
    #define axes and figure to subplots
    fig, ax = plt.subplots(figsize = (10,10))
    #show image 
    ax.imshow(img)
    #sign the identification points
    for point in points.T:
        circ = Circle((point[0], point[1]), 5 , color='red')
        ax.add_patch(circ)
        ax.set_axis_off()
    plt.show()
    plt.tight_layout()
    
def imageimport(img_dir,size = (240,240)):
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        im = Image.open(f1)
        im.resize((size, size))
        data.append(im)
    return data

def s_im2pattern(s,rec_size = 500):
    p_proj = np.array([[0, rec_size, rec_size, 0],[0, 0, rec_size, rec_size]]).astype(np.float32)
    r_, c_ = s[0].shape[:2]
    p_1    = np.array([[0, c_-1, c_-1, 0],[0, 0, r_-1, r_-1]]).astype(np.float32)
    s_proj = []
    M = cv2.getPerspectiveTransform(p_1.T,p_proj.T)
    for i in s:
        s_proj.append(cv2.warpPerspective(i,M,(rec_size,rec_size)).copy())
        
    return s_proj

def image_seq(List, output_path=os.path.join("my_data", "video", "images","vid_im")):
    count = 0
    for i in list:
        fname = str(count+bias).zfill(4)
        cv2.imwrite(os.path.join(output_path, fname + ".jpg"), cv2.cvtColor(np.uint8(i*255), cv2.COLOR_RGB2BGR))  # save frame as JPEG file
        # print('Read a new frame: ', success)
        count += 1
    print("total frames: ", count)
                
def video_to_image_seq(vid_path, output_path=os.path.join("my_data", "video", "images", "vid_im")):
    os.makedirs(output_path, exist_ok=True)
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    print("converting video to frames...")
    while success:
        fname = str(count).zfill(4)
        cv2.imwrite(os.path.join(output_path, fname + ".jpg"), image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
    print("total frames: ", count)

#Functions end

# HW functions:
def create_ref(im_path, rec_size = 500, size = 30):
    im = cv2.imread(im_path)
    im        = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB )
    im     =    im_resize(im.copy(),rec_size)
    
    p_proj = np.array([[0, rec_size, rec_size, 0],[0, 0, rec_size, rec_size]]).astype(np.float32)
    
    
    fig  = plt.figure(figsize=(10,10)) 
    plt.imshow(im)     
    plt.axis('off')
    p1 = fig.ginput(4, timeout = 20)
    plt.close()
    p1 = np.array(p1).astype(np.float32)
    pl_img_spot(im, p1.T)
    M = cv2.getPerspectiveTransform(p1,p_proj.T)
    
    ref_image = cv2.warpPerspective(im,M,(rec_size,rec_size)).copy()
    return ref_image
        
        
def im2im(path1, path2, path3, size):
    im3     =    cv2.imread(path3)
    im3     = cv2.cvtColor(im3.copy(), cv2.COLOR_BGR2RGB )
    im3     =    im_resize(im3.copy(),30)
    im1_ref = create_ref(path1, size)
    im2_ref = create_ref(path2, size)
    p_im3, p_im1_ref = getPoints_SIFT(im3.copy(), im1_ref.copy())
    M, _ = cv2.findHomography(p_im3, p_im1_ref, cv2.RANSAC,5.0)
    warp_ref2image = warp2book(im2_ref, M, im3,'cubic','LAB')
    return im1_ref, im2_ref,im3, warp_ref2image

def my_vid2vid(path1, path2):
    img_List1 = imageimport(path1, 240)
    img_List2 = imageimport(path2, 240)
    
    np_List1 = []
    for i in img_List1:
        np_List1.append(np.asarray(i))
    
    List1_ref = s_im2pattern(np_List1.copy())
        
    np_List2 = []
    for i in img_List2:
        np_List2.append(np.asarray(i))
    
    M_list = []

    ref_im2 =  create_ref(path1 + '0000.jpg')
    for j,i in enumerate(bv_np,0):
        p1,p2 = orb_fun(i,ref_steff)
        M, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
        M_list.append(M)
        if(j%10 == 0):
            ref_steff =  create_ref(path1 + str(j).zfill(4) + ".jpg").copy()
                
    Book_list = []
    n = len(M_list)
    for j,i in enumerate(range(n),0):
        Book_list.append(warp2book(List1_ref[i].copy(), M_list[j].copy(), np_List2[i].copy(),'cubic','LAB'))
    
    image_seq(Book_list)
    image_seq_to_video(os.path.join("my_data", "video", "images","vid_im"), '../output/vid2vid.mp4', fps=30.0)
    

if __name__ == '__main__':
    print('my_ar')