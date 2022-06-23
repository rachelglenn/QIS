import matplotlib.image as mpimg
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2yiq
from collections import defaultdict 
import cv2
from tqdm import tqdm
import dimod
import math
import scipy.stats as ss 
import matplotlib.pyplot as plt

def find_marked_locations(img, img_s):
    #g_img = cv2.cvtColor(io.imread(g_img), cv2.COLOR_RGB2GRAY)
    #g_img_s = cv2.cvtColor(io.imread(g_img_s), cv2.COLOR_RGB2GRAY)
    g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g_img_s = cv2.cvtColor(img_s, cv2.COLOR_RGB2GRAY)
    scribble = []
    non_scribble = []
    for i in range(g_img.shape[0]):
        for j in range(g_img.shape[1]):
            if g_img[i,j] !=  g_img_s[i,j]:
                scribble.append((i, j))
            else:
                non_scribble.append((i, j))
    return scribble, non_scribble



def assign_scribble_pixels( img, img_s, g_img, g_img_s, rgb_s):
   
    #img = io.imread(img_file)
    #rgb_s = mpimg.imread(img_file_scribbled)[:,:,:3]
    img = img

    scribble = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if g_img[i,j] !=  g_img_s[i,j]:
                scribble.append((i,j))
                
    img_s_key = set()
    
    yuv_s = rgb2yiq(rgb_s)
    for i, j in scribble:
        img_s_key.add(tuple(rgb_s[i,j,:3]))
    # img_s_key # should only have two values
    
    scribbles_pos, _ = find_marked_locations(img, img_s)
    #scribbles_pos, _ = find_marked_locations(img_file, img_file_scribbled)
    scribbles_pos_dict = defaultdict(list)

    for pos in scribbles_pos:
        if tuple(rgb_s[pos[0], pos[1]]) in img_s_key:
            scribbles_pos_dict[tuple(rgb_s[pos[0], pos[1]])].append(pos)
    
    # Sanity Check
    # Ensuring only two colors representing either foreground or background. 
    #print(len(scribbles_pos_dict))
    l = 0
    for k in scribbles_pos_dict:
        l += len(scribbles_pos_dict[k])
    
    if l == len(scribbles_pos):
        print('OK')
    # ------------
    return scribbles_pos_dict


def compute_pdfs(rgb, rgb_s, img, img_s):
    #rgb = mpimg.imread(imfile)[:,:,:3]
    yuv = rgb2yiq(rgb)
    #rgb_s = mpimg.imread(imfile_scrib)[:,:,:3]
    yuv_s = rgb2yiq(rgb_s)
    # io.imshow(rgb)
    # io.imshow(rgb_s)
        
    # find the scribble pixels    
    scribbles, _ = find_marked_locations(img, img_s)
    #scribbles, _ = find_marked_locations(imfile, imfile_scrib)
    if not scribbles:
        raise Exception

    imageo = np.zeros(yuv.shape)
    
    # separately store background and foreground scribble pixels in the dictionary comps
    comps = defaultdict(lambda:np.array([]).reshape(0,3))
    #print("dictionary", comps)
    for (i, j) in scribbles:
         imageo[i,j,:] = rgb_s[i,j,:]
         # scribble color as key of comps
         comps[tuple(imageo[i,j,:])] = np.vstack([comps[tuple(imageo[i,j,:])], yuv[i,j,:]])
         #print((imageo[i,j,:]))
    mu, sigma = {}, {}
    #print(comps.keys())
    # compute MLE parameters for Gaussians
    sig = 0.005
    for c in comps:
         mu[c] = np.mean(comps[c], axis=0)
         sigma[c] = np.cov(comps[c].T,)
         #print(sigma[c])
         #sigma[c] = np.array([[sig, sig*0.4, sig*0.6], [sig, sig*0.3, sig*0.7], [sig, sig*3, sig*0.4] ])
         #sigma[c] =0.000005
    #print(sigma)
    # sigma = np.array([[ 0.02057979,  0.00381766, -0.00635883],\
    #     [ 0.00381766,  0.0030234,  -0.00179508],\
    #     [-0.00635883, -0.00179508,  0.00228135]],\
    #     [[ 0.00535723, -0.00165761,  0.00094871],\
    #     [-0.00165761,  0.00145263, -0.00064738],\
    #     [ 0.00094871, -0.00064738,  0.00034484]])
    return (mu, sigma)


def spin_ice_J(rgb, g_img):
    row, col = g_img.shape[:2]
    #rgb = mpimg.imread(img_file)[:,:,:3]
    std = np.std(g_img)

    rgb_t = np.reshape(rgb.T, (3,-1)) # images.shape = (3,5120000)
    std = np.std(rgb_t, axis = 1)

    
    J = {}

    '''for i in enumerate(r):
        for j in enumerate(c):
            pass
            # # Just putting Intensity for now
            # h[i*c + j] = pxij
            #h[i*c + j] = 0
    '''
    for i in range(row):
        for j in range(col):
            # print(i , j)
            res = 0
            if i-1 >= 0:
                #Up
                a, b = i*col + j, (i-1)*col + j
                # print('UP:', a, b)
                Ia, Ib = rgb[i, j, :], rgb[i-1, j, : ]
                power = ((Ia[0] -Ib[0])**2 / (2*(std[0]**2))) + ((Ia[1] -Ib[1])**2 / (2*(std[1]**2))) + ((Ia[2] -Ib[2])**2 / \
                (2*(std[2]**2)))
                res = math.exp(-power)
                J[(a, b)] = -res
            if i+1 < row:
                #Down
                a, b = i*col + j, (i+1)*col + j
                # print('DOWN:', a, b)
                Ia, Ib = rgb[i, j, :], rgb[i+1, j, : ]
                power = ((Ia[0] -Ib[0])**2 / (2*(std[0]**2))) + ((Ia[1] -Ib[1])**2 / (2*(std[1]**2))) + ((Ia[2] -Ib[2])**2 / \
                (2*(std[2]**2)))
                res = math.exp(-power)
                J[(a, b)] = -res
            if j-1 >= 0:
                #Left
                a, b = i*col + j, i*col + (j-1)
                # print('LEFT:', a, b)
                Ia, Ib = rgb[i, j, :], rgb[i, j-1, : ]
                power = ((Ia[0] -Ib[0])**2 / (2*(std[0]**2))) + ((Ia[1] -Ib[1])**2 / (2*(std[1]**2))) + ((Ia[2] -Ib[2])**2 / \
                (2*(std[2]**2)))
                res = math.exp(-power)

                J[(a, b)] = -res
            if j+1 < col:
                #Right
                a, b = i*col + j, i*col + (j+1)
                # print('RIGHT:', a, b)
                Ia, Ib = rgb[i, j, :], rgb[i, j+1, : ]
                power = ((Ia[0] -Ib[0])**2 / (2*(std[0]**2))) + ((Ia[1] -Ib[1])**2 / (2*(std[1]**2))) + ((Ia[2] -Ib[2])**2 / \
                (2*(std[2]**2)))
                res = math.exp(-power)
                J[(a, b)] = -res
    return J

def spin_ice_non_J(rgb_s, g_img,  rgb, non_scribbles_pos, m, s, red, blue ):


    
    row, col = g_img.shape[:2]
    #print(rgb_s[0,0,:])
    #print("************* ",len(non_scribbles_pos))
    non_s_j = {}
    
    cnt = 0

    mult_factor_lambda = 40


    for val in m:
        if round(sum(val),2) == 1.33:
            blue = val
        else:
            red = val
    plotData = np.zeros((rgb.shape[0], rgb.shape[1]))
    for i, pos in tqdm(enumerate(non_scribbles_pos)):
        
        var0 = ss.multivariate_normal.pdf(rgb[pos[0],pos[1],:], m[blue], s[blue])
        var1 = ss.multivariate_normal.pdf(rgb[pos[0],pos[1],:], m[red], s[red])
        plotData[pos[0], pos[1]] = var1
    
        #print("var0: {} var1:{} res0:{}, res1:{} ".format(var0, var1, res0, res1))
        import math
        if var0 == 0 and var1 == 0:
            res0 = .5
            res1 = 99
        
            non_s_j[(pos[0]*col + pos[1], 's')] = -res0
            non_s_j[(pos[0]*col + pos[1], 't')] = -res1
            cnt +=1
        else:
            var0 += .001
            var1 += .001
            res = var0 + var1
            res0 = var0
            res1 = var1
            non_s_j[(pos[0]*col + pos[1], 's')] = - (mult_factor_lambda * np.log(res0))
            non_s_j[(pos[0]*col + pos[1], 't')] = - (mult_factor_lambda * np.log(res1))
            
        #print("******** got var1: {}, var2: {}".format(var0, var1))
    
    if True:
        x = range(plotData.shape[0])
        y = range(plotData.shape[1])
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.contourf(y, x, plotData)
        plt.show()
    return non_s_j
