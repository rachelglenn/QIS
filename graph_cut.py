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


def find_marked_locations(g_img, g_img_s):
    g_img = cv2.cvtColor(io.imread(g_img), cv2.COLOR_RGB2GRAY)
    g_img_s = cv2.cvtColor(io.imread(g_img_s), cv2.COLOR_RGB2GRAY)
    scribble = []
    non_scribble = []
    for i in range(g_img.shape[0]):
        for j in range(g_img.shape[1]):
            if g_img[i,j] !=  g_img_s[i,j]:
                scribble.append((i, j))
            else:
                non_scribble.append((i, j))
    return scribble, non_scribble



def assign_scribble_pixels( img_file, g_img, g_img_s, img_file_scribbled):
   
    img = io.imread(img_file)
    rgb_s = mpimg.imread(img_file_scribbled)[:,:,:3]

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
    
    #scribbles_pos, _ = find_marked_locations(img, img_s)
    scribbles_pos, _ = find_marked_locations(img_file, img_file_scribbled)
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


def compute_pdfs(imfile, imfile_scrib):
    rgb = mpimg.imread(imfile)[:,:,:3]
    yuv = rgb2yiq(rgb)
    rgb_s = mpimg.imread(imfile_scrib)[:,:,:3]
    yuv_s = rgb2yiq(rgb_s)
    # io.imshow(rgb)
    # io.imshow(rgb_s)
        
    # find the scribble pixels    
    #scribbles = find_marked_locations(rgb, rgb_s)
    scribbles, _ = find_marked_locations(imfile, imfile_scrib)
    if not scribbles:
        raise Exception

    imageo = np.zeros(yuv.shape)
    
    # separately store background and foreground scribble pixels in the dictionary comps
    comps = defaultdict(lambda:np.array([]).reshape(0,3))
    for (i, j) in scribbles:
         imageo[i,j,:] = rgb_s[i,j,:]
         # scribble color as key of comps
         comps[tuple(imageo[i,j,:])] = np.vstack([comps[tuple(imageo[i,j,:])], yuv[i,j,:]])
    mu, sigma = {}, {}
    # compute MLE parameters for Gaussians
    for c in comps:
         mu[c] = np.mean(comps[c], axis=0)
         sigma[c] = np.cov(comps[c].T)
    return (mu, sigma)


def spin_ice_J(img_file, g_img):
    row, col = g_img.shape[:2]
    rgb = mpimg.imread(img_file)[:,:,:3]
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

def spin_ice_non_J(rgb_s, g_img,  rgb, non_scribbles_pos, m, s ):


    
    row, col = g_img.shape[:2]
    #print(rgb_s[0,0,:])
    #print("************* ",len(non_scribbles_pos))
    non_s_j = {}
    
    cnt = 0

    mult_factor_lambda = 40

    # Positions for the sink and 
    red = (0.92941177, 0.10980392, 0.14117648)
    blue = (0.24705882, 0.28235295, 0.8)
    for val in m:
        if round(sum(val),2) == 1.33:
            blue = val
        else:
            red = val

    for i, pos in tqdm(enumerate(non_scribbles_pos)):
        
        var0 = ss.multivariate_normal.pdf(rgb[pos[0],pos[1],:], m[blue], s[blue])
        var1 = ss.multivariate_normal.pdf(rgb[pos[0],pos[1],:], m[red], s[red])
        
    
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
        
    return non_s_j
        #print(res0, " : ", res1)
    print("total:{} nane:{}".format(i, cnt))