import cv2
from skimage import io
from graph_cut import *
import scipy.stats as ss 
import math
import neal


# Files
img_file = "test2.png"
img_file_scribbled = "test2_s.png"

# Read in images
img = io.imread(img_file)
img_s = io.imread(img_file_scribbled)
print(img.dtype, img.shape)

# Convert to gray scale
g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
g_img_s = cv2.cvtColor(img_s, cv2.COLOR_RGB2GRAY)

# Show images
io.imshow(img)
io.imshow(img_s)
io.imshow(g_img)
io.imshow(g_img_s)

# Show difference
diff = g_img - g_img_s
io.imshow(diff)
io.imsave('diff.png', diff)



scribbles_pos_dict = assign_scribble_pixels(img_file, g_img, g_img_s, img_file_scribbled)


m, s = compute_pdfs(img_file, img_file_scribbled)

for val in s:
    s[val] = s[val] * 2
rgb = mpimg.imread(img_file)[:,:,:3]
rgb_s = mpimg.imread(img_file_scribbled)[:,:,:3]
pixel_val = rgb_s[0,0,:]

k1, k2 = list(scribbles_pos_dict.keys())
var0 = ss.multivariate_normal.pdf(pixel_val,m[k1], s[k1])
var1 = ss.multivariate_normal.pdf(pixel_val, m[k2], s[k2])

var2 = var0 + var1
print(var1, ": ",var2)
print(var0/var2, var1/var2) # Source and Sink Weights 

# Constructing Spin/Ising Model BQM

scribbled_pos, non_scribbles_pos = find_marked_locations(img_file, img_file_scribbled)

# print(len(non_scribbles_pos), "c = ",r*c)


h = {'s': -1, 't': 1}
rgb_s = mpimg.imread(img_file_scribbled)[:,:,:3]
rgb = mpimg.imread(img_file)[:,:,:3]


J = spin_ice_J(img_file, g_img)

non_s_j = spin_ice_non_J(rgb_s, g_img, rgb,  non_scribbles_pos, m, s )


s_J = {}
res_J = dict(s_J)
res_J.update(non_s_j)
res_J.update(J)
print(len(res_J))

print("\nSimulated Annealing....")

qpu = neal.SimulatedAnnealingSampler()
result = qpu.sample_ising(h=h, J=res_J, num_reads=10)


sol = result.first[0].copy()
del sol['s']
del sol['t']
sol = dict(sorted(sol.items())).copy()
result_img = np.array(list(sol.values()))
result_img.shape
cnt = 0
for i in range(result_img.shape[0]):
    if result_img[i] == -1:
       cnt +=1
 
print(cnt)
print(result_img.shape)
result.first[0]['t']
result_img = result_img.reshape(rgb.shape[:2])


final_img = np.where(result_img < 0, result_img, 0)

print("final_img", final_img.shape)
print("g_img", g_img.shape)

final_img_g = final_img * g_img

io.imshow(final_img, cmap='gray')
result.first[0]['t']
io.imsave(img_file.split('.')[0]+'_res.png', final_img)