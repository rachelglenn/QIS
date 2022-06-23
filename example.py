from pickle import FALSE, TRUE
import cv2
from skimage import io
from graph_cut import *
import scipy.stats as ss 
import math
import neal
import matplotlib
import matplotlib.pyplot as plt

# Files
img_file = "test2.png"
img_file_scribbled = "test2_s.png"
color = TRUE
# img_file = "original.png"
# img_file_scribbled = "marked.png"
# color = FALSE




if color:


    # Read in images
    img = io.imread(img_file)
    img_s = io.imread(img_file_scribbled)
    print(img.dtype, img.shape)

    # Convert to gray scale
    g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g_img_s = cv2.cvtColor(img_s, cv2.COLOR_RGB2GRAY)
    rgb_s = mpimg.imread(img_file_scribbled)[:,:,:3]
    rgb = mpimg.imread(img_file)[:,:,:3]
else:

    # Read in images
    img =   cv2.cvtColor(io.imread(img_file),cv2.COLOR_GRAY2RGB)
    img_s = io.imread(img_file_scribbled)
    print(img.dtype, img.shape)

    # Convert to gray scale
    g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g_img_s = cv2.cvtColor(img_s, cv2.COLOR_RGB2GRAY)
    rgb_s =  cv2.cvtColor(img_s, cv2.COLOR_RGB2RGBA)[:,:,:3]
    rgb = mpimg.imread(img_s)[:,:,:3]


# Show images
if False:
    io.imshow(img)
    matplotlib.pyplot.show()
    io.imshow(img_s)
    matplotlib.pyplot.show()



# Compute the probablitity density function
m, s = compute_pdfs(rgb, rgb_s, img, img_s)

for val in s:
    s[val] = s[val] * 2

# Constructing Spin/Ising Model BQM
print("Getting marked/nonmarked locations...")
scribbled_pos, non_scribbles_pos = find_marked_locations(img, img_s)

print(len(non_scribbles_pos), "c = ",rgb.shape[0]*rgb.shape[1])

# Positions for the sink and source
red = (0.92941177, 0.10980392, 0.14117648)
blue = (0.24705882, 0.28235295, 0.8)

h = {'s': -1, 't': 1}
print("Developing graph for spin-ice...")
print("Progress")
J = spin_ice_J(rgb, g_img)
non_s_j = spin_ice_non_J(rgb_s, g_img, rgb,  non_scribbles_pos, m, s, red, blue )


s_J = {}
res_J = dict(s_J)
res_J.update(non_s_j)
res_J.update(J)
#print(len(res_J))

print("\nSimulated Annealing...")
qpu = neal.SimulatedAnnealingSampler()
result = qpu.sample_ising(h=h, J=res_J, num_reads=10)

print("Processing image...")
sol = result.first[0].copy()
del sol['s']
del sol['t']
sol = dict(sorted(sol.items())).copy()
result_img = np.array(list(sol.values()))


result_img = result_img.reshape(rgb.shape[:2])
final_img = np.where(result_img < 0, result_img, 0)
final_img_g = final_img * g_img

# Convert to uint8
info = np.iinfo(final_img.dtype) # Get the information of the incoming image type
data = final_img.astype(np.float64) / info.max # normalize the data to 0 - 1
data = 255 * data # Now scale by 255
img = data.astype(np.uint8)

io.imshow(img, cmap='gray')
matplotlib.pyplot.show()

io.imsave(img_file.split('.')[0]+'_res.png', img)