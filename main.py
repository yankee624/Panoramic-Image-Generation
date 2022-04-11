import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    """
    
    nbrs1 = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(des1)
    distances2, indices2 = nbrs1.kneighbors(des2)

    nbrs2 = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(des2)
    distances1, indices1 = nbrs2.kneighbors(des1)

    RATIO_TEST_THRESH = 0.5
    # ratio test
    filter1 = [i for i in range(len(indices1)) if distances1[i][0] / distances1[i][1] < RATIO_TEST_THRESH]
    filter2 = [i for i in range(len(indices2)) if distances2[i][0] / distances2[i][1] < RATIO_TEST_THRESH]

    # bi-directional consistency
    filter1 = [i for i in filter1 if (indices1[i][0] in filter2) and indices2[indices1[i][0]][0] == i]
    filter2 = [indices1[i][0] for i in filter1]    
    return loc1[filter1], loc2[filter2]


def EstimateH(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the homography between images using RANSAC
    
    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """

    best_inlier = []
    best_H = None
    n = 4
    for _ in range(ransac_n_iter):
        idx = np.random.choice(x1.shape[0], n, replace=False)
        x1_samples = x1[idx]
        x2_samples = x2[idx]
        A = []
        for i in range(n):
            _x, _y = x1_samples[i]
            _xp, _yp = x2_samples[i]
            A.append([_x, _y, 1, 0, 0, 0, -_xp*_x, -_xp*_y, -_xp])
            A.append([0, 0, 0, _x, _y, 1, -_yp*_x, -_yp*_y, -_yp])
        A = np.array(A)
        U, S, VT = np.linalg.svd(A)
        H = VT[-1].reshape(3,3)
        inlier = []
        for i in range(len(x1)):
            x1_sample, x2_sample = x1[i], x2[i]
            x2_pred = H.dot(np.array([x1_sample[0], x1_sample[1], 1]))
            x2_pred /= x2_pred[-1]
            x2_pred = x2_pred[:2]
            if np.linalg.norm(x2_sample - x2_pred) <= ransac_thr:
                inlier.append(i)
        if len(inlier) > len(best_inlier):
            best_inlier = inlier
            best_H = H
    return best_H, best_inlier

def EstimateR(H, K):
    """
    Compute the relative rotation matrix
    
    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """
    
    R = np.linalg.inv(K).dot(H).dot(K)
    U, S, VT = np.linalg.svd(R)
    _R = U.dot(VT)
    return np.sign(np.linalg.det(_R)) * _R

def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface
    
    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Hc, Hc, 3)
        The 3D points corresponding to all pixels in the canvas
    """

    f = K[0][0]
    w = np.linspace(0, Wc-1, Wc)
    h = np.linspace(0, Hc-1, Hc)
    ww, hh = np.meshgrid(w, h)

    phi = ww*2*np.pi/Wc
    x = (f*np.sin(phi))
    y = (hh-Hc/2)
    z = (f*np.cos(phi))
    p = np.stack((x, y, z), axis=2)
    return p



def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane
    
    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """
    
    rotated_points = np.einsum("ij,hwj->hwi",R,p)
    projected_points = np.einsum("ij,hwj->hwi",K,rotated_points)
    projected_points /= projected_points[:,:,2:]
    u = projected_points[:,:,:2]
    mask = (u[:,:,0] >= 0) & (u[:,:,0] < W) & (u[:,:,1] >= 0) & (u[:,:,1] < H) & (rotated_points[:,:,2] > 0)

    # Hc, Wc, _ = p.shape
    # u = np.zeros((Hc,Wc,2))
    # mask = np.zeros((Hc,Wc))
    # for h in range(Hc):
    #     for w in range(Wc):
    #         rotated_point = R.dot(p[h][w])
    #         projected_point = K.dot(rotated_point)
    #         projected_point /= projected_point[2]
    #         u[h][w][0], u[h][w][1] = projected_point[0], projected_point[1]
    #         if rotated_point[2] > 0 and u[h][w][0] >= 0 and u[h][w][0] < W and u[h][w][1] >= 0 and u[h][w][1] < H:
    #             mask[h][w] = 1
    return u, mask


def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas
    
    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """

    # https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    def bilinear_interpolate(im, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1]-1)
        x1 = np.clip(x1, 0, im.shape[1]-1)
        y0 = np.clip(y0, 0, im.shape[0]-1)
        y1 = np.clip(y1, 0, im.shape[0]-1)

        Ia = im[ y0, x0 ]
        Ib = im[ y1, x0 ]
        Ic = im[ y0, x1 ]
        Id = im[ y1, x1 ]

        wa = ((x1-x) * (y1-y)).reshape(-1,1)
        wb = ((x1-x) * (y-y0)).reshape(-1,1)
        wc = ((x-x0) * (y1-y)).reshape(-1,1)
        wd = ((x-x0) * (y-y0)).reshape(-1,1)

        return wa*Ia + wb*Ib + wc*Ic + wd*Id
    
    Hc = u.shape[0]
    Wc = u.shape[1]
    canvas_i = np.zeros((Hc, Wc, 3))

    ux = u[:,:,0].flatten()
    uy = u[:,:,1].flatten()
    canvas_i = bilinear_interpolate(image_i, ux, uy).reshape(Hc, Wc, 3)
    canvas_i = canvas_i * mask_i[..., np.newaxis]
    canvas_i = canvas_i.astype(np.uint8)
    return canvas_i

def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image
    
    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """
    
    valid_indices = np.argwhere(mask_i == 1)
    valid_h = valid_indices[:, 0]
    valid_w = valid_indices[:, 1]
    canvas[valid_h, valid_w, :] = canvas_i[valid_h, valid_w, :]

    # Hc = canvas.shape[0]
    # Wc = canvas.shape[1]
    # for h in range(Hc):
    #     for w in range(Wc):
    #         if mask_i[h][w]==1:
    #             canvas[h][w] = canvas_i[h][w]
    
    return canvas



if __name__ == '__main__':
    ransac_n_iter = 500
    ransac_thr = 3
    K = np.asarray([
        [320, 0, 480],
        [0, 320, 270],
        [0, 0, 1]
    ])

    sift = cv2.SIFT_create()
    # Read all images
    im_list = []
    for i in range(1, 9):
        im_file = 'img/{}.jpg'.format(i)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_list.append(im)

    rot_list = []
    rot_list.append(np.eye(3))
    for i in range(len(im_list) - 1):
        # Load consecutive images I_i and I_{i+1}
        img1 = im_list[i]
        img2 = im_list[i+1]
		
        # Extract SIFT features
        kp1, des1 = sift.detectAndCompute(img1,None)
        loc1 = np.array([kp.pt for kp in kp1])
        kp2, des2 = sift.detectAndCompute(img2,None)
        loc2 = np.array([kp.pt for kp in kp2])

        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2)

        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)

        # Compute the relative rotation matrix R
        R = EstimateR(H, K)
		
		# Compute R_new (or R_i+1)
        R_new = R.dot(rot_list[-1])
        rot_list.append(R_new)

    Him = im_list[0].shape[0]
    Wim = im_list[0].shape[1]
    
    Hc = Him
    Wc = len(im_list) * Wim // 2
	
    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)

    fig = plt.figure('HW1')
    plt.axis('off')
    plt.ion()
    plt.show()
    for i, (im_i, rot_i) in enumerate(zip(im_list, rot_list)):
        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)
        # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)
        # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)
        plt.imshow(canvas)
        plt.savefig('img/output_{}.png'.format(i+1), dpi=600, bbox_inches = 'tight', pad_inches = 0)