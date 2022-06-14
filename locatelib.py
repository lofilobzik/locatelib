import cv2
import numpy as np
import requests

def get_image(ip='127.0.0.1', filename='image.jpg') -> np.ndarray:
    """Get image from ip

    Args:
        ip (str): ip address
        filename (str): name of the file

    Returns:
        np.ndarray: image
    """
    url = f'http://{ip}/{filename}'
    resp = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    
    return cv2.imdecode(image, -1)

def straighten_image(image: np.ndarray, aruco_dict: cv2.aruco.Dictionary, dim=(640, 480)) -> np.ndarray:
    """Returns straightened image using aruco markers and homography.
    Returns None if cannot find all the aruco's

    Args:
        image (np.ndarray): Image to straighten
        aruco_dict (cv2.aruco.Dictionary): Result of cv2.aruco.getPredefinedDictionary
        dim (tuple, optional): Resulting image dimensions. Defaults to (640, 480)

    Returns:
        np.ndarray: straightened image
    """
    (all_corners, ids, rejected) = cv2.aruco.detectMarkers(image, aruco_dict)
    
    pts_dst = np.array([[0, 0], [dim[0], 0], [dim[0], dim[1]], [0, dim[1]]])
    pts_src = np.zeros([4, 2])
    
    if ids is not None and len(ids) == 4 and sum(ids) == 6:
        for i, id in enumerate(ids):
            corners, id = all_corners[i].astype('int32')[0], id[0]    
            pts_src[id] = corners[id]

        h, _ = cv2.findHomography(pts_src, pts_dst)
        trg_image = cv2.warpPerspective(image, h, (dim[0], dim[1]))
        
        return trg_image
    
    return None

def find_bin_mask_difference(image: np.ndarray, back_image: np.ndarray, min_white=50) -> np.ndarray:
    """Find binary mask. cv2.absdiff version

    Args:
        image (np.ndarray): Image
        back_image (np.ndarray): Background image (straightened image without objects)
        min_white (int, optional): Minimal white threshold. Defaults to 50.

    Returns:
        np.ndarray: Binary mask
    """
    diff = back_image.copy()
    diff = cv2.absdiff(back_image, image, diff)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, bin_image = cv2.threshold(gray, min_white, 255, cv2.THRESH_BINARY)
    # bin_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    kernel = np.ones((5, 5), 'uint8')
    bin_image = cv2.erode(bin_image, kernel, iterations=2)
    bin_image = cv2.dilate(bin_image, kernel, iterations=2)    
    
    return bin_image

def find_bin_mask_hsv(image: np.ndarray) -> np.ndarray:
    """Find binary mask. cv2.absdiff version

    Args:
        image (np.ndarray): Image

    Returns:
        np.ndarray: Binary mask
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bin_image = cv2.inRange(hsv[:,:,0],0,70)
    kernel = np.ones((5, 5), 'uint8')
    bin_image = cv2.erode(bin_image, kernel, iterations=10)
    bin_image = cv2.dilate(bin_image, kernel, iterations=10)
    
    return bin_image

def get_coords(bin_image: np.ndarray, x_size_mm: int, y_size_mm: int) -> tuple[list, list]:
    """Get center coordinates (in mm, on paper) of objects in a binary mask

    Args:
        bin_image (np.ndarray): Binary mask
        x_size_mm (int): Field width
        y_size_mm (int): Fiels height

    Returns:
        tuple[list, list]: Centers and frame pixel coordinates
    """
    contours, hierarchy = cv2.findContours(image=bin_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    centers = []
    frames = []
    height, width = bin_image.shape
    
    for c in contours:
            area = cv2.contourArea(c)
            
            if area > 1000:
                x, y, w, h = cv2.boundingRect(c)
                
                x_center = int(x + w / 2)
                y_center = int(y + h / 2)
                
                x_mm = x_size_mm * x_center // width
                y_mm = y_size_mm - (y_size_mm * y_center // height)
                
                frames.append(((x, y, w, h), (x_center, y_center)))
                
                centers.append((x_mm, y_mm))
                
    return centers, frames

def get_robot_coords(coords) -> list:
    """PLEASE CHANGE THIS FUNCTION FOR YOUR NEEDS
    Converts paper coords to robot coords

    Args:
        coords (list): Paper coords

    Returns:
        list: Robot coords
    """
    offset_x = -143
    offset_y = 300
    
    robot_coords = []
    for c in coords:
        x_r = c[0] + offset_x
        y_r = c[1] + offset_y
        
        x_r, y_r = -y_r, x_r
        robot_coords.append((x_r, y_r))
    
    return robot_coords

if __name__ == '__main__':
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    back_image = cv2.imread('bg.png')
    while True:
        image = get_image(ip='77.37.184.204')
        straight_image = straighten_image(image, aruco_dict)
        
        if straight_image is None:
            cv2.imshow('image', image)
        else:
            bin_mask = find_bin_mask_difference(straight_image, back_image)
            coords, _ = get_coords(bin_mask, 287, 200)
            coords = get_robot_coords(coords)
            print(coords)
            cv2.imshow('image', bin_mask)
    
        if cv2.waitKey(100) == 27:
            break
        
    cv2.destroyAllWindows()
        