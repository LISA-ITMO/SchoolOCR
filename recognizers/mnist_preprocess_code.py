from scipy.ndimage import center_of_mass
import math 
import cv2
import numpy as np

def getBestShift(img):
    cy,cx = center_of_mass(img)
    
    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty
  
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted
  
def preprocess_image(img):
  gray = 255-img
  (_, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  
  while np.sum(gray[0]) == 0:
    gray = gray[1:]
  while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)
  while np.sum(gray[-1]) == 0:
    gray = gray[:-1]
  while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)
  rows,cols = gray.shape
  
  if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols*factor))
    gray = cv2.resize(gray, (cols,rows))
  else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))

  colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
  rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
  gray = np.pad(gray,(rowsPadding,colsPadding),'constant')

  shiftx,shifty = getBestShift(gray)
  shifted = shift(gray,shiftx,shifty)
  gray = shifted
  
  img = gray / 255.0
  img = np.array(img).reshape(-1, 28, 28, 1)
  return img