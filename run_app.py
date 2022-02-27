import streamlit as st
import streamlit.components.v1 as stc
from yolo import predict_floating_head_tank
# File Processing Pkgs
import pandas as pd
import os
from PIL import Image
from fastai.vision import open_image
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage import filters
from skimage import exposure
from skimage import measure
from skimage import segmentation
from skimage import morphology
from skimage import color

# https://www.youtube.com/watch?v=UyWEo-q4BGY

def intersection(bb1, bb2):
    '''
    Intersection between two bounding boxes
    '''
    y_min1, x_min1, y_max1, x_max1 = bb1
    y_min2, x_min2, y_max2, x_max2 = bb2
    
    x_left = max(x_min1, x_min2)
    x_right = min(x_max1, x_max2)
    y_top = max(y_min1, y_min2)
    y_bottom = min(y_max1, y_max2)

    intersection = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top+1)
    return intersection


def conv_bbox(box_dict):
    '''
    `conv_bbox` converts bounding boxes from corner coordinates to a 
    `[y_min, x_min, y_max, x_max]` format
    '''
    xs = np.array(list(set([i['x'] for i in box_dict])))
    ys = np.array(list(set([i['y'] for i in box_dict])))
    
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()
    
    return y_min, x_min, y_max, x_max

def check_bb(bbox, shape):
        '''
        The algorithm is designed to work with tanks that are fully in frame. 
        Bounding boxes that reach the edge of an image (
          indicating the tank extends beyond the image) 
          are excluded from processing.
        '''
        c, h, w = shape
        
        for d in bbox:
            if d['x'] <= 2 or d['x'] >= w-2:
                return False
            elif d['y'] <=2 or d['y'] >= h-2:
                return False

        return True 



class Tank():
    def __init__(self, box_dict, image, factor_x=0.5, factor_y=0.6):
        
        self.image = image
        self.gt_coords = box_dict # bounding box coordinates
        y_min, x_min, y_max, x_max = self.gt_coords
        
        # scale for tank cropping
        margin_x = int((x_max-x_min)*factor_x)
        margin_y = int((y_max-y_min)*factor_y)

        # y_min, y_max, x_min, x_max values for cropping
        self.y_min = max(y_min - margin_y, 0)
        self.y_max = max(y_max + int(margin_y//2), 0)
        self.x_min = max(x_min - margin_x, 0)
        self.x_max = max(x_max + margin_x, 0)
        
        # actual margins, given that the calculated margin might extend beyond the image
        margin_y_true = y_min - self.y_min
        margin_x_true = x_min - self.x_min
        
        # coordinates of the actual bounding box relative to the crop box
        self.bbox_relative = [margin_y_true, margin_x_true, (y_max-y_min)+margin_y_true, (x_max-x_min)+margin_x_true]

        # crop section of the image
        self.tank_crop = self.image.data[:, self.y_min:self.y_max, self.x_min:self.x_max].permute(1,2,0).numpy()
        
        self.proc_tank()
        self.get_regions()
        
    def proc_tank(self):
        # HSV conversion
        hsv = color.rgb2hsv(self.tank_crop)
        H = hsv[:,:,0]
        S = hsv[:,:,1]
        V = hsv[:,:,2]
        
        # LAB conversion
        lab = color.rgb2lab(self.tank_crop)
        l1 = lab[:,:,0]
        l2 = lab[:,:,1]
        l3 = lab[:,:,2]
        
        # Enhanced image
        self.tank_hsv = -(l1+l3)/(V+1)
        
        # Threshold values
        t1 = filters.threshold_minimum(self.tank_hsv)
        t2 = filters.threshold_mean(self.tank_hsv)
        
        # Thresholding
        self.tank_thresh = self.tank_hsv > (0.3*t1 + 0.5*t2)
        
        # Processed, labeled image
        self.label_image = measure.label(morphology.area_closing(morphology.closing(
            segmentation.clear_border(filters.hessian(self.tank_thresh)))))
        
    def get_regions(self):
        # Regions within image
        self.regions_all = measure.regionprops(self.label_image)
        
        self.regions = []
        
        # Some regions are noise. This ensures that regions have a decent area ( > 25 px),
        # that the region intersects the boudning box around the tank (removes lots of noisy features)
        # and that the processed region is also present in the thresholded image (the hessian filter can sometimes 
        # add artifacts that need to be removed this day)
        for region in self.regions_all:
            if intersection(self.bbox_relative, region.bbox) > 300:
                if region.area > 25:
                    b = region.bbox
                    if abs(self.tank_thresh[b[0]:b[2], b[1]:b[3]].mean() - region.image.mean()) < 0.06:
                        self.regions.append(region)
                            
        # areas of all regions
        areas = np.array([i.area for i in self.regions])
        
        # if there are more than two areas found, take the two largest
        # 1 - ratio of the two largest areas calculates the volume estimation
        if len(areas) > 1:
            idx2, idx1 = areas.argsort()[-2:]
            self.volume = 1 - self.regions[idx2].area / self.regions[idx1].area
        # if only 1 area is found, tank is assumed to be full
        else:
            idx2 = 0
            idx1 = 0
            self.volume = 1
        
        # Blank image onto which to paste only the two shadow regions
        self.blank = np.zeros(self.tank_crop.shape[:2])
        
        for region in [self.regions[idx1], self.regions[idx2]]:
            y_min, x_min, y_max, x_max = region.bbox
            self.blank[y_min:y_max, x_min:x_max] += region.image.astype('uint8')
                        
        # get contours of shadows
        self.contours = measure.find_contours(self.blank, 0.5)
        if len(self.contours) > 1:
            # If there are multiple contours, take the two longest
            contour_idxs = np.array([len(i) for i in self.contours]).argsort()[-2:]
        else:
            contour_idxs = [0]
        self.contours_select = [self.contours[i] for i in contour_idxs]            
    
    def plot_tank(self):
        fig, axes = plt.subplots(3,3, figsize=(12,12))

        fig.suptitle('Tank Volume {:.3f}%'.format(self.volume*100))

        axes[0][0].imshow(self.tank_crop)
        axes[0][0].set_title('Tank Crop')

        axes[0][1].imshow(self.tank_crop)
        axes[0][1].imshow(self.blank, alpha=0.5)
        axes[0][1].set_title('Shadow Overlay')

        axes[0][2].imshow(self.tank_crop)
        for cnt in self.contours_select:
            axes[0][2].plot(cnt[:,1], cnt[:,0])
        axes[0][2].set_title('Shadow Contour')

        axes[1][0].imshow(self.blank)
        axes[1][0].set_title('Shadow')

        axes[1][1].imshow(np.zeros(self.blank.shape))
        for cnt in self.contours:
            axes[1][1].plot(cnt[:,1], cnt[:,0])
        axes[1][1].set_title('All Contours')

        axes[1][2].imshow(np.zeros(self.blank.shape))
        for cnt in self.contours_select:
            axes[1][2].plot(cnt[:,1], cnt[:,0])
        axes[1][2].set_title('Major Contours')

        axes[2][0].imshow(self.tank_hsv)
        axes[2][0].set_title('HSV Ratio')

        axes[2][1].imshow(self.tank_thresh)
        axes[2][1].set_title('Tank Thresholding')

        axes[2][2].imshow(self.label_image)
        axes[2][2].set_title('Morphology Labeling')

        for ax in axes.flat:
            ax.axis('off')
        return fig,ax    


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    st.title("File Upload ")

    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        image_file = st.file_uploader(
            "Upload Image", type=['png', 'jpeg', 'jpg'])
        if image_file is not None:

            # To See Details
            # st.write(type(image_file))
            # st.write(dir(image_file))
            file_details = {"Filename": image_file.name,
                            "FileType": image_file.type, "FileSize": image_file.size}
            st.write(file_details)
            img = load_image(image_file)
            if not os.path.exists('tempDir'):
                os.mkdir('tempDir')
            lat_dir = os.path.join(os.getcwd(),'tempDir')
            with open(os.path.join(lat_dir,image_file.name),"wb") as f: 
                f.write(image_file.getbuffer())         
            st.success("Saved File")
            complete_path = os.path.join(lat_dir,image_file.name)
            bb_box = predict_floating_head_tank(complete_path)
            image_path = complete_path
            if len(bb_box)>0:
                box_dict = bb_box[0]
                img = open_image(image_path)
                obj = Tank(box_dict,img)
                fig,ax = obj.plot_tank()
                st.pyplot(fig)
            else:
                st.write("No Floating head Tank  detected in image")
            #st.image(obj.plot_tank())

    if choice == 'About':
        st.write("Source code is at https://github.com/shekhar1678/cv")
        st.write("First execute create_data.py internally to create dataset")
        st.write("create_data.py is used to create folder tank_image ")


if __name__ == '__main__':
    main()
