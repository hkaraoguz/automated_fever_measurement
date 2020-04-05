# -----------------------------------------------------------
# Performs head pose detection and relative temperature measurement
# for detected people in thermal images
# (C) 2020 Hakan Karaoguz, Stockholm, Sweden
# Released under MIT License
# email karaoguzh@gmail.com
# -----------------------------------------------------------
import pickle
import numpy as np
import os
import cv2


def calculate_head_poses(img, person_boxes):
    '''
    Reliably calculate head pose for detected people
    '''
    reliable_people = {}
    count = 0
    for person_box in person_boxes:
        
        # Calculate box center
        box_center = np.zeros((2))
        box_center[0] = round((person_box[0] + person_box[2]) /2)
        box_center[1] = round((person_box[1] + person_box[3]) /2)
        
        # Calculate the ROI for head pose calculation
        xmin = int(person_box[0]+ (box_center[0]-person_box[0])*0.9)
        xmax = int(person_box[2]- (person_box[2]-box_center[0])*0.9)
        
        height1 = int(person_box[1] + 0.1*(person_box[3]-person_box[1]))
        height2 = int(person_box[1] + 0.09*(person_box[3]-person_box[1]))
        height3 = int(person_box[1] + 0.11*(person_box[3]-person_box[1]))
        
        heights = [height1, height2, height3]

        head_pixels = []
        head_pose = None
        
        # Scan the ROI to find the reliable pixels that belong to head
        for height in heights:
            
            for k in range(xmin, xmax-4):
                
                val = img[height, k][0]
    
                head_part = True
                for j in range(k+1, k+4):
                    new_val = img[height, j][0]
                    #print(new_val)
                    if abs(val-new_val) > 5:
                        head_part = False
                        break
                if head_part:
                    head_pixels.append([k,height])
        
        # if any head pixel is available, calculate head pose
        if head_pixels:
            head_pixels = np.asarray(head_pixels)
            head_pose = np.mean(head_pixels, axis=0).astype(int)
            
            reliable_person = {}
            reliable_person['person_box'] = person_box
            reliable_person['head_pose'] = head_pose
            reliable_person['reference'] = img[head_pose[1], head_pose[0]][0]
            
            reliable_people[count] = reliable_person
            count += 1
    
    return reliable_people

def visualize_temperature_measurements(person_dicts,save_processed_images = False):
    """
    Visualizes relative temperature measurements
    for reliably detected people
    """

    for key, person_boxes in person_dicts.items():
        
        im = cv2.imread(key)
        org_im = im.copy()

        # Get reliably detected head poses for accurate temperature measurement
        reliable_people = calculate_head_poses(org_im, person_boxes)
        
        references = []

        # If there is more than one reliable person 
        if len(reliable_people.items()) > 1:
            
            # Get the minumum pixel value and set it as 36.5
            for _, value in reliable_people.items():
                
                references.append(value['reference'])
            
            references_np = np.asarray(references)

            min_reference_pixel = np.min(references_np)
            min_reference_temp = 36.5
            
            # For each reliable person, perform temperature computation
            for _, value in reliable_people.items():

                person_box = value['person_box']
                head_pose = value['head_pose']
                head_val = value['reference']

                cv2.rectangle(im,(person_box[0],person_box[1]),(person_box[2],person_box[3]),(255,0,0),2)
                cv2.circle(im,(head_pose[0],head_pose[1]),1,(0,0,255),1)

                temperature = (head_val* float(min_reference_temp))/min_reference_pixel
                temperature_str = '%.1f C'%(temperature)
                
                # Use green color for normal temperature
                temperature_color = (0, 255, 0)

                # Use red for elevated temperature
                if temperature > 37.5:
                    temperature_color = (0, 0, 255)

                cv2.putText(im,temperature_str , (int(person_box[0]), int(person_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX,0.5, temperature_color, 1, cv2.LINE_AA)

            cv2.imshow('img', im)
            cv2.waitKey(5000)
            
            if save_processed_images:
                base = os.path.basename(key)
                filename = os.path.splitext(base)[0]

                #print(filename)

                filename += '_processed.jpg'
            
                # Save the image
                cv2.imwrite(filename,im)

            
if __name__ == "__main__":

    """
    Read the person detection results from Artificial Neural Network
    """

    with open("flir_person_dict_val.pkl", "rb") as pickle_in:
        person_dicts = pickle.load(pickle_in)
    
    # Perform computations and display the results
    visualize_temperature_measurements(person_dicts,save_processed_images = True)
    