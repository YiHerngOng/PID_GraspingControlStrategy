# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:46:09 2019

@author: kartik
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 20:49:10 2019

@author: kartik
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 00:18:14 2019

@author: kartik
"""
import pdb
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from collections import Counter
import sys
for p in sys.path:
    print p
#print sys.path
class image_template_match():
    
    def __init__(self):
        #os.chdir('/home/kartik/session_3/opencv_test')
        os.chdir('/home/graspinglab/PID_GraspingControlStrategy/')

        self.MIN_MATCH_COUNT = 4
        self.img1 = cv2.imread('template_8.jpg',0)   # queryImage
        # cv2.waitKey(0)
        # os.chdir('/home/kartik/session_3/opencv_test')
        os.chdir('/home/graspinglab/PID_GraspingControlStrategy/images')


    def distances(self,a , b):
        return np.sqrt(np.power(a[0]-b[0],2) + np.power(a[1]-b[1],2))

    def findAngle(self,p0,p1,p2):
    #    print p0,p1,p2
        a = np.power(p1[0][0]-p0[0][0],2) + np.power(p1[0][1]-p0[0][1],2)
        b = np.power(p1[0][0]-p2[0][0],2) + np.power(p1[0][1]-p2[0][1],2)
        c = np.power(p2[0][0]-p0[0][0],2) + np.power(p2[0][1]-p0[0][1],2)
        return np.arccos( (a+b-c) / np.sqrt(4*a*b) )/np.pi*180

    def check_quality(self,polypoints):
        min_angle =360
        max_angle = 0
        for i in range(4):
            angle= self.findAngle( polypoints[np.mod(i,4)], polypoints[np.mod(i+1,4)], polypoints[np.mod(i+2,4)])
    #        print angle
            min_angle = np.min((min_angle,angle))
            max_angle = np.max((max_angle,angle))
        return min_angle,max_angle
        
    def template_match(self,image_):

        img2 = cv2.imread(image_,0) # trainImage
        
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        
        # Adding Mask for SIFT features to be found around the white paper
        Mask  = np.zeros_like(img2)
        Mask[175:350,200:400] = 1

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.img1,None)
        kp2, des2 = sift.detectAndCompute(img2,Mask)
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1,des2,k=2)
        
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        
                
        # filtering outlier matches
        #all_pts = [kp2[m.trainIdx] for m in good]
        #unique_keypts = []
        #unique_queryIdx= []
        #unique_trainIdx = []
        #for row in all_pts:
        #    print row
        #    print unique_keypts
        #    
        #    if not(row[0] in unique_keypts):
        #        unique_keypts = np.vstack((unique_keypts,tuple(row[0])))
        #        unique_queryIdx = np.vstack((unique_queryIdx,row[2]))
        #        unique_trainIdx = np.vstack((unique_trainIdx,row[1]))
        #        
                
                
        
            
                
        
        unique_keypts = np.vstack({tuple(row) for row in [kp2[m.trainIdx].pt for m in good]})
        
        #img4 = cv2.imread(image)
        #
        #for pt in unique_keypts:
        #    img4 = cv2.circle(img4,tuple((int(pt[0]),int(pt[1]))),4, 255)
        ##    cv2.imshow("lookat",img4)
        ##    cv2.waitKey(0)
        ##    cv2.destroyAllWindows()
        #
        #img4 = cv2.circle(img4,tuple((int(centroid[0]),int(centroid[1]))),4, 255)
        ##cv2.imshow("lookat",img4)
        ##cv2.waitKey(0)
        ##cv2.destroyAllWindows()
        
        #moments = cv2.moments(np.float32(unique_keypts), True)
        centroid = (np.mean([keypt[0] for keypt in unique_keypts]), np.mean([keypt[1] for keypt in unique_keypts]))
        distanceFromCentroid = [self.distances(centroid,ptId) for ptId in unique_keypts]
        
        # filtering based on Mean
        #mean_distance = np.mean(distanceFromCentroid)
        #std = np.std(distanceFromCentroid)
        #unique_keypts_filtered = unique_keypts[distanceFromCentroid <= mean_distance+0.5*std]
        
        ## filtering based on Median
        median_distance = np.median(distanceFromCentroid)
        unique_keypts_filtered = unique_keypts[distanceFromCentroid <= median_distance*1.5]
    
        good_filtered = []
        for pts in good:
            if kp2[pts.trainIdx].pt in unique_keypts_filtered:
                good_filtered.append(pts)
        
        
        
        if len(good_filtered)>self.MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_filtered ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_filtered ]).reshape(-1,1,2)
        
    #        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            M = cv2.estimateRigidTransform(src_pts, dst_pts,fullAffine =0 )
    #        matchesMask = mask.ravel().tolist()
        
            h,w = self.img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #        dst = cv2.perspectiveTransform(pts,M)
            
            
            ### these are the objects points
            
            dst = cv2.transform(pts,M)
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            
            quality_min, quality_max = self.check_quality(dst)
    #        if (quality_min > 79) & quality_max < 101:
    #            status = 'success'
    #        else:
    #            status = 'incorrect matches'
            
        else:
            print "Not enough matches are found - %d/%d" % (len(good_filtered),self.MIN_MATCH_COUNT)
    #        matchesMask = None
    #        status =  'not_enough_matches'
            
    #    if status =='success':
        
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
    #                       matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        
        
        img4 = cv2.imread(image_)
        img4 = cv2.circle(img4,(int(centroid[0]),int(centroid[1])),15,255,-1)
        img3 = cv2.drawMatches(self.img1,kp1,img2,kp2,good_filtered,None,**draw_params)
        #plt.imshow(img3, 'gray'),plt.show()
        
        cv2.imshow('template_matched',img3)
        key = cv2.waitKey(300)
        cv2.destroyAllWindows()
        # return quality_min,quality_max
        return dst
        
        if key == 27:
            return
        
#os.chdir('/home/kartik/session_3/opencv_test')
#
#MIN_MATCH_COUNT = 4
#
#img1 = cv2.imread('template_8.jpg',0)          # queryImage
#os.chdir('/home/kartik/session_3/opencv_test')
##
        
if __name__ == '__main__':
    
    template_matcher = image_template_match()

    i = 50

    image_ = 'left%04d.jpg'%i
    print image_
    dst = template_matcher.template_match(image_)
    print(dst)

#print image

