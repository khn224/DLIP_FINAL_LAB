import torch
import cv2 as cv
import math
import numpy as np

def convert(size, box): # Convert bounding box vertices(x1,y1,x2,y2) to (center point, width, height) form
    x = (box[0] + box[2]) / 2.0   
    y = (box[1] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    return (x,y,w,h)

def MovingAvg(Array): # Moving average filter (window size=30) 
    Nwindow =30 # Size of window
    fwindow = Array[-Nwindow:] # Sliced window
    avg=sum(fwindow)/Nwindow # Average value
    return avg


RECORD = 0 # For webcam
VIDEO  = 1 # For video file
MODE = VIDEO # Select Mode

# Video FileName
Filename1 = "test1.mp4" # Situation 1: Unexpected pedestrian stands in front of the car
Filename2 = "test2.mp4" # Situation 2: Pedestrian crosses the street
Filename3 = "test3.mp4" # Situation 3: Pedestrian crosses the street + Suddenly a pedestrian appears

if MODE == RECORD: # For realtime webcam recording
    cap = cv.VideoCapture(cv.CAP_DSHOW+1)
elif MODE == VIDEO: # For pre-recorded video
    cap = cv.VideoCapture(Filename3)
    
fps    = 30 # Frames per second
fourcc = cv.VideoWriter_fourcc('D', 'I', 'V', 'X') # Output videofile codec type
out    = cv.VideoWriter('demo.avi', fourcc, fps, (640, 480))# For exporting the output videofile

# Initialize count
count = 0 # Frame number
center_points_prev_frame = [] # Center point of a bounding box of the previous frame
depth_prev_frame         = [] # Radial distance of the object


tracking_objects = {}  # Center points of the tracking object
track_id = 0 # ID of the tracking object

depthConst      = 480*3 		# Pixel-to-distance coefficient
FOVconst        = 41 / 640 		# Pixel-to-angle coefficient
velocity        = 0.0      		# Angular velocity of the object
radial_velocity = 0.0       	# Radial velocity of the object
AvgRadVel       = 0.00000001 	# Average radial velocity
bufRadialVel    = []        	# Buffer of average radial velocity  for moving average filter
static_count    = 0         	# Count up if object is static
StopSignCount   = 0				# Count up if stop sign is displayed
pts1            = np.array([[0,0],[640,0],[640,480],[0,480]]) 	# Point vector of whole frame 
detected        = False			# True when object is detected
TimeToCollision = 0.0			# TTC
Epsilon         = 0.00000001 	# To prevent zero division error
time_prev       = cv.getTickCount() / cv.getTickFrequency() 	# Time of previous frame

model      = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True) # You can choose among yolo5n,yolo5s,yolo5m,yolo5l,yolo5x
model.conf = 0.6 # Confidence threshold of the model
model.iou  = 0.4 # IoU threshold of the model

while True:
    
    ret, frame    = cap.read() # Read frame
    if not ret: # If frame is not read, break
        break

    if MODE == RECORD : # Save frame to output videofile when mode: RECORD
        out.write(frame)

    height ,width = frame.shape[:2] # Height, width of the frame
    time_curr     = cv.getTickCount() / cv.getTickFrequency() # Time of current frame
    time_del      = time_curr - time_prev 					  # Time per frame

    cord   = []   # Point vectors of bounding box of the detected object
    idx    = []	  # Index of the detected object
    heightList =[]# List of bounding box height 
    maxH =0 # Maximum height of the bounding box of the detected object 
    maxHindex =0 # Index of the bounding box which has the maximum height
    count += 1 # Count up when each frame starts


    # Point current frame
    center_points_cur_frame = [] # Center point of the bounding box of current frame
    depth_cur_frame         = [] # Radial distance of the object of current frame
    results                 = model(frame) # Run inference

    LABEL , COORD           = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy() # Label, coordinate data from the inference results
    
    for i , la in enumerate(LABEL):
        if la == 0 :# Detect only for person
            idx.append(i) # Append index of the detected object
            detected =True # Set detected flag to true

    for i , index in enumerate(idx): 
        cord.append(COORD[index])# Append coordinate of detected object
        

    # Detect objects on frame
    boxes = cord # (x1,y1,x2,y2) type bounding box data
    x_shape, y_shape = frame.shape[1], frame.shape[0] #Size of the frame or roi

    for i,box in enumerate(boxes): # Append bounding box height
        heightList.append(cord[i][1] -cord[i][3])
    if detected and len(heightList)>0: # Detect the nearest object
        maxH = max(heightList) # Maximum height of the bounding box of the detected object 
        maxHindex = heightList.index(maxH)# Index of the bounding box which has the maximum height
        row            = cord[-maxHindex-1] # Each bounding box data
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)# Point vectors of bounding box of the detected object
        (x, y, w, h)   = convert((width,height), (x1,y1,x2,y2))# Converted bounding box data to (x,y,w,h) form
        
        center_points_cur_frame.append((x, y)) # Append center point of the bounding box of current frame
        depth_cur_frame.append(round(depthConst / (y2-y1),2)) # Append radial distance of the object of current frame
        
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)# Draw bounding box
        
         # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame: 
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])# L2 norm between center point of the previous bounding box and current bounding box
                velocity = ( FOVconst *(distance) * math.pi / 180.0 / time_del ) * depthConst / (y2-y1)# Calculate angular velocity of the object
                
                if distance < 20: # Object tracking L2 norm is less than 20
                    tracking_objects[track_id] = pt # Append center point of the bounding box of the tracking object
                    track_id += 1 # Set Id of the traking object

        for depth_cur in depth_cur_frame:
            for depth_prev in depth_prev_frame:
                radial_velocity = (depth_cur-depth_prev) / time_del # Calculate radial velocity of the object
                bufRadialVel.append(radial_velocity)  # Append radial velocity 
                AvgRadVel = radial_velocity # Calculate average radial velocity

            if depth_cur < 4.0: # Display stop sign if the distance is less than 4m
                cv.putText(frame, "STOP", (100,240), 0, 6, (0, 0, 255), 7)

            
    else:
        object_exists = False # Object detection flag
        tracking_objects_copy = tracking_objects.copy() # Make a copy of tracking objects
        center_points_cur_frame_copy = center_points_cur_frame.copy() # Make a copy of center points of the current frame 

        for object_id, pt2 in tracking_objects_copy.items():
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])#L2 norm between center point of the previous bounding box and current bounding box
                velocity = ( FOVconst *(distance) * math.pi / 180.0 / time_del ) * depthConst / (y2-y1)#Calculate angular velocity of the object
                # Update IDs position
                if distance < 40: #Object tracking
                    tracking_objects[object_id] = pt # Append center point of the bounding box of the tracking object
                    object_exists = True 
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt) # Remove duplicated points
                    continue

        # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)


        for depth_cur in depth_cur_frame:
            for depth_prev in depth_prev_frame:
                radial_velocity = (depth_cur-depth_prev) / time_del #Calculate the instantaneous radial velocity
                bufRadialVel.append(radial_velocity)# Append radial velocity 
                AvgRadVel = MovingAvg(bufRadialVel)+Epsilon # Calculate average radial velocity by moving average filter
                TimeToCollision = -depth_cur/AvgRadVel # Calculate TTC


            if depth_cur < 4.0 and StopSignCount < 80: # Display stop sign for 80 frames if the distance is less than 4m
                cv.putText(frame, "STOP", (100,240), 0, 5, (0, 0, 255), 10)
                copyframe = frame.copy()
                cv.fillConvexPoly(copyframe,pts1,(0,0,255)) # Paint the whole frame slightly red to warn
                frame = cv.addWeighted(frame,0.7,copyframe,0.3,0) # Adjust opacity 
                StopSignCount+=1
                if velocity < 15: # Count when angular velocity is less than 15 deg/s
                    static_count += 1
            
            if StopSignCount >= 80 and static_count >50: # Display avoid sign if the object is stationary
                static_count += 1
                if object_exists:
                    cv.putText(frame, "AVOID", (100,240), 0, 5, (0, 165, 255), 10)
                elif not object_exists:
                    static_count = 0 # Reset static count
                    StopSignCount =0 # Reset stop sign count
        
                # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1
    for object_id, pt in tracking_objects.items():# Display Distance,Velocity , TTC and collision waring
        cv.putText(frame, f"Distance[m]: { round(depthConst / (y2-y1),2)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv.putText(frame, f"Distance[m]: { round(depthConst / (y2-y1),2)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,255), 2)
        cv.putText(frame, "velocity(horizontal)[m/s] : " + str(round(velocity,   2)),   (10, 60) , 0, 0.7, (0, 0, 0), 5)
        	
        cv.putText(frame, "velocity(horizontal)[m/s] : " + str(round(velocity,   2)),   (10, 60) , 0, 0.7, (0, 255, 255), 2)
        	
        cv.putText(frame, "velocity(radial)[m/s]     : " + str(round(AvgRadVel,  2)),   (10, 90) , 0, 0.7, (0, 0, 0), 5)
       	 	
        cv.putText(frame, "velocity(radial)[m/s]     : " + str(round(AvgRadVel,  2)),   (10, 90) , 0, 0.7, (0, 255, 255), 2)
       	 	
        if TimeToCollision < 8 and TimeToCollision >0  : # Display collision warning if TTC is less than 8 sec
            cv.putText(frame, "Time to collision[s]      : " + str(round(TimeToCollision, 2)),   (10, 120), 0, 0.7, (0, 0, 0), 5)
            	
            cv.putText(frame, "Time to collision[s]      : " + str(round(TimeToCollision, 2)),   (10, 120), 0, 0.7, (0, 255, 255), 2)
            cv.putText(frame, "COLLISION WARNING!", (150,400), 0, 1, (0, 0, 0), 5)
            cv.putText(frame, "COLLISION WARNING!", (150,400), 0, 1, (0, 165, 255), 2)
    
    time_prev = time_curr 

    cv.imshow("Frame", frame)
    if MODE == VIDEO: 
        out.write(frame)

        # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()
    depth_prev_frame = depth_cur_frame.copy()     
    
    key = cv.waitKey(30)
    if key == 27:
        break
    elif key == ord('s'): # s for pause video
        cv.waitKey()
     
cv.destroyAllWindows()
cap.release()
out.release()
              

  



