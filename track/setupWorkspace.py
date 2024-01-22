import cv2 
import numpy as np
from ultralytics import YOLO
import time
#from collections import defaultdict


##################################################################################
# Open a screen to mark out workspace, then show workspace transformed view live
##################################################################################

#fname = "tracking_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"  # filename for csv log
#track_history = defaultdict(lambda: [])

  
def mouse_click(event, x, y, flags, param): 
    global clicks, clkptr, done, confirm

    # to check if left mouse button was clicked 
    if event == cv2.EVENT_LBUTTONDOWN: 
        clicks[clkptr] = [x, y]
        #print(clicks)
        clkptr += 1
        if clkptr > 3 :
            clkptr = 0
            done = True

    if event == cv2.EVENT_MBUTTONDOWN: 
        if done:
            confirm = True


# Part One - Open Camera To Mark Workspace Corners

# opencv text format for overlays
font        = cv2.FONT_HERSHEY_SIMPLEX
fontScale   = 1
thickness   = 2
lineType    = 2

done    = False
confirm = False

clkptr  = 0                 # pointer to keep track of what click we are on
clicks  = np.zeros((4,2))   # empty array to begin
corners = [ "Top-Left" , 
            " Bottom-Left", 
            "Bottom-Right", 
            "Top-Right" ]   # Names for each corner in sequence

colours = [(255,0,0), 
           (0,255,0), 
           (0,0,255), 
           (255,255,255)]   # Colours for each corner in sequence
  
cap = cv2.VideoCapture(2) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if cap.isOpened() == False: 
    # give error message 
    print("Error in opening file.") 
else: 
    # proceed forward 
    while(cap.isOpened() and not confirm): 
        ret, frame  = cap.read()
        lr_image    = np.split(frame, 2, axis=1)    # only use half image from zed
        lframe      = lr_image[0]
        
        if ret == True: 
            
            if done:
                cv2.putText(lframe, "Middle Click To Confirm", (10, 50), font, fontScale, (0,250,250), thickness+1128, lineType )
            else:
                cv2.putText(lframe, corners[clkptr], (10, 50), font, fontScale, colours[clkptr], thickness, lineType )
            
            for i in range(4):
                cv2.drawMarker(lframe, clicks[i].astype(int) ,colours[i], markerType=cv2.MARKER_CROSS, thickness=5)
            
            cv2.imshow("Set Workspace", lframe) 
            cv2.setMouseCallback("Set Workspace", mouse_click, param=lframe) 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
        else: 
            break
  
  
cap.release() 
cv2.destroyAllWindows() 

print(clicks)



# Part Two - Transform Workspace View
# region - Workspace Transform
pt_A = clicks[0]  
pt_B = clicks[1]
pt_C = clicks[2]
pt_D = clicks[3]
print(pt_A)

# Here, I have used L2 norm. You can use L1 also.
width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))

height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])
# endregion



model = YOLO('1080_v1n.pt')

cap = cv2.VideoCapture(2) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    left_right_image = np.split(frame, 2, axis=1)

    lframe = left_right_image[0]

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(lframe, persist=True, max_det = 1)
        # Get the boxes and track IDs
        # boxes = results[0].boxes.xywh.cpu()
        # track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        # for box, track_id in zip(boxes, track_ids):
        #     x, y, w, h = box
        #     track = track_history[track_id]
        #     track.append((float(x), float(y)))  # x, y center point
        #     if len(track) > 30:  # retain 90 tracks for 90 frames
        #         track.pop(0)

            # # Draw the tracking lines
            # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

            # if track_id == 1:
            #     tracecolor = (250, 0, 0)
            # else:
            #     tracecolor = (0, 250, 0)
            # #cv2.polylines(annotated_frame, [points], isClosed=False, color=tracecolor, thickness=10)
            # print(points)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Display the annotated transformed frame
        M = cv2.getPerspectiveTransform(input_pts,output_pts)
        topViewFrame = cv2.warpPerspective(annotated_frame,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
        cv2.imshow("Top View", topViewFrame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
