#This file is used to deploy a CV model
#This file enables object classification from a live video feed
import cv2
from ultralytics import YOLO
import math
import winsound #for alarm
import time

# Load the model
yolo = YOLO(r"runs\detect\train5\weights\best.pt")# Paste in path to your trained model if you have one
#models are locate in runs/detect/train/weights folder


# Load the video capture
videoCap = cv2.VideoCapture(1)

#code setup
target_class = "smt here" ####
prev_flask_centre = None

#return class id for a given class name
def get_target_id(result, target_class_name):
    target_id = None
    for class_id, class_name in result.names.items():
        if class_name == target_class_name:
            target_id = class_id
            break
    return target_id

#find centre of a detected object box
def box_center(box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy

#allows you to pick out a certain zone in the video livestream using percentage of FOV
#(0,0) of a frame is at the top left
# (0,0) ───────────────►  X increases →
#   │
#   │
#   ▼ Y increases
def zone_select(x1,x2,y1,y2,frame):#input zone area as percentage of total FOV
     h, w = frame.shape[:2]
     return (int(x1*w), int(y1*h), int(x2*w), int(y2*h))
    
    

#function definition to detect specific classes counted in FOV
def detection_count(result, target_class_name):
    target_id = None
    for class_id, class_name in result.names.items():
        if class_name == target_class_name:
            target_id = class_id
            break

    if target_id is None:
        return 0
    
    count = 0
    for box in result.boxes:
        if int(box.cls[0]) == target_id:
            count +=1

    return count

#functions counting number of specific class objects in a specific zone

def location_detect(result, target_class_name, zone_rect):
    target_id = get_target_id(result, target_class_name)
    if target_id is None:
        return 0
    count = 0
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id != target_id:
            continue

        cx, cy = box_center(box)
        x1,y1,x2,y2 = zone_rect
        if cx > x1 and cx < x2 and cy > y1 and cy < y2:
            count += 1

    return count

#function that returns a boolean if a nuclear flask is moving
#nuclear flask must be recognized as an image class first
#assume there can be only 1 nuclear flask in frame at a time
def is_moving(result, target_class_name, dist_thresh_px):
    global prev_flask_centre
    target_id = get_target_id(result, target_class_name)

    #case: no flask right now, no flask from before
    if target_id is None and prev_flask_centre is None: 
        return False

    for box in result.boxes:
        if int(box.cls[0]) == target_id:
            cx,cy = box_center(box)

        #case: object just appeared
        if prev_flask_centre == None:
            prev_flask_centre = (cx,cy)
            return False
        #case: object appeared before and is still in frame
        dist = math.hypot(cx - prev_flask_centre[0], cy - prev_flask_centre[1])
        prev_flask_centre = (cx, cy)
        return dist > dist_thresh_px
        
    #case: object is in last frame but now disappered
    prev_flask_centre = None
    return False

danger_detected = {"intruder": False, "leak": False}
last_alert_time = {"intruder": 0, "leak": 0}
alert_duration = 3

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    results = yolo.track(frame, persist=True) #persist = true tracks the iage even when it disppears for a small amount of time


    for result in results:
        # get the classes names
        classes_names = result.names

        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                if class_name == "leak":
                    if not danger_detected["leak"]:
                        danger_detected["leak"] = True
                        print("LEAK")
                        cv2.putText(frame, "LEAK DETECTED!",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        cv2.putText(frame, "FOLLOW THE ERAP",
                                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        last_alert_time["leak"] = time.time() #get the current time
                        winsound.Beep(1000, 700) #beeping noise

                elif class_name == "intruder":
                    if not danger_detected["intruder"]:
                        danger_detected["intruder"] = True
                        print("INTRUDER: CONTACT POLICE (1-888-310-1122)")
                        cv2.putText(frame, "INTRUDER DETECTED!",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        cv2.putText(frame, "CONTACT POLICE",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        cv2.putText(frame, "(1-888-310-1122)",
                            (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        last_alert_time["intruder"] = time.time() #get the current time
                        winsound.Beep(1000, 700) #beeping noise

                elif class_name == "readyfortransport" and not (danger_detected["intruder"] and danger_detected["leak"]):
                    print("Ready for transport!")
                    cv2.putText(frame, "Ready for transport!",
                            (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                if danger_detected["leak"] == True and time.time()-last_alert_time["leak"]<alert_duration:
                    print("LEAK")
                    cv2.putText(frame, "LEAK DETECTED!",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(frame, "FOLLOW THE ERAP",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                elif danger_detected["intruder"] == True and time.time()-last_alert_time["intruder"]<alert_duration:
                    print("INTRUDER: CONTACT POLICE (1-888-310-1122)")
                    cv2.putText(frame, "INTRUDER DETECTED!",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(frame, "CONTACT POLICE",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(frame, "(1-888-310-1122)",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
                else:
                    danger_detected["leak"] = False
                    danger_detected["intruder"] = False

                # get the respective colour
                colour = (0, 255, 0) #green

                match class_name:
                    case "intruder" | "leak": #messahe for intruder or leak is in red text
                        # draw the rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) #red box
                        cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    case _:
                        # draw the rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    # put the class name and confidence on the image
                        cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                
    # show the image
    cv2.imshow('frame', frame)



    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()


    
