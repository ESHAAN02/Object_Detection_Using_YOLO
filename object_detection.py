#importing numpy and opencv
import numpy as np
import cv2 as cv
from tkinter import messagebox #import messagebox from tkinter

def select_image():
    global panelA,panelB
    #selecting the path of an images using filedialog
    path = tkinter.filedialog.askopenfilename(filetypes=[("Image File",'*.jpg')])#taking only .jpg images
    if len(path)>0:
        # Load YOLO
        net = cv.dnn.readNet("D:/Datasets/yolov3.weights", "D:/Datasets/yolov3.cfg")#give yolov3.weights,yolov3.cfg file path
        classes = []
        f = open("D:/Datasets/coco.names", 'r')#give coco.names file path
        for line in f:
            c = line[:len(line) - 1]
            classes.append(c)
        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        #-------------------_

        img=cv.imread(path)#reading the original image
        detect_img = cv.resize(img, (600, 850), cv.INTER_AREA)#resize an output image form the original image
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)#converting oringinal image form BGR to RGb
        detect_img = cv.cvtColor(detect_img, cv.COLOR_BGR2RGB)#converting output image form BGR to RGB


        img = Image.fromarray(img)
        img=img.resize((600,850),Image.ANTIALIAS)

        img = ImageTk.PhotoImage(img)
        # Loading image


        height, width, channels = detect_img.shape

        # Detecting Objects
        blob = cv.dnn.blobFromImage(detect_img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(outputlayers)

        # Showing information on the screen

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        number_object_detected = len(boxes)
        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv.FONT_ITALIC
        for i in range(number_object_detected):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv.rectangle(detect_img, (x, y), (x + w, y + h), color, 2)
                cv.putText(detect_img, label, (x + 10, y), font, 1, color, 3)

        detect_img = Image.fromarray(detect_img)
        detect_img = ImageTk.PhotoImage(detect_img)
        if panelA is None or panelB is None:
            original_image=Label(root,text='BEFORE DETECTION',fg='red',bg='black',font = "Helvetica 16 bold italic")
            original_image.place(x=150,y=60)
            output_image=Label(root,text='AFTER DETECTION',fg='red',bg='black',font = "Helvetica 16 bold italic")
            output_image.place(x=800,y=60)
            panelA = Label(image=img)
            panelA.image = img


            panelA.pack(side="left", padx=10, pady=10)
            panelB = Label(image=detect_img)
            panelB.image = detect_img
            panelB.pack(side="right", padx=10, pady=10)
        else:
            panelA.configure(image=img)
            panelB.configure(image=detect_img)
            panelA.image = img
            panelB.image = detect_img
    else:

        messagebox.showerror("ERROR", "NOT SELECTED ANY IMAGE")

#GUI part for creating a window
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
root =Tk()
panelA = None
panelB = None

root.geometry("1200x1000+400+100")
root.title("OBJECT DETECTION ON IMAGE")
root.config(bg="black")
select_image_frame=Frame(root,bg="black",height=50)
select_image_frame.pack(fill=X)
btn = Button(root, text="Select an image",font=("Canbria",16),bg="BLUE",fg="white",
             relief=FLAT,activebackground="#b0abae",command=select_image)
btn.place(x=550,y=10)

root.mainloop()


