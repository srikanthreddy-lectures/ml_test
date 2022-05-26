import face_recognition
import cv2
import mysql.connector

mydb = mysql.connector.connect(host="localhost",user="root",passwd="root",database="PHOTOS")
mycursor = mydb.cursor()
def register(filepath,username):
    with open(filepath,"rb") as f:
        binarydata = f.read()
    s1 = "INSERT into images(name,img_dir,image) VALUES(%s,%s,%s)"
    mycursor.execute(s1,(username,filepath,binarydata))
    mydb.commit()


def capture_recog():
    known_faces=[]
    known_names=[]
    sqlst = "SELECT name,img_dir FROM images WHERE img_dir IS NOT NULL"
    mycursor.execute(sqlst)
    for imgrecord in mycursor:
        timg = face_recognition.load_image_file(imgrecord[1])
        tloc = face_recognition.face_locations(timg)
        known_faces.append(face_recognition.face_encodings(timg)[0])
        known_names.append(imgrecord[0])
    print(known_names)
    video = cv2.VideoCapture(2)
    while True:
        suc,img = video.read()
        cv2.imshow("video",img)
        k = cv2.waitKey(1)
        if k%256==32:
            name = "opencv/image1.jpg"
            cv2.imwrite(name,img)
            print("Image captured")
            break
    test_img = face_recognition.load_image_file(name)
    facelocations = face_recognition.face_locations(test_img)
    faceencodings = face_recognition.face_encodings(test_img,facelocations)
    
    for (t,r,b,l),encoding in zip(facelocations,faceencodings):
        print(known_names)
        match = face_recognition.compare_faces(known_faces[0],encoding,tolerance=0.6)
        #for i in known_faces:
            #match = face_recognition.compare_faces(i,encoding,tolerance=0.6)
        name = "Unknown"
        print(match)
        
        if True in match:
            match_ind = match.index(True)
            print(known_names)
            print(match_ind)
            name = known_names[match_ind]
        cv2.rectangle(test_img,(l,t),(r,b),(0,255,0),2)
        cv2.rectangle(test_img,(l,b+20),(r,b),(0,255,0),cv2.FILLED)
        cv2.putText(test_img,name,(l+6,b+14),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),2)
    test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
    cv2.imshow("fj",test_img)
    cv2.waitKey(0)


def capture_register(un):
    video = cv2.VideoCapture(2)
    while True:
        ret,img = video.read()
        cv2.imshow("yi",img)
        k = cv2.waitKey(1)
        if k%256==32:
            name = "Resources/{0}.jpg".format(un)
            cv2.imwrite(name,img)
            print("Image captured")
            break
    register(name,un)
    print("Registered successfully")

print("1.New User\n2.Existing user\n")
n = int(input("Enter your choice: "))

if n==2:
    capture_recog()
if n==1:
    un = input("Please Enter your name: ")
    capture_register(un)