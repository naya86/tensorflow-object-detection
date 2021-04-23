import os
import time
import cv2
import numpy as np


def process_image( img ) :
    """ 이미지 리사이즈하고, 차원 확장 
    img : 원본이미지
    결과는 ( 64, 64, 3 ) 으로 프로세싱된 이미지 반환 """

    image_org = cv2.resize( img, ( 416, 416 ), interpolation = cv2.INTER_CUBIC) 
    image_org = np.array( image_org, dtype='float32' )
    image_org = image_org / 255.0
    image_org = np.expand_dims( image_org, axis = 0 )

    return image_org


def get_classes( file ) :
    """ 클래스의 이름을 가져온다.
    리스트로 클래스 이름을 반환한다. """
    with open(file) as f :
        name_of_class = f.readlines()

    name_of_class = [ class_name.strip() for class_name in name_of_class  ]

    return name_of_class


def box_draw( image, boxes, scores, classes, all_classes):
   
    """ image : 원본이미지
    boxes : 오브젝트의 박스데이터, ndarry
    classes : 오브젝트의 클래스
    scores : 오브젝트의 확률
    all_classes : 모든 클래스 이름 """

    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()    



def detect_image( image, yolo, all_classes ) :
    """ image : 원본 이미지
    yolo : YOLO 모델
    all_classes : 전체 클래스 이름 
    변환된 이미지 리턴 """

    pimage = process_image(image)

    image_boxes, image_classes, image_scores = yolo.predict(pimage, image.shape)

    if image_boxes is not None : 
        box_draw(image, image_boxes, image_scores, image_classes, all_classes)

    return image



## 욜로 모델 만들기
from yolo.model.yolo_model import YOLO

yolo = YOLO(0.6, 0.5)

all_classes = get_classes('yolo/data/coco_classes.txt')


## 비디오를 실행하는 코드로

cap = cv2.VideoCapture('data/videos/video.mp4')



if cap.isOpened() == False :    # True False로 값이 나옴 isOpened
    print('Error opening video stream of file')

else :
    # 반복문 필요이유 : 비디오는 여러 사진으로 구성되어 있으니까.! 여러개니까
    while cap.isOpened() :
        
        # 사진을 한장씩 가져와서 

        ret, frame = cap.read()       # ret 에는 True , False 로 가져오고 , frame 에는 numpy 로 가져옴(이미지). 비디오에 관한 프레임이 있으면 ret은 True

        # 제대로 사진 가져왔으면, 화면에 표시
        if ret == True :
            # 이 부분을 모델 추론, 화면에 보여주는 코드로 변경
            # cv2.imshow('frame', frame)
            start_time = time.time() # 추론시간 계산.
            result = detect_image( frame, yolo, all_classes)
            cv2.imshow('result', result)     #  가공이 필요할때는 이 부분에 가공을 해주면 된다.
            end_time = time.time()
            print(end_time - start_time)

            #키보드에서 esc키를 누르면 exit 하라
            if cv2.waitKey(25) & 0xFF == 27 :
                break

        else : 
            break

    cap.release()  # 비디오파일 닫는 느낌

    cv2.destroyAllWindows()
