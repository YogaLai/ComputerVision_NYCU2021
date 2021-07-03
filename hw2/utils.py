import cv2


def show_img(img_title, img):

    cv2.imshow(img_title, img)
    TIME_WAIT = 0
    key = cv2.waitKey(TIME_WAIT)
    
    if key == ord("q"):
       quit()
    if TIME_WAIT == 0:
        cv2.destroyWindow("img")
