import cv2
import lYolo as ly

im = cv2.imread("t3.jpg")
obj = ly.LYL()
obj.init(im)

img = obj.drawObj("", True)

count = obj.countObj("con chim")
ly.settext(img, "co " + str(count) + " con chim", (30,35))
cv2.imshow("ket qua", img)
cv2.waitKey(0)
cv2.destroyAllWindows()