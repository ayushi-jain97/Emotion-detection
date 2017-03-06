import dlib
import cv2

predictor_path = "shape_predictor_68_face_landmarks.dat"
feed = cv2.VideoCapture(0)
print feed.read()[1].shape
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

while True:
	img = feed.read()[1]
	win.clear_overlay()
	win.set_image(img)

	dets = detector(img, 1)
	print("Number of faces detected: {}".format(len(dets)))
	for k, d in enumerate(dets):
		print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
		shape = predictor(img, d)
		print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
		win.add_overlay(shape)
	print("")
	win.add_overlay(dets)
	dlib.hit_enter_to_continue()