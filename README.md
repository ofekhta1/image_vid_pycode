# image_vid_pycode
python code that checks if images are  existing in vid

the user need to enter his dirs for images and vids( the default for images is C:\python\images and for vids is C:\python\vids)
the program will run over each frame in the vid . its tries to find a face in the frame. if it's find one,he embedds it and then compare it to the embedded face from the image. if the similarity is over 0.47( which means it's probably the same person) - the program does some actions:
1. move the image to new folder named: c:\python\archive( if this folder doesn't exist, the program creates it)
2. create new folder that her name is same as the image file name. inside this folder its add and create some images:
   1. the original image
   2. the original vid
   3. the first frame in the vid which the program finds similarity over 0.47( which means the first frame that the person from the image probably appears in the vid) . 
