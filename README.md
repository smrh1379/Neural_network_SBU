# This is final project of the course. Each HW has it's own branch
# Math_equation_solver_using_object_detection_Detectron2
Final project of Master course that was taken during bachelor degree.
![alt text](https://github.com/smrh1379/Neural_network_SBU/blob/main/result.jpg?raw=true)
<br>
In the beginning of the project the dataset was created by me by using roboflow. objects that the model should be able to recognize was: 2,4,6,+,-
<br>
Then by writing lots of equations in paper, 5 images was uploaded to roboflow. Then, with the use of data augmentation techniques such as rotation,shear,brighness,noise,and blur dataset got bigger.<br>
After that by getting COCO output we used Detectron 2 pre-trined models (transfer learning)<br>
model faster_rcnn_R_101_FPN_3x was used because of it's fast pace and good accuracy.<br>
after training the model on different dataset that each was created by using different augmentation the model was able to perfectly detect each number. the result picture is down below.<br>
![alt text](https://github.com/smrh1379/Neural_network_SBU/blob/main/final.jpg?raw=true)
By using the algorithm developed by our team the result get calculated and then was written to the picture.
! important: The algorithm is written such that each equation should be started with => sign to be able to calculate it. Numbers far from this sign would not be considered as a number in equation. if the equation is such that numbers are not followed by an operator it will be ignored
