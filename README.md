# IMAGE-TRANSFORMATIONS
## Developed By: SHARON STEFFANI F
## Register Number: 212223110049

## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1:
Import necessary libraries (cv2, numpy, matplotlib) and load the source image.


### Step 2:
Create transformation matrices for translation, rotation, scaling, and shearing using functions like cv2.getRotationMatrix2D().


### Step 3:
Apply the geometric transformations using cv2.warpAffine() for affine transformations and cv2.flip() for reflection.


### Step 4:
Crop the image by selecting a specific rectangular region using NumPy array slicing.


### Step 5:
Display the original and all transformed images with appropriate titles using matplotlib.pyplot.


## Program:
```python
Developed By: SHARON STEFFANI F
Register Number: 212223110049
i)Image Translation
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image=cv2.imread("iron.jpg")
input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
print("Input Image:")
plt.imshow(input_image)
plt.show()
rows,cols,dim=input_image.shape
M=np.float32([[1,0,100],[0,1,200],[0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(cols,rows))
plt.axis('off')
print("Image Translation:")
plt.imshow(translated_image)
plt.show()
```

ii) Image Scaling
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image=cv2.imread('iron.jpg')
input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
print("Input Image:")
plt.imshow(input_image)
plt.show()
rows,cols,dim=input_image.shape
M=np.float32([[1.5,0,0],[0,1.8,0],[0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(cols*2,rows*2))
plt.axis('off')
print("Image Scaling:")
plt.imshow(translated_image)
plt.show()
```

iii)Image shearing
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image=cv2.imread('iron.jpg')
input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
print("Input Image:")
plt.imshow(input_image)
plt.show()
rows,cols,dim=input_image.shape
M1=np.float32([[1,0.5,0],[0,1,0],[0,0,1]])
M2=np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
translated_image1=cv2.warpPerspective(input_image,M1,(int(cols*1.5),int(rows*1.5)))
translated_image2=cv2.warpPerspective(input_image,M2,(int(cols*1.5),int(rows*1.5)))
plt.axis('off')
print("Image Shearing:")
plt.imshow(translated_image1)
plt.imshow(translated_image2)
plt.show()
```


iv)Image Reflection
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image=cv2.imread('iron.jpg')
input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
print("Input Image:")
plt.imshow(input_image)
plt.show()
rows,cols,dim=input_image.shape
M1=np.float32([[1,0,0],[0,-1,rows],[0,0,1]])
M2=np.float32([[-1,0,cols],[0,1,0],[0,0,1]])
translated_image1=cv2.warpPerspective(input_image,M1,(int(cols),int(rows)))
translated_image2=cv2.warpPerspective(input_image,M2,(int(cols),int(rows)))
plt.axis('off')
print("Image Reflection:")
plt.imshow(translated_image1)
plt.imshow(translated_image2)
plt.show()
```



v)Image Rotation
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image=cv2.imread('iron.jpg')
input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
print("Input Image:")
plt.imshow(input_image)
plt.show()
rows,cols,dim=input_image.shape
angle=np.radians(10)
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(int(cols),int(rows)))
plt.axis('off')
print("Image Rotation:")
plt.imshow(translated_image)
plt.show()
```


vi)Image Cropping
```Python
import cv2
import matplotlib.pyplot as plt
image = cv2.imread("iron.jpg")
h, w, _ = image.shape
cropped_face = image[int(h*0.2):int(h*0.8), int(w*0.3):int(w*0.7)]
cv2.imwrite("cropped_pigeon_face.jpg", cropped_face)
plt.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
```
## Output:
### i)Image Translation
<img width="554" height="557" alt="image" src="https://github.com/user-attachments/assets/9645982c-0a20-4b9e-bac6-bc2398fcb524" />


### ii) Image Scaling
<img width="550" height="551" alt="image" src="https://github.com/user-attachments/assets/0dcd3a2a-a0ef-41fe-a3a8-a20789816046" />



### iii)Image shearing
<img width="554" height="553" alt="image" src="https://github.com/user-attachments/assets/7fde6d3d-682d-45a2-b623-cdf9c1a3caa1" />



### iv)Image Reflection
<img width="556" height="560" alt="image" src="https://github.com/user-attachments/assets/64bbc3f8-3437-4a27-a3c0-191cef475401" />



### v)Image Rotation
<img width="553" height="555" alt="image" src="https://github.com/user-attachments/assets/bf9cb53d-2d88-47f1-bb96-b7d79ad256c0" />



### vi)Image Cropping
<img width="367" height="553" alt="image" src="https://github.com/user-attachments/assets/021ce790-c06b-48a5-88de-9bc61d984c13" />




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
