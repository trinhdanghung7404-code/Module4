import cv2, numpy as np

img1 = cv2.imread(r'images\1.jpg')
img2 = cv2.imread(r'images\2.jpg')

# Tri 3 center is around x=484, y=1156
cx, cy = 484, 1156
c1 = img1[cy-100:cy+100, cx-100:cx+100]
c2 = img2[cy-100:cy+100, cx-100:cx+100]

vis = np.hstack([c1, c2])
cv2.circle(vis, (100, 100), 5, (0, 0, 255), -1)
cv2.circle(vis, (300, 100), 5, (0, 0, 255), -1)
cv2.putText(vis, '1.jpg (Product)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(vis, '2.jpg (Return)', (210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imwrite('scratch/tri3_context_rgb.jpg', vis)
print('Saved scratch/tri3_context_rgb.jpg')
