import cv2
import numpy as np


class GeometryFeature:

    def build_mask(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        if num_labels <= 1:
            return mask

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        return np.where(labels == largest_label, 255, 0).astype("uint8")

    def extract(self, image):

        thresh = self.build_mask(image)

        # Lưu mask để kiểm tra
        cv2.imwrite("debug_mask.png", thresh)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise Exception("Object not found.")

        contour = max(contours, key=cv2.contourArea)

        object_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        x, y, w, h = cv2.boundingRect(contour)

        # Debug contour
        debug = image.copy()
        cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imwrite("debug_contour.png", debug)

        return {
            "object_area": object_area,
            "perimeter": perimeter,
            "width": w,
            "height": h,
            "bbox": [x, y, w, h]
        }