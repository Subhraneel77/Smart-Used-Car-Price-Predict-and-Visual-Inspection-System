import cv2

def detect_damages(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    damage_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            damage_detected = True
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    
    cv2.imwrite(image_path, image)  # Save the image with detected damages

    return damage_detected

if __name__ == "__main__":
    damage = detect_damages('visual_inspection/car_image.jpg')
    if damage:
        print("Damage detected")
    else:
        print("No damage detected")
