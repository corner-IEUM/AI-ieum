import cv2

def get_traffic_light_color(frame,xyxy):
    """신호등 박스 영역을 잘라서 빨간불/초록불을 판단"""
    x1,y1,x2,y2=map(int,xyxy)

    # 보행자 신호등은 보통 아래쪽에 실제로 켜진 불빛이 있음
    # 박스의 아래쪽 1/3만 잘라서 분석 (꺼진 위쪽 칸의 노이즈 제거)
    height = y2 - y1
    y1_bottom = y1 + int(height * 0.66)
    crop = frame[y1_bottom:y2,x1:x2]

    if crop.size == 0 :
        return "unknown"
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, (0,120,120),(10,255,255))
    red_mask2 = cv2.inRange(hsv, (160,120,120) , (180,255,255))
    red_mask = red_mask1 | red_mask2

    green_mask = cv2.inRange(hsv,(40,100,100),(90,255,255))

    red_count = cv2.countNonZero(red_mask)
    green_count = cv2.countNonZero(green_mask)

    # 노이즈 필터링용 최소 픽셀 개수 기준
    min_pixels = 30

    if red_count > green_count and red_count > min_pixels:
        return "red"
    elif green_count > red_count and green_count > min_pixels:
        return "green"
    else:
        return "unknown"