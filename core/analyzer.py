def estimate_proximity(xyxy, frame_height):
    """바운딩 박스로 가까움 정도를 0~1로 추정"""
    x1, y1, x2, y2 = xyxy

    height_ratio = (y2 - y1) / frame_height      # 클수록 가까움
    bottom_ratio = y2 / frame_height             # 화면 아래일수록 가까움

    score = (height_ratio * 0.5) + (bottom_ratio * 0.5)
    return round(min(max(score, 0.0), 1.0), 3)


def estimate_direction(xyxy, frame_width):
    """화면을 3구역으로 나눠 방향 판단"""
    x1, _, x2, _ = xyxy
    center_x = (x1 + x2) / 2

    zone = int(center_x // (frame_width / 3))
    zone = min(zone, 2)

    return ["왼쪽", "정면", "오른쪽"][zone]


def analyze(detection, frame_width, frame_height):
    """감지 결과 하나에 거리/방향 정보를 추가"""
    xyxy = detection["xyxy"]

    detection["proximity"] = estimate_proximity(xyxy, frame_height)
    detection["direction"] = estimate_direction(xyxy, frame_width)

    return detection