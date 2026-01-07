import cv2
import sys

# ==========================================
# 1. 配置区域 (在这里修改参数)
# ==========================================
VIDEO_PATH = "bear.mp4"  # 视频路径，如果是数字 0 则表示摄像头
OUTPUT_TXT = "tracking_result.txt"  # 结果保存路径

# 定义所有支持的算法名称列表
TRACKER_TYPES = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

# 【核心】在这里选择你想使用的算法索引 (0-6)
# 0: BOOSTING    (旧，慢)
# 1: MIL         (比Boosting准，但不能处理遮挡)
# 2: KCF         (速度快，准确率一般，推荐)
# 3: TLD         (误报多，不推荐)
# 4: MEDIANFLOW  (对快速移动目标效果差，但在移动缓慢且平滑时很好)
# 5: MOSSE       (速度极快，但准确率不如KCF)
# 6: CSRT        (精度最高，比KCF慢，处理遮挡效果好，推荐)
SELECTED_INDEX = 6  # 当前选择：KCF


# ==========================================

def create_tracker_by_name(tracker_type):
    """
    根据名称创建跟踪器，适配 OpenCV 4.3.0.36 环境
    """
    tracker_type = tracker_type.upper()
    print(f"正在创建跟踪器: {tracker_type}...")

    # 注意：这是针对 OpenCV 4.3.x 的写法 (无 cv2.legacy)
    # 如果你是 OpenCV 4.5+，需要加上 cv2.legacy.
    try:
        if tracker_type == 'BOOSTING':
            return cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            return cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            return cv2.TrackerMedianFlow_create()
        elif tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        elif tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        else:
            print(f"不支持的算法类型: {tracker_type}")
            return None
    except AttributeError:
        print("错误：你的OpenCV版本可能不包含该跟踪器，或者需要安装 opencv-contrib-python")
        print("当前尝试使用的语法是 OpenCV 3.x/4.0-4.4 的写法。")
        return None


def main():
    # 1. 获取当前选择的算法名称
    if 0 <= SELECTED_INDEX < len(TRACKER_TYPES):
        tracker_name = TRACKER_TYPES[SELECTED_INDEX]
    else:
        print("索引超出范围，默认使用 KCF")
        tracker_name = 'KCF'

    # 2. 初始化跟踪器
    tracker = create_tracker_by_name(tracker_name)
    if tracker is None:
        sys.exit()

    # 3. 读取视频
    video_source = int(VIDEO_PATH) if VIDEO_PATH.isdigit() else VIDEO_PATH
    video = cv2.VideoCapture(video_source)

    if not video.isOpened():
        print(f"无法打开视频: {VIDEO_PATH}")
        sys.exit()

    # 4. 读取第一帧
    ok, frame = video.read()
    if not ok:
        print("无法读取视频内容")
        sys.exit()

    # 5. 手动框选目标
    print(f"当前使用算法: [{tracker_name}]")
    print("请框选目标，按 [空格] 或 [回车] 确认，按 [c] 取消。")
    bbox = cv2.selectROI("Tracker", frame, False)

    if bbox[2] == 0 or bbox[3] == 0:
        print("未选择区域，程序退出")
        sys.exit()

    # 初始化跟踪器
    tracker.init(frame, bbox)

    # 6. 准备写入文件
    try:
        f_out = open(OUTPUT_TXT, "w")
        f_out.write(f"Algorithm: {tracker_name}\n")
        f_out.write("Frame_ID, X, Y, W, H\n")
    except IOError:
        print("无法创建输出文件")
        sys.exit()

    frame_id = 1

    while True:
        ok, frame = video.read()
        if not ok:
            break

        frame_id += 1
        timer = cv2.getTickCount()

        # 更新跟踪器
        ok, bbox = tracker.update(frame)

        # 计算 FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            # 跟踪成功
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # 画框
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            # 实时显示坐标
            text_coord = f"X:{x} Y:{y} W:{w} H:{h}"
            cv2.putText(frame, text_coord, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 写入TXT
            f_out.write(f"{frame_id}, {x}, {y}, {w}, {h}\n")
        else:
            # 跟踪失败
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            f_out.write(f"{frame_id}, -1, -1, -1, -1\n")

        # 显示状态信息
        cv2.putText(frame, tracker_name + " Tracker", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("Tracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break

    video.release()
    cv2.destroyAllWindows()
    f_out.close()
    print(f"完成。结果已保存至 {OUTPUT_TXT}")


if __name__ == '__main__':
    main()