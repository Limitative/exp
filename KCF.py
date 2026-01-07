import cv2
import sys
import os

# --- 配置部分 ---
VIDEO_PATH = "bear.mp4"  # 输入视频路径，如果是0则使用摄像头
OUTPUT_TXT = "tracking_data.txt"  # 输出坐标文件的路径
TRACKER_TYPE = "KCF"  # 指定使用KCF算法

def create_tracker(tracker_type):
    """
    根据OpenCV版本创建跟踪器
    注意：在OpenCV 4.3.0.36中，直接使用cv2.TrackerKCF_create()
    """
    tracker_type = tracker_type.upper()

    # 尝试创建 KCF 跟踪器
    if tracker_type == 'KCF':
        # 检查是否是旧版本(4.5以下)写法 或者 新版本(4.5以上) legacy 写法
        try:
            # 针对 OpenCV 4.3.0.36 的写法
            tracker = cv2.TrackerKCF_create()
        except AttributeError:
            try:
                # 针对 OpenCV 4.5+ 的写法
                tracker = cv2.legacy.TrackerKCF_create()
            except AttributeError:
                print("错误: 无法创建KCF跟踪器，请确保安装了 opencv-contrib-python")
                return None
        return tracker
    else:
        print(f"本实验主要演示 KCF，暂不支持 {tracker_type}")
        return None

def main():
    # 1. 初始化跟踪器
    tracker = create_tracker(TRACKER_TYPE)
    if tracker is None:
        sys.exit()

    # 2. 读取视频
    # 如果 VIDEO_PATH 是数字字符串，则转为整数打开摄像头
    video_source = int(VIDEO_PATH) if VIDEO_PATH.isdigit() else VIDEO_PATH
    video = cv2.VideoCapture(video_source)

    if not video.isOpened():
        print(f"无法打开视频文件或摄像头: {VIDEO_PATH}")
        sys.exit()

    # 3. 读取第一帧
    ok, frame = video.read()
    if not ok:
        print("无法读取视频内容")
        sys.exit()

    # 4. 手动选择目标区域 (ROI)
    print("请在弹出的窗口中框选目标，选好后按 [空格] 或 [回车] 确认，按 [c] 取消")
    bbox = cv2.selectROI("KCF Tracker", frame, False)
    # bbox 格式为 (x, y, w, h)

    # 如果用户取消选择（bbox全为0），则退出
    if bbox[2] == 0 or bbox[3] == 0:
        print("未选择区域，程序退出")
        sys.exit()

    # 初始化跟踪器
    tracker.init(frame, bbox)

    # 5. 打开txt文件准备写入
    try:
        f_out = open(OUTPUT_TXT, "w")
        # 写入表头
        f_out.write("Frame_ID, X, Y, W, H\n")
        print(f"开始跟踪，结果将写入: {OUTPUT_TXT}")
    except IOError as e:
        print(f"无法创建输出文件: {e}")
        sys.exit()

    frame_id = 1

    while True:
        ok, frame = video.read()
        if not ok:
            break

        frame_id += 1

        # 计时开始
        timer = cv2.getTickCount()

        # 更新跟踪器
        ok, bbox = tracker.update(frame)

        # 计算 FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 可视化与数据记录
        if ok:
            # 跟踪成功
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # 1. 画矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, 1)

            # 2. 实时显示坐标信息
            coord_text = f"Coord: x={x}, y={y}, w={w}, h={h}"
            cv2.putText(frame, coord_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 3. 写入 txt 文件
            f_out.write(f"{frame_id}, {x}, {y}, {w}, {h}\n")
            # 也可以在控制台打印
            # print(f"Frame {frame_id}: {x}, {y}, {w}, {h}")

        else:
            # 跟踪失败
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            # 失败时也可以记录，或者记录为 -1
            f_out.write(f"{frame_id}, -1, -1, -1, -1\n")

        # 显示通用信息
        cv2.putText(frame, "Tracker: " + TRACKER_TYPE, (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # 显示结果窗口
        cv2.imshow("KCF Tracker", frame)

        # 按 ESC 键退出
        if cv2.waitKey(1) & 0xFF == 27:
            print("用户手动停止")
            break

    # 资源释放
    video.release()
    cv2.destroyAllWindows()
    f_out.close()
    print("程序结束，资源已释放")

if __name__ == '__main__':
    main()