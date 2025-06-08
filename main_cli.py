import argparse
import os

import cv2
from deepface import DeepFace


# python main_cli.py verify images/cxk/cxk1.png images/cxk/cxk2.png
def verify_faces(img1_path, img2_path):
    """人脸验证功能"""
    print(f"正在验证: {img1_path} 和 {img2_path}")
    try:
        result = DeepFace.verify(img1_path, img2_path)
        verified = "匹配" if result['verified'] else "不匹配"
        similarity = 1 - result['distance']

        print("\n===== 人脸验证结果 =====")
        print(f"验证结果: {verified}")
        print(f"相似度: {similarity:.4f}")
        print(f"距离值: {result['distance']:.4f} (阈值: {result['threshold']:.4f})")
        print(f"模型: {result['model']}")
    except Exception as e:
        print(f"验证失败: {str(e)}")


# python main_cli.py find images/cxk/cxk1.png images
def find_face(img_path, db_path):
    """人脸识别功能"""
    print(f"在数据库 {db_path} 中识别图像 {img_path}")
    try:
        dfs = DeepFace.find(img_path=img_path, db_path=db_path)

        if not dfs or dfs[0].empty:
            print("\n===== 人脸识别结果 =====")
            print("未在数据库中找到匹配的人脸")
            return

        print("\n===== 人脸识别结果 =====")
        print(f"找到 {len(dfs[0])} 个匹配结果")
        print("\n前5个匹配结果:")

        for df in dfs:
            top_results = df.head(5)
            for idx, row in top_results.iterrows():
                identity = os.path.basename(os.path.dirname(row['identity']))
                similarity = (1 - row['distance']) * 100

                print(f"\n身份: {identity}")
                print(f"相似度: {similarity:.2f}%")
                print(f"文件路径: {row['identity']}")
                print("-" * 50)
    except Exception as e:
        print(f"识别失败: {str(e)}")


# python main_cli.py analyze images/cxk/cxk1.png
def analyze_face(img_path):
    """面部属性分析功能"""
    print(f"正在分析图像: {img_path}")
    try:
        objs = DeepFace.analyze(img_path, actions=['age', 'gender', 'race', 'emotion'])

        print("\n===== 面部分析结果 =====")
        for i, obj in enumerate(objs):
            if len(objs) > 1:
                print(f"\n人脸 {i + 1}:")

            # 性别分析
            gender_data = obj['gender']
            man_prob = gender_data.get('Man', 0)
            woman_prob = gender_data.get('Woman', 0)

            if man_prob > 90 or woman_prob > 90:
                gender_str = "男" if man_prob > woman_prob else "女"
            elif man_prob > 60 or woman_prob > 60:
                dominant = "男" if man_prob > woman_prob else "女"
                gender_str = f"大概率{dominant} (男:{man_prob:.1f}%, 女:{woman_prob:.1f}%)"
            else:
                gender_str = f"无法判断 (男:{man_prob:.1f}%, 女:{woman_prob:.1f}%)"

            # 种族分析
            race_data = obj['race']
            dominant_race = max(race_data, key=race_data.get)
            race_mapping = {'asian': '亚洲人', 'indian': '印度人', 'black': '黑人', 'white': '白人',
                            'middle eastern': '中东人', 'latino hispanic': '拉丁裔'}
            race_str = race_mapping.get(dominant_race, dominant_race)

            # 情感分析
            emotion_mapping = {'angry': '生气', 'disgust': '厌恶', 'fear': '恐惧', 'happy': '开心', 'sad': '伤心',
                               'surprise': '惊讶', 'neutral': '中性'}
            emotion_str = emotion_mapping.get(obj['dominant_emotion'], obj['dominant_emotion'])

            # 输出结果
            print(f"年龄: {obj['age']}")
            print(f"性别: {gender_str}")
            print(f"种族: {race_str}")
            print(f"情感: {emotion_str}")

            # 详细数据
            print("\n详细分析数据:")
            print(f"  - 性别概率: 男 {man_prob:.1f}%, 女 {woman_prob:.1f}%")

            # 种族概率
            print("  - 种族概率 (降序):")
            sorted_race = sorted(race_data.items(), key=lambda x: x[1], reverse=True)
            for race, prob in sorted_race:
                chinese_race = race_mapping.get(race, race)
                print(f"      {chinese_race}: {prob:.1f}%")

            # 情感概率
            print("  - 情感概率 (降序):")
            emotion_data = obj['emotion']
            sorted_emotion = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
            for emotion, prob in sorted_emotion:
                chinese_emotion = emotion_mapping.get(emotion, emotion)
                print(f"      {chinese_emotion}: {prob:.1f}%")
            print("=" * 50)
    except Exception as e:
        print(f"分析失败: {str(e)}")


# python main_cli.py stream images
def stream_analysis(db_path):
    """实时分析功能"""
    print(f"启动实时分析，数据库: {db_path}")
    print("按ESC键可退出实时分析")

    try:
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("无法打开摄像头")
            return

        cv2.namedWindow('Real-time Face Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Face Recognition', 800, 600)

        print("实时分析运行中...")
        while True:
            if cv2.waitKey(1) == 27:  # ESC键
                break

            ret, frame = capture.read()
            if not ret:
                break

            try:
                results = DeepFace.find(img_path=frame, db_path=db_path, enforce_detection=False, silent=True)

                if results and not results[0].empty:
                    result = results[0].iloc[0]
                    identity = os.path.basename(os.path.dirname(result['identity']))
                    similarity = (1 - result['distance']) * 100

                    text = f"{identity}: {similarity:.2f}%"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except:
                pass

            cv2.imshow('Real-time Face Recognition', frame)

        capture.release()
        cv2.destroyAllWindows()
        print("实时分析已停止")
    except Exception as e:
        print(f"实时分析出错: {str(e)}")


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description="面部识别系统命令行版", formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest='command', title='可用命令', help='选择要执行的功能')

    # 人脸验证命令
    verify_parser = subparsers.add_parser('verify', help='人脸验证 - 比较两张人脸是否相同')
    verify_parser.add_argument('img1', help='第一张图片路径')
    verify_parser.add_argument('img2', help='第二张图片路径')

    # 人脸识别命令
    find_parser = subparsers.add_parser('find', help='人脸识别 - 在数据库中查找相似人脸')
    find_parser.add_argument('img', help='待识别人脸图片路径')
    find_parser.add_argument('db', help='数据库文件夹路径')

    # 面部属性分析命令
    analyze_parser = subparsers.add_parser('analyze', help='面部属性分析 - 分析年龄、性别、种族和情感')
    analyze_parser.add_argument('img', help='待分析图片路径')

    # 实时分析命令
    stream_parser = subparsers.add_parser('stream', help='实时分析 - 摄像头实时人脸识别')
    stream_parser.add_argument('db', help='数据库文件夹路径')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'verify':
        verify_faces(args.img1, args.img2)
    elif args.command == 'find':
        find_face(args.img, args.db)
    elif args.command == 'analyze':
        analyze_face(args.img)
    elif args.command == 'stream':
        stream_analysis(args.db)


if __name__ == "__main__":
    main()
