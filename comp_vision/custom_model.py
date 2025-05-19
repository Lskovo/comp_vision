from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import logging

# Настройка логов
logging.basicConfig(
    filename="comp_vision/data/log_file.log",       # файл логов
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("yolo_demo")

class Custom_Model:
    def demo(self, model_path, video_path, output_path):
        # Загружаем модель
        model = YOLO(model_path)

        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Создаём выходной видеопоток
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Создаём списки для записи статистики
        unique_ids = set()
        object_counts = []
        object_areas = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            log.info("New frame read")

            # YOLOv8 с трекингом
            results = model.track(
                source=frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.5,
                imgsz=640,
                verbose=False,
                stream=False
            )[0]

            annotated = frame.copy() # копируем кадр, на котором будем рисовать

            if results.masks is not None: # убеждаемся, что для кадра есть маски

                log.info(f"Masks found: {len(results.masks.xyn)}")

                h, w, _ = frame.shape
                masks = results.masks.xyn  # нормализованные контуры
                ids = results.boxes.id
                ids = ids.cpu().numpy().astype(int) if ids is not None else [-1] * len(masks)

                # Ищем площади всех объектов
                areas = []
                contours_list = []
                for polygon in masks:
                    pts = (np.array(polygon) * np.array([w, h])).astype(np.int32)
                    contours_list.append(pts)
                    area = int(cv2.contourArea(pts))
                    areas.append(area)

                # Получаем индекс самого большого объекта
                if areas:
                    max_index = np.argmax(areas)
                else:
                    max_index = -1

                # Считаем и сохраняем статистику
                object_count = len(areas) # считаем количество объектов в кадре
                object_counts.append(object_count) # сохраняем для нахождения среднего количества объектов в кадре
                object_areas.extend(areas) # сохраняем площади объектов для нахождения средней площади

                # Обновляем множество уникальных ID
                unique_ids.update(ids)

                # Подпись в левом верхнем углу
                cv2.putText(annotated,
                            f"Objects in frame: {object_count}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 0),
                            2)
                
                # Отрисовываем контуры
                for i, (pts, obj_id, area) in enumerate(zip(contours_list, ids, areas)):
                    color = (0, 255, 0)
                    thickness = 2

                    if i == max_index:
                        color = (0, 0, 255)  # самый большой объект обводим красным
                        thickness = 4 # и толще

                    cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)

                    M = cv2.moments(pts)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        label = f"ID:{obj_id} | {area}px"
                        cv2.putText(annotated, label, (cx - 40, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Записываем нарисованное в результат
                out.write(annotated)
            else:
                log.info("No masks found")

        cap.release()
        out.release()
        print("Result saved to:", output_path)
        print("\n=== Video statistics ===")
        print(f"Unique objects (ID): {len(unique_ids)}")
        print(f"Mean number of objects per frame: {np.mean(object_counts):.2f}")
        print(f"Mean object area: {np.mean(object_areas):.1f} pixels")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model demo.")
    parser.add_argument('mode', choices=['demo'], help="Demo of the trained model.")
    parser.add_argument('--mp', type=str, required=True, help="Path to the model.")
    parser.add_argument('--vp', type=str, required=True, help="Path to the input video.")
    parser.add_argument('--op', type=str, required=True, help="Path to the output video.")

    args = parser.parse_args()

    model = Custom_Model()

    if args.mode == 'demo':
        model.demo(args.mp, args.vp, args.op)