import cv2
from ultralytics import YOLO
import cvzone
from collections import defaultdict

# Load the YOLOv8 model
model = YOLO("best.pt")

# Define the video source
video_path = "pac.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0

# Definir as coordenadas da linha de contagem
line_p1 = (412, 417)
line_p2 = (562, 432)

# Inicializar contadores e dicionários para rastreamento
in_count = 0
out_count = 0
object_positions = defaultdict(lambda: {'status': None, 'last_pos': (0, 0)})

if not cap.isOpened():
    print(f"Erro: Nao foi possível abrir o vídeo em {video_path}")
    exit()

print("Vídeo aberto com sucesso. Processando frames...")

# Função RGB (para o mouse)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(f"Mouse moved to: {[x]},{[y]}")

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

while True:
    ret, frame = cap.read()
    frame_count += 1
    if frame_count % 3 != 0:
        continue

    if not ret:
        break

    # O código está a funcionar com 1020x600, vamos manter para simplificar
    frame = cv2.resize(frame, (1020, 600))

    results = model.track(frame, persist=True)
            
    if results and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        # Remover objetos que não estão mais na tela
        current_ids = set(ids)
        all_ids = set(object_positions.keys())
        removed_ids = all_ids - current_ids

        for r_id in removed_ids:
            del object_positions[r_id]

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
            
        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            name = model.names[class_id]
            print(f"ID: {track_id}, Center: ({cx}, {cy})")
            
            color = (0, 255, 0)

            # Lógica para verificar se um objeto cruza a linha
            if object_positions[track_id]['status'] is None:
                if cy > line_p1[1]:
                    object_positions[track_id]['status'] = 'IN'
                else:
                    object_positions[track_id]['status'] = 'OUT'

            if object_positions[track_id]['status'] == 'IN' and cy < line_p1[1]:
                out_count += 1
                object_positions[track_id]['status'] = 'OUT'
                color = (255, 0, 0)

            if object_positions[track_id]['status'] == 'OUT' and cy > line_p1[1]:
                in_count += 1
                object_positions[track_id]['status'] = 'IN'
                color = (255, 0, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cvzone.putTextRect(frame, f'{name}', (x1, y1), 1, 1)

    # Desenhar a linha de contagem e exibir contagem (na imagem original)
    cv2.line(frame, line_p1, line_p2, (0, 0, 255), 2)
    
    cvzone.putTextRect(frame, f'IN: {in_count}', (50, 30), 1, 2, colorR=(0, 255, 0), colorT=(255, 255, 255), colorB=(0, 255, 0))
    cvzone.putTextRect(frame, f'OUT: {out_count}', (50, 70), 1, 2, colorR=(0, 255, 0), colorT=(255, 255, 255), colorB=(0, 255, 0))

    cv2.imshow("RGB", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()