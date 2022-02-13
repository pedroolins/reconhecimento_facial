## importando as funções necessárias
import cv2

## classificador de face
haar_cascade_path = "classificadores/haarcascade_frontalface_default.xml"
modelo = cv2.CascadeClassifier(haar_cascade_path)

# instanciando o modelo
modelo_lbph = cv2.face.LBPHFaceRecognizer_create()
# treinando o modelo
modelo_lbph.read("models/modelo_treinado.yml")

DEPARA_PESSOAS = {1: "JOAO PEDRO", 3: "BEA", 2: "MARCIA"}

## abrindo a webcam
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

while not cv2.waitKey(20) & 0xFF == ord("q"):
    # captura frame
    ret, frame_video = capture.read()
    img_gray = cv2.cvtColor(frame_video, cv2.COLOR_BGR2GRAY)
    ## detectando face
    faces = modelo.detectMultiScale(img_gray, 1.3, 5)
    for x, y, w, h in faces:
        roi_gray = img_gray[y : y + h, x : x + w]
        pred, _ = modelo_lbph.predict(roi_gray)
        pessoa = DEPARA_PESSOAS.get(pred)
        # desenhando o retângulo do rosto
        cv2.rectangle(
            frame_video,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )
        # escrevendo o nome da pessoa detectada
        cv2.putText(
            frame_video,
            f"{pessoa}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
        )
    ## escrevendo o número de pessoas detectadas
    cv2.putText(
        frame_video,
        f"Pessoas detectadas: {len(faces)}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
    )
    cv2.imshow("Computer vision", frame_video)
capture.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
