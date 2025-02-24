import cv2


def equalize_brightness_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # CLAHE significa "Contrast Limited Adaptive Histogram Equalization"
    # Processa o contraste das imagens realizando uma equalização
    # Deixa áreas escuras mais claras e áreas muito claras mais escuras
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Aplicamos a equalização apenas no canal Value mantendo
    # a cor e a saturação
    v_eq = clahe.apply(v)

    hsv_eq = cv2.merge((h, s, v_eq))
    img = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

    return img
