import os
import cv2


def draw_bboxes_on_image(image_path, label_file_path, output_path):
    img = cv2.imread(image_path)

    # Ler o arquivo de labels YOLO

    label_result = None
    with open(label_file_path, "r") as label_file:
        for line in label_file:
            # Resgata as informações da BoundingBox

            label, x_center, y_center, width, height = map(float, line.split())

            h, w, _ = img.shape

            # Calcula as coordenadas da BoundingBox
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            color = (100, 255, 0)

            # Desenhamos a bounding box correspondente na imagem

            label_result = (label, x_center, y_center, width, height)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Salvamos a imagem para avaliação posterior
    cv2.imwrite(output_path, img)

    return label_result


def iterate_images_and_labels(input_images_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    labels_dir = input_images_dir.replace("images", "labels")
    labels_list = []

    for label_file in os.listdir(labels_dir):
        if label_file.endswith(".txt"):

            # Assumimos que o arquivo de label tem o mesmo nome da imagem
            image_name = label_file.replace(".txt", ".jpg")

            image_path = os.path.join(input_images_dir, image_name)
            label_file_path = os.path.join(labels_dir, label_file)
            output_image_path = os.path.join(output_dir, image_name)

            if os.path.exists(image_path):
                # Resgata nova imagem agora com a boundingbox desenhada
                label_result = draw_bboxes_on_image(
                    image_path, label_file_path, output_image_path
                )
                labels_list.append(label_result)

    print("Imagens para avaliação visual criadas.")
    return labels_list
