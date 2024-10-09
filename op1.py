import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregar a imagem
def load_image(image_path):
    return cv2.imread(image_path)

# 1. Mudança de brilho e contraste
def change_brightness_contrast(image, brightness=0, contrast=0):
    result = cv2.convertScaleAbs(image, alpha=1 + contrast / 100.0, beta=brightness)
    return result

# 2. Redimensionamento da imagem
def resize_image(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# 3. Rotação da imagem
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# 4. Corte da imagem
def crop_image(image, start_x, start_y, width, height):
    return image[start_y:start_y+height, start_x:start_x+width]

# 5. Filtragem para suavizar ou realçar
def apply_filter(image, filter_type="blur"):
    if filter_type == "blur":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    return image

# 6. Segmentação (exemplo com detecção de bordas)
def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# 7. Equalização de histograma
def equalize_histogram(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # Color image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# Exibição de imagens
def show_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Caminho da imagem
    image_path = 'C:/Users/Elen/Desktop/op1/cruzeiro.jpg'
    image = load_image(image_path)

    # 1. Mudança de brilho e contraste
    bright_contrast_image = change_brightness_contrast(image, brightness=40, contrast=30)
    show_image(bright_contrast_image, "Brilho e Contraste Ajustados")

    # 2. Redimensionamento
    resized_image = resize_image(image, width=400)
    show_image(resized_image, "Imagem Redimensionada")

    # 3. Rotação
    rotated_image = rotate_image(image, angle=45)
    show_image(rotated_image, "Imagem Rotacionada")

    # 4. Corte
    cropped_image = crop_image(image, start_x=100, start_y=100, width=200, height=200)
    show_image(cropped_image, "Imagem Cortada")

    # 5. Filtragem
    blurred_image = apply_filter(image, filter_type="blur")
    show_image(blurred_image, "Imagem Suavizada (Blur)")

    sharpened_image = apply_filter(image, filter_type="sharpen")
    show_image(sharpened_image, "Imagem Realçada (Sharpen)")

    # 6. Segmentação (Detecção de Bordas)
    edges = edge_detection(image)
    show_image(edges, "Detecção de Bordas")

    # 7. Equalização de Histograma
    equalized_image = equalize_histogram(image)
    show_image(equalized_image, "Equalização de Histograma")
