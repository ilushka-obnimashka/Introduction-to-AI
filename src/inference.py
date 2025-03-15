import os
import click
import torch
from PIL import Image, ImageDraw, ImageFile
from torchvision.transforms import v2

from config import IMG_SIZE

transform = v2.Compose([
    v2.Resize(IMG_SIZE),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ['abraham_grampa_simpson',
               'agnes_skinner',
               'apu_nahasapeemapetilon',
               'barney_gumble',
               'bart_simpson',
               'carl_carlson',
               'charles_montgomery_burns',
               'chief_wiggum',
               'cletus_spuckler',
               'comic_book_guy',
               'disco_stu',
               'edna_krabappel',
               'fat_tony',
               'gil',
               'groundskeeper_willie',
               'homer_simpson',
               'kent_brockman',
               'krusty_the_clown',
               'lenny_leonard',
               'lionel_hutz',
               'lisa_simpson',
               'maggie_simpson',
               'marge_simpson',
               'martin_prince',
               'mayor_quimby',
               'milhouse_van_houten',
               'miss_hoover',
               'moe_szyslak',
               'ned_flanders',
               'nelson_muntz',
               'otto_mann',
               'patty_bouvier',
               'principal_skinner',
               'professor_john_frink',
               'rainier_wolfcastle',
               'ralph_wiggum',
               'selma_bouvier',
               'sideshow_bob',
               'sideshow_mel',
               'snake_jailbird',
               'troy_mcclure',
               'waylon_smithers']


def inference(model: torch.nn.Module, input_data) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    return output


def check_path(path):
    """Проверяет, является ли путь файлом или директорией."""
    if os.path.isfile(path):
        click.echo(f"'{path}' is a file.")
        return "file"
    elif os.path.isdir(path):
        click.echo(f"'{path}' is a directory.")
        return "directory"
    else:
        click.echo(f"'{path}' is neither a file nor a directory.")


def process_image(image_path: str) -> tuple[Image.Image, torch.Tensor]:
    original_image = Image.open(image_path)
    image = transform(original_image)
    image = image.unsqueeze(0)
    return original_image, image


def process_directory(directory_path: str) -> tuple[list[Image.Image], list[torch.Tensor]]:
    original_images = []
    images_to_model = []
    for filename in os.listdir(directory_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            original_image, image_to_model = process_image(os.path.join(directory_path, filename))
            original_images.append(original_image)
            images_to_model.append(image_to_model)

    return original_images, torch.stack(images_to_model)


@click.command()
@click.option(
    'input_path',
    type=click.Path(exists=True),
    help="Path to a file or directory for input data to model inference"
)
@click.option(
    'model_save_path',
    type=click.Path(exists=True),
    help="The path to the file with a saved Model"
)
def main(**kwargs):
    path_type = check_path(kwargs['input_path'])
    model_save_path = kwargs['model_save_path']

    model = torch.load(model_save_path, map_location=torch.device('cuda:0'))

    if path_type == "file":
        original_image, image_to_model = process_image(kwargs['input_path'])
        output = inference(model=model, input_data=image_to_model)
        predicted_class_index = torch.argmax(output,1).item()
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        draw = ImageDraw.Draw(original_image)
        draw.text((10, 10), predicted_class_name, fill=(0, 0, 0))
        original_image.show()

    if path_type == "directory":
        original_images, images_to_model = process_directory(kwargs['input_path'])
        output = inference(model=model, input_data=images_to_model)
        predicted_class_index = torch.argmax(output, 1).tolist()
        predicted_class_names = [CLASS_NAMES[idx] for idx in predicted_class_index]

        for i, original_image in enumerate(original_images):

            draw = ImageDraw.Draw(original_image)
            draw.text((10, 10), predicted_class_names[i], fill=(0, 0, 0))

            original_image.show()


if __name__ == '__main__':
    main()
