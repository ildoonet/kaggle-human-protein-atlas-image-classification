import os
import errno
from multiprocessing.pool import Pool
from tqdm import tqdm
import requests
import pandas as pd
from PIL import Image


def download(pid, image_list, base_url, save_dir, image_size=(512, 512)):
    colors = ['red', 'green', 'blue', 'yellow']
    for i in tqdm(image_list, postfix=pid):
        img_id = i.split('_', 1)
        for color in colors:
            img_path = img_id[0] + '/' + img_id[1] + '_' + color + '.jpg'
            img_name = i + '_' + color + '.png'
            img_url = base_url + img_path

            # Get the raw response from the url
            r = requests.get(img_url, allow_redirects=True, stream=True)
            r.raw.decode_content = True

            # Use PIL to resize the image and to convert it to L
            # (8-bit pixels, black and white)
            im = Image.open(r.raw)
            im = im.resize(image_size, Image.LANCZOS).convert('L')
            im.save(os.path.join(save_dir, img_name), 'PNG')


if __name__ == '__main__':
    # Parameters
    process_num = 24
    image_size = (2048, 2048)
    url = 'http://v18.proteinatlas.org/images/'
    csv_path = "/data/public/rw/kaggle-human-protein-atlas/hpa_v18/HPAv18RBGY_wodpl.csv"
    save_dir = "/data/public/rw/kaggle-human-protein-atlas/hpa_v18/images_2048/"

    # Create the directory to save the images in case it doesn't exist
    try:
        os.makedirs(save_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    print('Parent process %s.' % os.getpid())
    img_list = pd.read_csv(csv_path)['Id']
    list_len = len(img_list)
    p = Pool(process_num)
    for i in range(process_num):
        start = int(i * list_len / process_num)
        end = int((i + 1) * list_len / process_num)
        process_images = img_list[start:end]
        p.apply_async(
            download, args=(str(i), process_images, url, save_dir, image_size)
        )
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
