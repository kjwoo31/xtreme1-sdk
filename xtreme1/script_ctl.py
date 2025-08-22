import argparse
import os
import shutil
import json
from xtreme1.exporter.converter import Result
from xtreme1.importer.parser import Parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='import or export', choices=['import', 'export'], required=True)
    parser.add_argument('--src', type=str, help='source path(It is a folder for import or a zip package for export)',
                        required=True)
    parser.add_argument('--dst', type=str, help='The path to save the results', required=True)
    parser.add_argument('--rps', type=str, default=None, help='The json file in which the response is stored,'
                                                              'if none, the output is in the console')
    parser.add_argument('--fmt', type=str, default='xtreme1', help='object format(xtreme1,coco,kitti,voc,labelme)',
                        choices=['coco', 'voc', 'labelme', 'kitti', 'kitti_like', 'nuscenes'], required=True)
    args = parser.parse_args()

    mode = args.mode
    src_path = args.src
    dst_path = args.dst
    response = args.rps
    format = args.fmt

    try:
        if mode == 'export':
            zip_file_created = False
            zip_name = ''
            if os.path.isdir(src_path):
                print("src_path is directory. Making zip folder to upload xtreme1")
                zip_name = src_path.rstrip(os.sep)  # remove trailing slash if any
                zip_file = shutil.make_archive(zip_name, 'zip', src_path)
                src_path = zip_file
                zip_file_created = True

            anno = Result(src_zipfile=src_path)
            anno.convert(format=format, export_folder=dst_path)

            if zip_file_created:
                os.remove(zip_name + '.zip')
            code = 'OK'
            message = ''

        else:
            data_parser = Parser(source_path=src_path)
            message = data_parser.parser(format=format, output=dst_path)
            if not message:
                code = "OK"
            else:
                code = 'ERROR'

    except Exception as e:
        code = "ERROR"
        message = str(e)

    rps_data = {
        "code": code,
        "message": message
    }
    if not response:
        print(rps_data)
    else:
        with open(response, 'w', encoding='utf-8') as rf:
            json.dump(rps_data, rf, ensure_ascii=False)


if __name__ == '__main__':
    main()
