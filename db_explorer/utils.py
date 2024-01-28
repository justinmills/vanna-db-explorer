import pathlib


class Utils:
    def __init__(self):
        pass

    def read_file(self, file_path: str, file_type: str) -> str:
        return pathlib.Path(file_path).read_text(encoding="utf-8")

    def read_files(self, path: str, file_type: str) -> str:
        data = []
        for p in pathlib.Path(path).glob(f"*.{file_type}"):
            data.append(p.read_text(encoding="utf-8"))
        return "\n".join(data)
