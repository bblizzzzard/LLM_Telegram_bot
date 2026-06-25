import contextlib
import glob
import io
import json
import os
import sys
import traceback


#Скрипт песочницы - отдельный процесс на каждый вызов кода
#Запуск: python sandbox_runner.py <data_path> <output_dir> <cell_path>
#Грузит таблицу в df, выполняет код модели, ловит stdout/ошибки и графики
#Результат печатает одной JSON-строкой с меткой __SANDBOX_RESULT__
#Падение или утечка памяти убивают только этот процесс, а не бота


def smart_load(path):
    #Загрузка CSV/TSV/Excel в DataFrame: перебираем разделители и кодировки
    #Для Excel берём первый лист
    import pandas as pd

    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""

    if ext in ("xlsx", "xls", "xlsm"):
        return pd.read_excel(path)
    if ext == "tsv":
        return pd.read_csv(path, sep="\t")

    attempts = [
        dict(),
        dict(sep=None, engine="python"),
        dict(sep=";"),
        dict(encoding="cp1251"),
        dict(sep=";", encoding="cp1251"),
        dict(encoding="latin-1"),
    ]
    last_err = None
    best = None
    for kw in attempts:
        try:
            frame = pd.read_csv(path, **kw)
        except Exception as exc:  #пробуем следующий вариант
            last_err = exc
            continue
        if frame.shape[1] > 1:
            return frame
        best = best if best is not None else frame
    if best is not None:
        return best
    raise last_err if last_err else ValueError("Не удалось разобрать файл")


def main():
    data_path, output_dir, cell_path = sys.argv[1], sys.argv[2], sys.argv[3]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(output_dir, "*.png")))

    #Окружение, которое видит код модели
    namespace = {
        "pd": pd, "np": np, "plt": plt,
        "DATA_PATH": data_path, "OUTPUT_DIR": output_dir,
        "__name__": "__sandbox__",
    }

    load_error = None
    try:
        namespace["df"] = smart_load(data_path)
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        namespace["df"] = None

    with open(cell_path, "r", encoding="utf-8") as fh:
        code = fh.read()

    #Выполняем код модели, перехватывая весь вывод и ошибки
    out_buf = io.StringIO()
    error_text = None
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(out_buf):
            exec(compile(code, "<cell>", "exec"), namespace)
    except Exception:
        error_text = traceback.format_exc(limit=4)

    #Сохраняем все открытые фигуры, у которых есть содержимое
    base = os.path.splitext(os.path.basename(cell_path))[0]
    images = []
    for i, num in enumerate(plt.get_fignums()):
        fig = plt.figure(num)
        if not fig.get_axes():
            continue
        path = os.path.join(output_dir, f"{base}_fig_{i}.png")
        try:
            fig.savefig(path, bbox_inches="tight", dpi=130)
            title = None
            if getattr(fig, "_suptitle", None) is not None:
                title = fig._suptitle.get_text() or None
            if not title and fig.get_axes():
                title = fig.get_axes()[0].get_title() or None
            images.append({"path": path, "title": title})
        except Exception:
            pass
    plt.close("all")

    #Подхватываем картинки, если модель сохранила их вручную
    after = set(glob.glob(os.path.join(output_dir, "*.png")))
    known = {im["path"] for im in images}
    for path in sorted(after - before - known):
        images.append({"path": path, "title": None})

    stdout_text = out_buf.getvalue()
    if len(stdout_text) > 12000:
        stdout_text = stdout_text[:12000] + "\n...[вывод обрезан]..."

    result = {
        "stdout": stdout_text,
        "error": error_text,
        "load_error": load_error,
        "images": images,
    }
    #отдаём результат родительскому процессу
    sys.stdout.write("\n__SANDBOX_RESULT__" + json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
