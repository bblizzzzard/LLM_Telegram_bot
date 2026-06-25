import json
import os
import subprocess
import sys


#Интерпретатор кода - инструмент, который агент вызывает для анализа
#Каждый run() запускает новый подпроцесс sandbox_runner.py
#У подпроцесса таймаут, лимиты ресурсов (на Linux) и нет доступа к секретам бота
class CodeInterpreter:
    def __init__(self, data_path, session_dir, timeout=45, mem_mb=2048):
        self.data_path = os.path.abspath(data_path)
        self.session_dir = os.path.abspath(session_dir)
        self.output_dir = os.path.join(self.session_dir, "charts")
        os.makedirs(self.output_dir, exist_ok=True)
        self.timeout = timeout
        self.mem_mb = mem_mb
        self.runner = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "sandbox_runner.py"
        )
        self._n = 0

    def _preexec(self):
        #Лимиты ресурсов, применяемые в дочернем процессе до exec (только POSIX/Linux)
        def _set_limits():
            try:
                import resource
                cpu = max(5, int(self.timeout))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 2))
                if self.mem_mb:
                    nbytes = self.mem_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
                resource.setrlimit(
                    resource.RLIMIT_FSIZE, (200 * 1024 * 1024, 200 * 1024 * 1024)
                )
            except Exception:
                pass
        return _set_limits

    def run(self, code):
        #Пишем код модели в файл и запускаем его отдельным процессом
        self._n += 1
        cell_path = os.path.join(self.session_dir, f"cell_{self._n}.py")
        with open(cell_path, "w", encoding="utf-8") as fh:
            fh.write(code)

        #Минимальное окружение: только PATH/локаль/HOME и бэкенд matplotlib.
        #Секреты (OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN) в подпроцесс НЕ идут.

        env = {
            "PATH": os.environ.get("PATH", ""),
            "MPLBACKEND": "Agg",
            "HOME": self.session_dir,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }

        try:
            proc = subprocess.run(
                [sys.executable, self.runner, self.data_path,
                 self.output_dir, cell_path],
                capture_output=True, text=True, timeout=self.timeout,
                cwd=self.session_dir, env=env,
                preexec_fn=self._preexec() if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return {"stdout": "", "error": f"Превышен таймаут {self.timeout}с. "
                    "Пиши более лёгкий код.", "load_error": None, "images": []}

        #Результат процесс печатает одной JSON-строкой после метки __SANDBOX_RESULT__
        out = proc.stdout or ""
        marker = "__SANDBOX_RESULT__"
        if marker in out:
            _, _, tail = out.partition(marker)
            try:
                return json.loads(tail.strip().splitlines()[0])
            except Exception as exc:
                return {"stdout": out, "error": f"Ошибка разбора результата: {exc}",
                        "load_error": None, "images": []}

        #метки нет - процесс упал или убит по лимиту
        return {"stdout": out,
                "error": (proc.stderr or "Процесс не вернул результат "
                          "(возможно, убит по лимиту ресурсов).").strip(),
                "load_error": None, "images": []}
