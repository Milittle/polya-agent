"""测试基线：界面文案以 zh 断言为主，锁 POLYA_LANG 防本机环境干扰。

默认语言已改为 en（见 polya/i18n.py），测试套件的中文断言在此固定基线；
显式设置 POLYA_LANG 的环境（如 CI 想跑 en 基线）不会被覆盖。
"""

import os

os.environ.setdefault("POLYA_LANG", "zh")
