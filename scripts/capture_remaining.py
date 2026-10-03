import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QTimer

app = QApplication(sys.argv)
view = QWebEngineView()
view.resize(1440, 960)
view.show()

def step():
    js = """
    const scopeBtn = document.querySelector(".nav-btn[data-tab='tab-scope']");
    if (scopeBtn) scopeBtn.click();
    window.scrollTo({ top: 0, behavior: 'instant' });
    """
    view.page().runJavaScript(js)
    QTimer.singleShot(2000, do_capture)

def do_capture():
    out_file = 'c:/ReactJS/MachineLearning/reports/figures/hinh_4_9_dac_ta_model_card.png'
    view.grab().save(out_file)
    print('Updated hinh_4_9_dac_ta_model_card.png! Size:', os.path.getsize(out_file))
    app.quit()

view.load(QUrl('http://127.0.0.1:8000/'))
view.loadFinished.connect(lambda ok: QTimer.singleShot(1500, step))
app.exec_()
