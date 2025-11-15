# UI/style.py

GLOBAL_STYLE = """
    QWidget {
        background-color: #f5f6fa;
        color: #333;
        font-size: 15px;
    }
    QPushButton {
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        border-radius: 8px;
        padding: 8px 14px;
    }
    QPushButton:hover {
        background-color: #f0f0f0;
    }
"""

CARD_STYLE = """
    QFrame {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
    }
"""

SIDEBAR_STYLE = """
    QPushButton {
        background-color: transparent;
        color: #555;
        padding: 10px;
        border-radius: 6px;
        text-align: left;
    }
    QPushButton:hover {
        background-color: #dfe6f3;
    }
    QPushButton:checked {
        background-color: #4e73df;
        color: white;
    }
"""
