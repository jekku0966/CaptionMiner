"""Shared Miner-family desktop theme for CaptionMiner."""

from __future__ import annotations

from typing import Any

MINER_COLORS: dict[str, str] = {
    "background": "#0D1117",
    "panel": "#171E27",
    "panel_raised": "#1B2531",
    "sidebar": "#111821",
    "text": "#EEF2F6",
    "muted_text": "#A9B3BE",
    "primary": "#E8A63A",
    "primary_hover": "#F2B84B",
    "border": "#303A46",
    "danger": "#E06C75",
}


MINER_STYLESHEET = """
QWidget {
    color: #EEF2F6;
    font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
    font-size: 10pt;
}

QMainWindow, QDialog, QMessageBox, QWidget#centralRoot {
    background-color: #0D1117;
}

QWidget#brandHeader {
    background-color: #111821;
    border-bottom: 2px solid #E8A63A;
}

QWidget#brandHeader QLabel {
    background-color: transparent;
}

QLabel[role="brandMark"] {
    background-color: #E8A63A;
    color: #0D1117;
    border-radius: 6px;
    font-weight: 700;
    padding: 5px 7px;
}

QLabel[role="heading"] {
    color: #EEF2F6;
    font-size: 22px;
    font-weight: 700;
}

QLabel[role="family"] {
    color: #F2B84B;
    border: 1px solid #303A46;
    border-radius: 7px;
    padding: 2px 6px;
    font-size: 9px;
    font-weight: 600;
}

QLabel[role="muted"] {
    color: #A9B3BE;
}

QLabel[role="status"] {
    color: #F2B84B;
    font-weight: 600;
}

QGroupBox {
    background-color: #171E27;
    border: 1px solid #303A46;
    border-radius: 7px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: #EEF2F6;
}

QLineEdit, QComboBox, QPlainTextEdit, QListWidget {
    background-color: #111821;
    color: #EEF2F6;
    border: 1px solid #303A46;
    border-radius: 5px;
    selection-background-color: #E8A63A;
    selection-color: #0D1117;
}

QLineEdit, QComboBox {
    min-height: 28px;
    padding: 2px 8px;
}

QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QListWidget:focus {
    border: 1px solid #E8A63A;
}

QLineEdit:read-only {
    color: #A9B3BE;
    background-color: #171E27;
}

QComboBox::drop-down {
    border: 0;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #171E27;
    color: #EEF2F6;
    border: 1px solid #303A46;
    selection-background-color: #E8A63A;
    selection-color: #0D1117;
}

QListWidget::item {
    padding: 6px;
}

QListWidget::item:selected {
    background-color: #1B2531;
    color: #EEF2F6;
    border-left: 3px solid #E8A63A;
}

QPushButton {
    background-color: #1B2531;
    color: #EEF2F6;
    border: 1px solid #303A46;
    border-radius: 5px;
    min-height: 28px;
    padding: 3px 12px;
}

QPushButton:hover {
    background-color: #253241;
    border-color: #E8A63A;
}

QPushButton:pressed {
    background-color: #111821;
}

QPushButton:disabled {
    color: #68727E;
    background-color: #171E27;
    border-color: #252E38;
}

QPushButton[primary="true"] {
    background-color: #E8A63A;
    color: #0D1117;
    border-color: #E8A63A;
    font-weight: 700;
}

QPushButton[primary="true"]:hover {
    background-color: #F2B84B;
    border-color: #F2B84B;
}

QCheckBox, QRadioButton {
    spacing: 7px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
}

QProgressBar {
    background-color: #1B2531;
    color: #EEF2F6;
    border: 1px solid #303A46;
    border-radius: 6px;
    min-height: 12px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #E8A63A;
    border-radius: 5px;
}

QScrollBar:vertical {
    background-color: #111821;
    width: 12px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #303A46;
    border-radius: 5px;
    min-height: 28px;
}

QScrollBar::handle:vertical:hover {
    background-color: #E8A63A;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
    height: 0;
}

QToolTip {
    background-color: #1B2531;
    color: #EEF2F6;
    border: 1px solid #E8A63A;
    padding: 4px;
}
"""


def apply_miner_theme(application: Any) -> None:
    """Apply the HighlightMiner-derived palette and Qt stylesheet."""

    from PySide6.QtGui import QColor, QPalette

    application.setStyle("Fusion")
    palette = application.palette()
    palette.setColor(QPalette.Window, QColor(MINER_COLORS["background"]))
    palette.setColor(QPalette.WindowText, QColor(MINER_COLORS["text"]))
    palette.setColor(QPalette.Base, QColor(MINER_COLORS["sidebar"]))
    palette.setColor(QPalette.AlternateBase, QColor(MINER_COLORS["panel"]))
    palette.setColor(QPalette.Text, QColor(MINER_COLORS["text"]))
    palette.setColor(QPalette.Button, QColor(MINER_COLORS["panel_raised"]))
    palette.setColor(QPalette.ButtonText, QColor(MINER_COLORS["text"]))
    palette.setColor(QPalette.Highlight, QColor(MINER_COLORS["primary"]))
    palette.setColor(QPalette.HighlightedText, QColor(MINER_COLORS["background"]))
    palette.setColor(QPalette.Link, QColor(MINER_COLORS["primary_hover"]))
    application.setPalette(palette)
    application.setStyleSheet(MINER_STYLESHEET)
