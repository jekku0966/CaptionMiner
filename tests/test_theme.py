from __future__ import annotations

from captionminer.theme import MINER_COLORS, MINER_STYLESHEET


def test_miner_theme_matches_highlightminer_brand_palette() -> None:
    assert MINER_COLORS == {
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


def test_miner_theme_styles_primary_actions_progress_and_focus() -> None:
    for selector in (
        'QPushButton[primary="true"]',
        "QProgressBar::chunk",
        "QLineEdit:focus",
        "QListWidget::item:selected",
    ):
        assert selector in MINER_STYLESHEET

    assert MINER_COLORS["primary"] in MINER_STYLESHEET
    assert MINER_COLORS["background"] in MINER_STYLESHEET
