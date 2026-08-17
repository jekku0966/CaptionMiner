"""A deliberately focused drag-and-drop desktop interface."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from captionminer import __version__
from captionminer.config import COMMON_LANGUAGES, MODEL_PROFILES, options_for_profile
from captionminer.pipeline import transcribe_to_srt
from captionminer.progress import INDETERMINATE_PROGRESS, batch_progress_value, format_elapsed
from captionminer.transcribe import TranscriptionCancelled, TranscriptionEngine


class MediaList(QListWidget):
    paths_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setMinimumHeight(170)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        self.paths_dropped.emit(paths)
        event.acceptProposedAction()


class BatchWorker(QObject):
    progress = Signal(int, str)
    log = Signal(str)
    completed = Signal(int, int, bool)

    def __init__(
        self,
        files: list[Path],
        *,
        profile: str,
        language: str | None,
        device: str,
        initial_prompt: str | None,
        output_directory: Path | None,
        overwrite: bool,
    ) -> None:
        super().__init__()
        self.files = files
        self.options = options_for_profile(
            profile,
            language=language,
            device=device,
            initial_prompt=initial_prompt,
        )
        self.output_directory = output_directory
        self.overwrite = overwrite
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        completed = 0
        failed = 0
        engine = TranscriptionEngine(self.options)
        total = len(self.files)

        for file_index, source in enumerate(self.files):
            if self.cancel_event.is_set():
                break
            self.log.emit(f"[{file_index + 1}/{total}] {source}")

            def callback(
                fraction: float | None,
                message: str,
                current_index: int = file_index,
            ) -> None:
                if fraction is None:
                    self.log.emit(message)
                overall = batch_progress_value(current_index, total, fraction)
                self.progress.emit(overall, f"[{current_index + 1}/{total}] {message}")

            try:
                result = transcribe_to_srt(
                    engine,
                    source,
                    output_directory=self.output_directory,
                    overwrite=self.overwrite,
                    progress=callback,
                    cancel=self.cancel_event.is_set,
                )
                completed += 1
                language = result.metadata.language or "unknown language"
                recovery = (
                    f", recovered {result.metadata.recovered_word_count} word(s)"
                    if result.metadata.recovered_word_count
                    else ""
                )
                self.log.emit(
                    f"Created {result.output} ({result.cue_count} cues, {language}, "
                    f"{result.metadata.device}{recovery})."
                )
            except TranscriptionCancelled:
                self.log.emit("Cancellation requested; no partial SRT was written.")
                break
            except Exception as exc:
                failed += 1
                self.log.emit(f"ERROR: {source.name}: {exc}")

        cancelled = self.cancel_event.is_set()
        processed = completed + failed
        self.progress.emit(
            100 if not cancelled else round(processed / total * 100),
            "Finished" if not cancelled else "Cancelled",
        )
        self.completed.emit(completed, failed, cancelled)

    @Slot()
    def cancel(self) -> None:
        self.cancel_event.set()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"CaptionMiner {__version__}")
        self.resize(840, 670)
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None
        self._close_after_cancel = False
        self._batch_started_at: float | None = None
        self._status_message = "Ready"
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_status_label)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        heading = QLabel("CaptionMiner")
        heading.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(heading)
        description = QLabel(
            "Drop exported clips below. CaptionMiner transcribes them locally and creates "
            "plain SRT subtitle files; styling remains in your video editor."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.media_list = MediaList()
        self.media_list.setToolTip("Drop local video or audio files here")
        self.media_list.paths_dropped.connect(self.add_paths)
        layout.addWidget(self.media_list)

        file_buttons = QHBoxLayout()
        self.add_button = QPushButton("Add files")
        self.remove_button = QPushButton("Remove selected")
        self.clear_button = QPushButton("Clear")
        file_buttons.addWidget(self.add_button)
        file_buttons.addWidget(self.remove_button)
        file_buttons.addWidget(self.clear_button)
        file_buttons.addStretch(1)
        layout.addLayout(file_buttons)

        settings_box = QGroupBox("Transcription")
        settings = QFormLayout(settings_box)
        self.profile_combo = QComboBox()
        for profile in MODEL_PROFILES.values():
            self.profile_combo.addItem(f"{profile.label} — {profile.model_name}", profile.key)
        self.profile_combo.setCurrentIndex(1)
        settings.addRow("Accuracy profile", self.profile_combo)

        self.language_combo = QComboBox()
        for label, code in COMMON_LANGUAGES:
            self.language_combo.addItem(label, code)
        settings.addRow("Spoken language", self.language_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Automatic", "auto")
        self.device_combo.addItem("NVIDIA CUDA", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        settings.addRow("Processing device", self.device_combo)

        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText(
            "Optional names/terms: Snarkos, Veadotube, TwitchDownloader"
        )
        settings.addRow("Custom vocabulary", self.prompt_edit)

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Empty = create each SRT beside its source clip")
        self.output_browse = QPushButton("Browse")
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.output_browse)
        settings.addRow("Output folder", output_row)

        self.overwrite_check = QCheckBox("Overwrite an existing same-named SRT")
        settings.addRow("", self.overwrite_check)
        layout.addWidget(settings_box)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("Transcribe")
        self.start_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.cancel_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setMinimumHeight(120)
        layout.addWidget(self.log)

        self.setCentralWidget(root)
        self.add_button.clicked.connect(self.choose_files)
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.media_list.clear)
        self.output_browse.clicked.connect(self.choose_output_directory)
        self.start_button.clicked.connect(self.start_batch)
        self.cancel_button.clicked.connect(self.cancel_batch)

    @Slot(list)
    def add_paths(self, paths: list[str]) -> None:
        existing = {self.media_list.item(i).text() for i in range(self.media_list.count())}
        for value in paths:
            path = Path(value).expanduser().resolve()
            if path.is_file() and str(path) not in existing:
                self.media_list.addItem(str(path))
                existing.add(str(path))

    @Slot()
    def choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video or audio files",
            "",
            "Media files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mp3 *.wav "
            "*.m4a *.aac *.flac *.ogg *.opus);;All files (*)",
        )
        self.add_paths(files)

    @Slot()
    def remove_selected(self) -> None:
        for item in self.media_list.selectedItems():
            self.media_list.takeItem(self.media_list.row(item))

    @Slot()
    def choose_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose subtitle output folder")
        if selected:
            self.output_edit.setText(selected)

    def _set_running(self, running: bool) -> None:
        for widget in (
            self.add_button,
            self.remove_button,
            self.clear_button,
            self.start_button,
            self.profile_combo,
            self.language_combo,
            self.device_combo,
            self.prompt_edit,
            self.output_edit,
            self.output_browse,
            self.overwrite_check,
            self.media_list,
        ):
            widget.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _set_progress_value(self, value: int) -> None:
        if value == INDETERMINATE_PROGRESS:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setTextVisible(False)
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setValue(max(0, min(100, value)))

    def _set_status_message(self, message: str) -> None:
        self._status_message = message
        self._refresh_status_label()

    @Slot()
    def _refresh_status_label(self) -> None:
        message = self._status_message
        if self._batch_started_at is not None:
            elapsed = format_elapsed(time.monotonic() - self._batch_started_at)
            message = f"{message} • elapsed {elapsed}"
        self.status_label.setText(message)

    @Slot()
    def start_batch(self) -> None:
        files = [Path(self.media_list.item(i).text()) for i in range(self.media_list.count())]
        if not files:
            QMessageBox.information(self, "CaptionMiner", "Add at least one media file first.")
            return

        output_text = self.output_edit.text().strip()
        output_directory = Path(output_text).expanduser() if output_text else None
        if (
            output_directory is not None
            and output_directory.exists()
            and not output_directory.is_dir()
        ):
            QMessageBox.warning(self, "CaptionMiner", "The selected output path is not a folder.")
            return

        self.log.clear()
        self._batch_started_at = time.monotonic()
        self._set_progress_value(INDETERMINATE_PROGRESS)
        self._set_status_message("Starting...")
        self._status_timer.start()
        self._set_running(True)

        self._thread = QThread(self)
        self._worker = BatchWorker(
            files,
            profile=str(self.profile_combo.currentData()),
            language=self.language_combo.currentData(),
            device=str(self.device_combo.currentData()),
            initial_prompt=self.prompt_edit.text().strip() or None,
            output_directory=output_directory,
            overwrite=self.overwrite_check.isChecked(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.on_progress)
        self._worker.log.connect(self.log.appendPlainText)
        self._worker.completed.connect(self.on_completed)
        self._worker.completed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @Slot()
    def cancel_batch(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)
            self._set_status_message("Cancelling after the current safe checkpoint...")

    @Slot(int, str)
    def on_progress(self, value: int, message: str) -> None:
        self._set_progress_value(value)
        self._set_status_message(message)

    @Slot(int, int, bool)
    def on_completed(self, completed: int, failed: int, cancelled: bool) -> None:
        self._set_running(False)
        self._status_timer.stop()
        if cancelled:
            message = f"Cancelled. Created {completed} subtitle file(s); {failed} failed."
        else:
            message = f"Finished. Created {completed} subtitle file(s); {failed} failed."
        self._set_status_message(message)
        self._batch_started_at = None
        self.log.appendPlainText(message)
        self._worker = None
        self._thread = None
        if self._close_after_cancel:
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is None:
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            "CaptionMiner is running",
            "Cancel the current batch and close after processing reaches a safe checkpoint?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self._close_after_cancel = True
            self._worker.cancel()
            self._set_status_message("Cancelling after the current safe checkpoint...")
            event.ignore()
        else:
            event.ignore()


def run_gui() -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("CaptionMiner")
    app.setOrganizationName("CaptionMiner")
    window = MainWindow()
    window.show()
    return app.exec()
