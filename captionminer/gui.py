"""A deliberately focused drag-and-drop desktop interface."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from captionminer import __version__
from captionminer.config import (
    COMMON_LANGUAGES,
    MODEL_PROFILES,
    TranscriptionOptions,
    options_for_profile,
)
from captionminer.diagnostics import (
    BatchDiagnostics,
    DiagnosticPreferences,
    DiagnosticSession,
)
from captionminer.model_management import (
    CUSTOM_MODEL_KEY,
    DownloadConsentAction,
    DownloadPolicy,
    ModelPreferences,
    ModelSelection,
    apply_download_consent_action,
    huggingface_cache_directory,
    resolve_cached_model,
    resolve_custom_model,
)
from captionminer.pipeline import transcribe_to_srt
from captionminer.progress import INDETERMINATE_PROGRESS, batch_progress_value, format_elapsed
from captionminer.theme import apply_miner_theme
from captionminer.transcribe import TranscriptionCancelled, TranscriptionEngine

CUSTOM_MODEL_BASE_PROFILE = "balanced"


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
        options: TranscriptionOptions,
        output_directory: Path | None,
        overwrite: bool,
        diagnostics: BatchDiagnostics,
    ) -> None:
        super().__init__()
        self.files = files
        self.options = options
        self.output_directory = output_directory
        self.overwrite = overwrite
        self.diagnostics = diagnostics
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
            file_diagnostics = self.diagnostics.start_file(file_index + 1, source)

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
                    diagnostics=file_diagnostics,
                )
                completed += 1
                file_diagnostics.finish("completed")
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
                file_diagnostics.finish("cancelled")
                self.log.emit("Cancellation requested; no partial SRT was written.")
                break
            except Exception as exc:
                failed += 1
                file_diagnostics.log_exception("file_failed", exc, stage="pipeline")
                file_diagnostics.finish("failed")
                self.log.emit(f"ERROR: {source.name}: {exc}")

        cancelled = self.cancel_event.is_set()
        processed = completed + failed
        self.progress.emit(
            100 if not cancelled else round(processed / total * 100),
            "Finished" if not cancelled else "Cancelled",
        )
        self.diagnostics.finish(
            completed_count=completed,
            failed_count=failed,
            cancelled=cancelled,
        )
        self.completed.emit(completed, failed, cancelled)

    @Slot()
    def cancel(self) -> None:
        self.diagnostics.record("cancellation_requested", level="warning")
        self.cancel_event.set()


class ModelSettingsDialog(QDialog):
    """Small user-facing model and download-preference editor."""

    def __init__(
        self,
        preferences: ModelPreferences,
        diagnostic_preferences: DiagnosticPreferences,
        diagnostic_session: DiagnosticSession,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.preferences = preferences
        self.diagnostic_preferences = diagnostic_preferences
        self.diagnostic_session = diagnostic_session
        self.custom_model_changed = False
        self.setWindowTitle("CaptionMiner Settings")
        self.resize(680, 500)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Control model downloads and configure one custom local speech-recognition "
            "model. Accuracy profiles are selected on the main screen."
        )
        explanation.setWordWrap(True)
        explanation.setProperty("role", "muted")
        layout.addWidget(explanation)

        form = QFormLayout()
        self.policy_combo = QComboBox()
        self.policy_combo.addItem("Ask before downloading", DownloadPolicy.ASK)
        self.policy_combo.addItem("Download automatically", DownloadPolicy.ALLOW)
        self.policy_combo.addItem("Never download automatically", DownloadPolicy.DENY)
        policy_index = self.policy_combo.findData(self.preferences.download_policy)
        self.policy_combo.setCurrentIndex(max(0, policy_index))
        form.addRow("When a model is missing", self.policy_combo)

        self.custom_model_path = QLineEdit()
        self.custom_model_path.setReadOnly(True)
        self.custom_model_path.setPlaceholderText("No custom model configured")
        form.addRow("Custom model folder", self.custom_model_path)
        layout.addLayout(form)

        self.model_status = QLabel()
        self.model_status.setWordWrap(True)
        self.model_status.setProperty("role", "muted")
        layout.addWidget(self.model_status)

        model_buttons = QHBoxLayout()
        self.choose_local_button = QPushButton("Choose custom model folder")
        self.clear_local_button = QPushButton("Clear custom model")
        self.open_cache_button = QPushButton("Open downloaded model folder")
        model_buttons.addWidget(self.choose_local_button)
        model_buttons.addWidget(self.clear_local_button)
        model_buttons.addStretch(1)
        model_buttons.addWidget(self.open_cache_button)
        layout.addLayout(model_buttons)

        logging_box = QGroupBox("Diagnostic logging")
        logging_layout = QVBoxLayout(logging_box)
        self.standard_logging_radio = QRadioButton("Standard logging")
        self.detailed_logging_radio = QRadioButton("Detailed diagnostics for next batch")
        detailed_next = self.diagnostic_preferences.detailed_next_batch
        self.detailed_logging_radio.setChecked(detailed_next)
        self.standard_logging_radio.setChecked(not detailed_next)
        logging_layout.addWidget(self.standard_logging_radio)
        logging_layout.addWidget(self.detailed_logging_radio)

        logging_explanation = QLabel(
            "Standard logging is always local and always on. Detailed diagnostics adds "
            "redacted timings and counts to the next batch only, then returns to Standard."
        )
        logging_explanation.setWordWrap(True)
        logging_explanation.setProperty("role", "muted")
        logging_layout.addWidget(logging_explanation)

        logging_buttons = QHBoxLayout()
        self.open_logs_button = QPushButton("Open log folder")
        self.copy_summary_button = QPushButton("Copy diagnostic summary")
        self.delete_logs_button = QPushButton("Delete logs")
        logging_buttons.addWidget(self.open_logs_button)
        logging_buttons.addWidget(self.copy_summary_button)
        logging_buttons.addStretch(1)
        logging_buttons.addWidget(self.delete_logs_button)
        logging_layout.addLayout(logging_buttons)
        layout.addWidget(logging_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.policy_combo.currentIndexChanged.connect(self._save_policy)
        self.choose_local_button.clicked.connect(self._choose_local_model)
        self.clear_local_button.clicked.connect(self._clear_local_model)
        self.open_cache_button.clicked.connect(self._open_cache)
        self.detailed_logging_radio.toggled.connect(self._save_diagnostic_mode)
        self.open_logs_button.clicked.connect(self._open_log_folder)
        self.copy_summary_button.clicked.connect(self._copy_diagnostic_summary)
        self.delete_logs_button.clicked.connect(self._delete_logs)
        self._refresh_status()

    @Slot()
    def _save_policy(self) -> None:
        self.preferences.set_download_policy(self.policy_combo.currentData())

    @Slot()
    def _refresh_status(self) -> None:
        lookup = resolve_custom_model(self.preferences)
        saved_path = self.preferences.custom_model_path()
        self.custom_model_path.setText(str(saved_path) if saved_path is not None else "")

        if lookup.selection is not None:
            text = (
                "Custom model validated and ready. CaptionMiner will select it on the main "
                "screen and use the Balanced transcription behavior."
            )
        elif lookup.invalid_local_reason:
            text = f"Saved custom model cannot be used:\n{lookup.invalid_local_reason}"
        else:
            text = (
                "Choose a compatible local faster-whisper model folder. Once configured, "
                "it appears as Custom in the main Accuracy profile / model list."
            )

        self.model_status.setText(text)
        self.clear_local_button.setEnabled(saved_path is not None)

    @Slot()
    def _choose_local_model(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose a faster-whisper model folder",
        )
        if not selected:
            return
        try:
            self.preferences.set_custom_model_path(Path(selected))
        except ValueError as exc:
            self.diagnostic_session.log_exception(
                "custom_model_validation_failed",
                exc,
                level="warning",
                secrets=(selected,),
            )
            QMessageBox.warning(self, "Invalid model folder", str(exc))
            return
        self.custom_model_changed = True
        self._refresh_status()

    @Slot()
    def _clear_local_model(self) -> None:
        self.preferences.clear_custom_model_path()
        self.custom_model_changed = True
        self._refresh_status()

    @Slot()
    def _open_cache(self) -> None:
        cache = huggingface_cache_directory()
        if not cache.is_dir():
            QMessageBox.information(
                self,
                "Downloaded model folder",
                f"No downloaded model folder exists yet.\n\nExpected location:\n{cache}",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(cache)))

    @Slot(bool)
    def _save_diagnostic_mode(self, detailed: bool) -> None:
        self.diagnostic_preferences.set_detailed_next_batch(detailed)

    @Slot()
    def _open_log_folder(self) -> None:
        try:
            self.diagnostic_session.log_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.diagnostic_session.log_exception("log_folder_open_failed", exc)
            QMessageBox.warning(self, "Diagnostic logs", "The local log folder could not open.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.diagnostic_session.log_directory)))

    @Slot()
    def _copy_diagnostic_summary(self) -> None:
        summary = self.diagnostic_session.summary(
            detailed_next_batch=self.diagnostic_preferences.detailed_next_batch
        )
        QApplication.clipboard().setText(summary)
        QMessageBox.information(
            self,
            "Diagnostic summary",
            "A redacted diagnostic summary was copied to the clipboard.",
        )

    @Slot()
    def _delete_logs(self) -> None:
        answer = QMessageBox.question(
            self,
            "Delete diagnostic logs",
            "Delete all local CaptionMiner diagnostic logs? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        deleted = self.diagnostic_session.delete_logs()
        QMessageBox.information(
            self,
            "Diagnostic logs",
            f"Deleted {deleted} local diagnostic log file(s).",
        )


class MainWindow(QMainWindow):
    def __init__(
        self,
        preferences: ModelPreferences | None = None,
        diagnostic_preferences: DiagnosticPreferences | None = None,
        diagnostic_session: DiagnosticSession | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"CaptionMiner {__version__}")
        self.resize(840, 670)
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None
        self._close_after_cancel = False
        self._batch_started_at: float | None = None
        self._status_message = "Ready"
        self._model_preferences = preferences or ModelPreferences()
        self._diagnostic_preferences = diagnostic_preferences or DiagnosticPreferences()
        self._diagnostic_session = diagnostic_session or DiagnosticSession("gui")
        self._last_standard_profile_key = CUSTOM_MODEL_BASE_PROFILE
        self._opening_settings = False
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_status_label)

        root = QWidget()
        root.setObjectName("centralRoot")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 18)
        layout.setSpacing(12)

        brand_header = QWidget()
        brand_header.setObjectName("brandHeader")
        heading_row = QHBoxLayout(brand_header)
        heading_row.setContentsMargins(12, 9, 12, 9)
        brand_mark = QLabel("CC")
        brand_mark.setProperty("role", "brandMark")
        heading = QLabel("CaptionMiner")
        heading.setProperty("role", "heading")
        family_label = QLabel("MINER TOOLS")
        family_label.setProperty("role", "family")
        self.settings_button = QPushButton("Settings")
        heading_row.addWidget(brand_mark)
        heading_row.addWidget(heading)
        heading_row.addWidget(family_label)
        heading_row.addStretch(1)
        heading_row.addWidget(self.settings_button)
        layout.addWidget(brand_header)
        description = QLabel(
            "Drop exported clips below. CaptionMiner transcribes them locally and creates "
            "plain SRT subtitle files; styling remains in your video editor."
        )
        description.setWordWrap(True)
        description.setProperty("role", "muted")
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
        self.profile_combo.addItem("Custom — not configured", CUSTOM_MODEL_KEY)
        self._custom_profile_index = self.profile_combo.count() - 1
        self._refresh_custom_profile_item()
        self.profile_combo.setCurrentIndex(1)
        settings.addRow("Accuracy profile / model", self.profile_combo)

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
        self.start_button.setProperty("primary", True)
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
        self.status_label.setProperty("role", "status")
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
        self.settings_button.clicked.connect(self.open_settings)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)

    def _refresh_custom_profile_item(self) -> None:
        lookup = resolve_custom_model(self._model_preferences)
        if lookup.selection is not None:
            name = lookup.selection.location.name or str(lookup.selection.location)
            label = f"Custom — {name}"
            tooltip = str(lookup.selection.location)
        elif lookup.invalid_local_reason:
            label = "Custom — unavailable"
            tooltip = lookup.invalid_local_reason
        else:
            label = "Custom — not configured"
            tooltip = "Choose a compatible local model in Settings"
        self.profile_combo.setItemText(self._custom_profile_index, label)
        self.profile_combo.setItemData(
            self._custom_profile_index,
            tooltip,
            Qt.ItemDataRole.ToolTipRole,
        )

    def _set_profile_key(self, profile_key: str) -> None:
        index = self.profile_combo.findData(profile_key)
        if index < 0:
            return
        blocked = self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(index)
        self.profile_combo.blockSignals(blocked)

    @Slot()
    def _on_profile_changed(self) -> None:
        profile_key = str(self.profile_combo.currentData())
        if profile_key != CUSTOM_MODEL_KEY:
            self._last_standard_profile_key = profile_key
            return

        if resolve_custom_model(self._model_preferences).selection is not None:
            return
        if self._opening_settings:
            return

        self.open_settings()
        if resolve_custom_model(self._model_preferences).selection is None:
            self._set_profile_key(self._last_standard_profile_key)

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
            self.settings_button,
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
            self._diagnostic_session.record(
                "output_directory_validation_failed",
                level="warning",
                reason="selected_output_is_not_a_directory",
            )
            QMessageBox.warning(self, "CaptionMiner", "The selected output path is not a folder.")
            return

        selected_profile_key = str(self.profile_combo.currentData())
        if selected_profile_key == CUSTOM_MODEL_KEY:
            worker_profile_key = CUSTOM_MODEL_BASE_PROFILE
            model_selection = self._select_custom_model_for_transcription()
        else:
            worker_profile_key = selected_profile_key
            model_selection = self._select_model_for_transcription(selected_profile_key)
        if model_selection is None:
            return
        if str(self.profile_combo.currentData()) == CUSTOM_MODEL_KEY:
            worker_profile_key = CUSTOM_MODEL_BASE_PROFILE

        prompt = self.prompt_edit.text().strip() or None
        base_options = options_for_profile(
            worker_profile_key,
            language=self.language_combo.currentData(),
            device=str(self.device_combo.currentData()),
            initial_prompt=prompt,
        )
        options = replace(
            base_options,
            model_name=model_selection.reference,
            local_files_only=model_selection.local_files_only,
        )
        detailed = self._diagnostic_preferences.consume_detailed_next_batch()
        language_mode = self.language_combo.currentData() or "auto"
        diagnostic_secrets = (
            [str(output_directory), str(output_directory.resolve())]
            if output_directory is not None
            else []
        )
        batch_diagnostics = self._diagnostic_session.start_batch(
            profile=str(self.profile_combo.currentData()),
            language_mode=str(language_mode),
            total_files=len(files),
            options=options,
            detailed=detailed,
            secrets=diagnostic_secrets,
            overwrite=self.overwrite_check.isChecked(),
            output_directory_selected=output_directory is not None,
        )
        decision = {
            "cache": "existing_downloaded_model",
            "download": "user_authorized_download",
            "local": "custom_local_model",
        }.get(model_selection.source, "resolved_model")
        batch_diagnostics.model_resolved(
            model_selection.reference,
            model_selection.source,
            decision,
        )

        self.log.clear()
        self._batch_started_at = time.monotonic()
        self._set_progress_value(INDETERMINATE_PROGRESS)
        self._set_status_message("Starting...")
        self._status_timer.start()
        self._set_running(True)

        self._thread = QThread(self)
        self._worker = BatchWorker(
            files,
            options=options,
            output_directory=output_directory,
            overwrite=self.overwrite_check.isChecked(),
            diagnostics=batch_diagnostics,
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

    def _select_model_for_transcription(self, profile_key: str) -> ModelSelection | None:
        profile = MODEL_PROFILES[profile_key]

        cached_selection = resolve_cached_model(profile.model_name)
        if cached_selection is not None:
            return cached_selection

        policy = self._model_preferences.download_policy
        if policy is DownloadPolicy.ALLOW:
            return ModelSelection(
                reference=profile.model_name,
                location=huggingface_cache_directory(),
                source="download",
                local_files_only=False,
            )
        if policy is DownloadPolicy.DENY:
            action = self._show_downloads_disabled(profile_key)
            if action == "settings":
                self.open_settings()
                return None
            if action == "local":
                selection = self._choose_custom_model()
                if selection is not None:
                    return selection
            return None

        consent_action, remember = self._ask_for_download_consent(profile_key)
        effect = apply_download_consent_action(
            self._model_preferences,
            consent_action,
            remember=remember,
        )
        if effect.allow_once:
            return ModelSelection(
                reference=profile.model_name,
                location=huggingface_cache_directory(),
                source="download",
                local_files_only=False,
            )
        if effect.choose_local:
            selection = self._choose_custom_model()
            if selection is not None:
                return selection
            return None
        return None

    def _select_custom_model_for_transcription(self) -> ModelSelection | None:
        lookup = resolve_custom_model(self._model_preferences)
        if lookup.invalid_local_reason:
            self._diagnostic_session.record(
                "custom_model_unavailable",
                level="warning",
                reason="saved_custom_model_failed_validation",
            )
            QMessageBox.warning(
                self,
                "Custom model unavailable",
                lookup.invalid_local_reason
                + "\n\nChoose the folder again in Settings after fixing it.",
            )
            return None
        if lookup.selection is not None:
            return lookup.selection

        self.open_settings()
        lookup = resolve_custom_model(self._model_preferences)
        return lookup.selection

    def _ask_for_download_consent(
        self,
        profile_key: str,
    ) -> tuple[DownloadConsentAction, bool]:
        profile = MODEL_PROFILES[profile_key]
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Question)
        message.setWindowTitle("Speech-recognition model required")
        message.setText(f"The {profile.label} model is not installed.")
        message.setInformativeText(
            "CaptionMiner can download it once from Hugging Face and reuse it later. "
            "Your video or audio will not be uploaded.\n\n"
            f"Model: {profile.model_name}\n"
            f"Download location: {huggingface_cache_directory()}\n\n"
            "If Don't ask me again is selected, Download model enables automatic "
            "downloads and No disables them. This can be changed later in Settings."
        )
        remember_check = QCheckBox("Don't ask me again")
        message.setCheckBox(remember_check)
        download_button = message.addButton("Download model", QMessageBox.AcceptRole)
        local_button = message.addButton("Choose custom model folder", QMessageBox.ActionRole)
        deny_button = message.addButton("No", QMessageBox.RejectRole)
        message.setDefaultButton(download_button)
        message.exec()

        clicked = message.clickedButton()
        if clicked is download_button:
            action = DownloadConsentAction.DOWNLOAD
        elif clicked is local_button:
            action = DownloadConsentAction.LOCAL
        elif clicked is deny_button:
            action = DownloadConsentAction.DENY
        else:
            action = DownloadConsentAction.DISMISS
        return action, remember_check.isChecked()

    def _show_downloads_disabled(self, profile_key: str) -> str:
        profile = MODEL_PROFILES[profile_key]
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Information)
        message.setWindowTitle("Model downloads are disabled")
        message.setText(f"The {profile.label} model is not installed.")
        message.setInformativeText(
            "Choose a custom model folder or change the model-download preference in Settings."
        )
        local_button = message.addButton("Choose custom model folder", QMessageBox.ActionRole)
        settings_button = message.addButton("Open Settings", QMessageBox.ActionRole)
        cancel_button = message.addButton("Cancel", QMessageBox.RejectRole)
        message.setDefaultButton(local_button)
        message.exec()

        clicked = message.clickedButton()
        if clicked is local_button:
            return "local"
        if clicked is settings_button:
            return "settings"
        if clicked is cancel_button:
            return "cancel"
        return "cancel"

    def _choose_custom_model(self) -> ModelSelection | None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose a faster-whisper model folder",
        )
        if not selected:
            return None
        try:
            self._model_preferences.set_custom_model_path(Path(selected))
        except ValueError as exc:
            self._diagnostic_session.log_exception(
                "custom_model_validation_failed",
                exc,
                level="warning",
                secrets=(selected,),
            )
            QMessageBox.warning(self, "Invalid model folder", str(exc))
            return None
        self._refresh_custom_profile_item()
        self._set_profile_key(CUSTOM_MODEL_KEY)
        lookup = resolve_custom_model(self._model_preferences)
        return lookup.selection

    @Slot()
    def open_settings(self) -> None:
        if self._opening_settings:
            return
        self._opening_settings = True
        try:
            dialog = ModelSettingsDialog(
                self._model_preferences,
                self._diagnostic_preferences,
                self._diagnostic_session,
                self,
            )
            dialog.exec()
        finally:
            self._opening_settings = False

        self._refresh_custom_profile_item()
        custom_lookup = resolve_custom_model(self._model_preferences)
        if dialog.custom_model_changed and custom_lookup.selection is not None:
            self._set_profile_key(CUSTOM_MODEL_KEY)
        elif (
            dialog.custom_model_changed
            and str(self.profile_combo.currentData()) == CUSTOM_MODEL_KEY
        ):
            self._set_profile_key(self._last_standard_profile_key)

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
    diagnostic_session = DiagnosticSession("gui")
    diagnostic_session.install_exception_hooks()
    try:
        app = QApplication.instance() or QApplication([])
        app.setApplicationName("CaptionMiner")
        app.setOrganizationName("CaptionMiner")
        apply_miner_theme(app)
        window = MainWindow(diagnostic_session=diagnostic_session)
        window.show()
        return app.exec()
    except Exception as exc:
        diagnostic_session.log_exception("gui_failed", exc)
        raise
    finally:
        diagnostic_session.close()
