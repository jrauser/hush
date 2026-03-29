import argparse
import logging
import os
import signal
import subprocess
import time
from datetime import datetime

from hush.label_store import LabelStore

DEFAULT_INTERVAL = 23
DEFAULT_TIMEOUT_HOURS = 2
DEFAULT_OUTPUT_DIR = 'screenshots'
DEFAULT_DB = 'hush.db'

logging.basicConfig(
    format='%(asctime)s  %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def take_screenshot(path):
    """Capture the main display to path as JPEG. Returns True on success."""
    result = subprocess.run(
        ['screencapture', '-x', '-t', 'jpg', '-m', path],
        capture_output=True,
    )
    return (
        result.returncode == 0
        and os.path.isfile(path)
        and os.path.getsize(path) > 0
    )


def main():
    parser = argparse.ArgumentParser(
        description='Capture screenshots at regular intervals for ad-detection training.'
    )
    parser.add_argument(
        '--interval', type=int, default=DEFAULT_INTERVAL,
        help=f'Seconds between captures (default: {DEFAULT_INTERVAL})',
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help=f'Directory to save screenshots (default: {DEFAULT_OUTPUT_DIR})',
    )
    parser.add_argument(
        '--db', default=DEFAULT_DB,
        help=f'SQLite database path (default: {DEFAULT_DB})',
    )
    parser.add_argument(
        '--timeout', type=float, default=DEFAULT_TIMEOUT_HOURS,
        metavar='HOURS',
        help=f'Stop automatically after this many hours (default: {DEFAULT_TIMEOUT_HOURS}; '
             f'use 0 to run indefinitely)',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    store = LabelStore(args.db)

    deadline = None
    if args.timeout > 0:
        deadline = time.monotonic() + args.timeout * 3600

    running = True

    def _handle_sigterm(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _handle_sigterm)

    timeout_str = f'{args.timeout}h' if args.timeout > 0 else 'none'
    log.info(
        f'Starting. interval={args.interval}s  timeout={timeout_str}'
        f'  dir={args.output_dir}  db={args.db}'
    )
    log.info('Press Ctrl+C to stop.')

    count = 0
    try:
        while running:
            if deadline and time.monotonic() >= deadline:
                log.info(f'Timeout reached ({args.timeout}h). Stopping.')
                break

            now = datetime.now()
            basename = now.strftime('%Y%m%d_%H%M%S') + '.jpg'
            filepath = os.path.join(args.output_dir, basename)

            if take_screenshot(filepath):
                store.add_screenshot(filepath, now.isoformat())
                count += 1
                log.info(f'[{count}] {filepath}')
            else:
                log.warning(f'screencapture failed for {filepath}')
                log.warning(
                    'Check Screen Recording permission: '
                    'System Settings → Privacy & Security → Screen Recording'
                )

            time.sleep(args.interval)

    except KeyboardInterrupt:
        pass

    log.info(f'Done. Captured {count} frame{"s" if count != 1 else ""}.')


if __name__ == '__main__':
    main()
