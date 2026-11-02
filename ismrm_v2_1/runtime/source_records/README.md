# Source and migration records

Run `scripts/capture_live_source_record.sh` on the server before copying a live route. It records the current code HEAD/status, active model/protocol SHA and all five subject run roots without modifying them. Later migration manifests must include source path, destination path, byte SHA256 and terminal status.

