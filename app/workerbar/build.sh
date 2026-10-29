#!/bin/sh
set -e
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
VERSION="${ORCHARD_VERSION:-0.4.0}"

swift build -c release --package-path "$HERE"

APP="$HERE/build/OrchardWorker.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$HERE/.build/release/OrchardWorkerBar" "$APP/Contents/MacOS/OrchardWorker"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>io.orchard.worker</string>
    <key>CFBundleName</key><string>Orchard Worker</string>
    <key>CFBundleDisplayName</key><string>Orchard Worker</string>
    <key>CFBundleExecutable</key><string>OrchardWorker</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

codesign --force --sign - "$APP" >/dev/null 2>&1 || true

INSTALLED="$HOME/Applications/OrchardWorker.app"
mkdir -p "$HOME/Applications"
rm -rf "$INSTALLED"
cp -R "$APP" "$INSTALLED"
codesign --force --sign - "$INSTALLED" >/dev/null 2>&1 || true

echo "built:     $APP"
echo "installed: $INSTALLED"
echo "launch:    open \"$INSTALLED\""
