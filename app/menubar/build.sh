#!/bin/sh
set -e
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO="$(CDPATH= cd -- "$HERE/../.." && pwd)"
VERSION="${ORCHARD_VERSION:-0.4.0}"

swift build -c release --package-path "$HERE"

APP="$HERE/build/Orchard.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$HERE/.build/release/OrchardMenuBar" "$APP/Contents/MacOS/Orchard"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>io.orchard.menubar</string>
    <key>CFBundleName</key><string>Orchard</string>
    <key>CFBundleDisplayName</key><string>Orchard</string>
    <key>CFBundleExecutable</key><string>Orchard</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

mkdir -p "$HOME/.orchard"
printf '{"repo": "%s"}\n' "$REPO" > "$HOME/.orchard/app.json"

codesign --force --sign - "$APP" >/dev/null 2>&1 || true

INSTALLED="$HOME/Applications/Orchard.app"
mkdir -p "$HOME/Applications"
rm -rf "$INSTALLED"
cp -R "$APP" "$INSTALLED"
codesign --force --sign - "$INSTALLED" >/dev/null 2>&1 || true

echo "built:     $APP"
echo "installed: $INSTALLED"
echo "app.json:  $HOME/.orchard/app.json -> $REPO"
echo "launch:    open \"$INSTALLED\""
