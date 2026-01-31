#!/bin/bash
# Diagnostic script to check addr2line functionality

echo "==================================="
echo "Backtrace Addr2line Diagnostic"
echo "==================================="
echo ""

# Find the test binary
BINARY=$(find /data/g00655722/new-ir/pypto_open -name "*tile_fwk_utest" 2>/dev/null | head -1)

if [ -z "$BINARY" ]; then
    echo "ERROR: Cannot find tile_fwk_utest binary"
    exit 1
fi

echo "Binary: $BINARY"
echo ""

# Check if binary has debug symbols
echo "Checking for debug symbols..."
if file "$BINARY" | grep -q "not stripped"; then
    echo "✓ Binary is NOT stripped (has debug symbols)"
else
    echo "✗ Binary is STRIPPED (no debug symbols)"
    echo "  This explains why addr2line cannot resolve file/line info"
    echo ""
    echo "Solution: Rebuild with debug symbols:"
    echo "  cmake -DCMAKE_BUILD_TYPE=Debug .."
    echo "  or"
    echo "  add -g flag to compilation"
fi
echo ""

# Check if addr2line is available
echo "Checking addr2line availability..."
if command -v addr2line &> /dev/null; then
    echo "✓ addr2line is available: $(which addr2line)"
    addr2line --version | head -1
else
    echo "✗ addr2line is NOT available"
    echo "  Install binutils package"
fi
echo ""

# Try to get an address from nm
echo "Testing addr2line with a known symbol..."
if command -v nm &> /dev/null; then
    # Get address of main function
    MAIN_ADDR=$(nm "$BINARY" 2>/dev/null | grep " main$" | awk '{print $1}')
    if [ -n "$MAIN_ADDR" ]; then
        echo "Found 'main' at address: 0x$MAIN_ADDR"
        echo ""
        echo "Running addr2line test:"
        echo "$ addr2line -e $BINARY -f -C -p 0x$MAIN_ADDR"
        addr2line -e "$BINARY" -f -C -p "0x$MAIN_ADDR" 2>&1
    else
        echo "Could not find 'main' symbol in binary"
    fi
else
    echo "nm command not available, skipping symbol test"
fi
echo ""

echo "==================================="
echo "Diagnostic Complete"
echo "==================================="
