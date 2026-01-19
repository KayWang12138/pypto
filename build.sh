#!/bin/bash
# PyPTO Build Script
# This script provides a convenient interface to build the PyPTO project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
BUILD_TYPE="Release"
CLEAN_BUILD=false
VERBOSE=false
EDITABLE=false
BACKEND="cost_model"
FRONTEND="python3"
JOB_NUM=-1

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to print usage
usage() {
    cat << EOF
PyPTO Build Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -c, --clean             Clean build directory before building
    -v, --verbose           Enable verbose output
    -e, --editable          Install in editable mode (for development)
    -t, --build-type TYPE   Build type: Debug, Release (default: Release)
                            Options: Debug, Release, MinSizeRel, RelWithDebInfo
    -b, --backend TYPE      Backend type: npu, cost_model (default: cost_model)
    -f, --frontend TYPE     Frontend type: python3, cpp (default: python3)
    -j, --jobs N            Number of parallel jobs (default: auto)
    --cmake-verbose         Enable CMake verbose output
    --cmake-generator GEN   Specify CMake generator (e.g., "Unix Makefiles", "Ninja")
    --third-party-path PATH Path to third-party libraries

Examples:
    # Quick build (editable mode, Release)
    $0

    # Clean build with Debug mode
    $0 -c -t Debug

    # Production build (non-editable)
    $0 -t Release

    # Build with specific backend
    $0 -b npu

    # Build with verbose output
    $0 -v --cmake-verbose

EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.9 or later."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_info "Found Python: $PYTHON_VERSION"
    
    # Check CMake
    CMAKE_FOUND=false
    if command -v cmake &> /dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n 1 | awk '{print $3}')
        CMAKE_FOUND=true
    elif [ -f "/opt/homebrew/bin/cmake" ]; then
        export PATH="/opt/homebrew/bin:$PATH"
        CMAKE_VERSION=$(cmake --version | head -n 1 | awk '{print $3}')
        CMAKE_FOUND=true
    elif [ -f "/usr/local/bin/cmake" ]; then
        export PATH="/usr/local/bin:$PATH"
        CMAKE_VERSION=$(cmake --version | head -n 1 | awk '{print $3}')
        CMAKE_FOUND=true
    fi
    
    if [ "$CMAKE_FOUND" = false ]; then
        print_error "CMake not found. Please install CMake 3.16.3 or later."
        echo "  macOS: brew install cmake"
        echo "  Linux: apt-get install cmake  or  yum install cmake"
        exit 1
    fi
    
    print_info "Found CMake: $CMAKE_VERSION"
    
    # Check compiler
    if command -v gcc &> /dev/null || command -v gcc-13 &> /dev/null || command -v gcc-12 &> /dev/null || command -v gcc-11 &> /dev/null; then
        print_info "Using GNU GCC (recommended)"
    elif [ -f "/opt/homebrew/bin/gcc-13" ] || [ -f "/opt/homebrew/bin/gcc-12" ]; then
        export PATH="/opt/homebrew/bin:$PATH"
        print_info "Using GNU GCC from Homebrew"
    else
        print_warning "GNU GCC not found. Using default compiler (may be Clang)."
        print_warning "For best compatibility, install GCC:"
        echo "  macOS: brew install gcc"
        echo "  Linux: apt-get install gcc g++  or  yum install gcc gcc-c++"
    fi
    
    # Check pip and required packages
    print_info "Checking Python dependencies..."
    python3 -m pip --version > /dev/null 2>&1 || {
        print_error "pip not found. Please install pip."
        exit 1
    }
    
    # Try to install/upgrade required packages
    print_info "Installing/upgrading build dependencies..."
    python3 -m pip install --upgrade --user setuptools pybind11 sympy PyYAML > /dev/null 2>&1 || {
        print_warning "Some dependencies may not have installed correctly"
    }
    
    print_success "Prerequisites check completed"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -e|--editable)
            EDITABLE=true
            shift
            ;;
        -t|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -b|--backend)
            BACKEND="$2"
            shift 2
            ;;
        -f|--frontend)
            FRONTEND="$2"
            shift 2
            ;;
        -j|--jobs)
            JOB_NUM="$2"
            shift 2
            ;;
        --cmake-verbose)
            VERBOSE=true
            shift
            ;;
        --cmake-generator)
            CMAKE_GENERATOR="$2"
            shift 2
            ;;
        --third-party-path)
            THIRD_PARTY_PATH="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main build function
main() {
    echo "=========================================="
    echo "  PyPTO Build Script"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    
    # Build command arguments
    BUILD_ARGS=()
    
    # Add common arguments
    [ "$CLEAN_BUILD" = true ] && BUILD_ARGS+=("-c")
    [ "$VERBOSE" = true ] && BUILD_ARGS+=("--verbose")
    [ "$EDITABLE" = true ] && BUILD_ARGS+=("--editable")
    
    # Add build configuration
    BUILD_ARGS+=("--build_type" "$BUILD_TYPE")
    BUILD_ARGS+=("-f" "$FRONTEND")
    BUILD_ARGS+=("-b" "$BACKEND")
    
    # Add job number if specified
    if [ "$JOB_NUM" != "-1" ]; then
        BUILD_ARGS+=("-j" "$JOB_NUM")
    fi
    
    # Add CMake options
    if [ -n "$CMAKE_GENERATOR" ]; then
        BUILD_ARGS+=("--generator" "$CMAKE_GENERATOR")
    fi
    
    if [ -n "$THIRD_PARTY_PATH" ]; then
        BUILD_ARGS+=("--third_party_path" "$THIRD_PARTY_PATH")
    fi
    
    # Print build configuration
    print_info "Build Configuration:"
    echo "  Frontend:      $FRONTEND"
    echo "  Backend:       $BACKEND"
    echo "  Build Type:    $BUILD_TYPE"
    echo "  Editable Mode: $EDITABLE"
    echo "  Clean Build:   $CLEAN_BUILD"
    echo "  Verbose:       $VERBOSE"
    [ -n "$JOB_NUM" ] && [ "$JOB_NUM" != "-1" ] && echo "  Jobs:          $JOB_NUM"
    [ -n "$CMAKE_GENERATOR" ] && echo "  Generator:     $CMAKE_GENERATOR"
    echo ""
    
    # Run build
    print_info "Starting build..."
    print_info "Command: python3 build_ci.py ${BUILD_ARGS[*]}"
    echo ""
    
    if python3 build_ci.py "${BUILD_ARGS[@]}"; then
        echo ""
        print_success "Build completed successfully!"
        echo ""
        
        if [ "$EDITABLE" = true ]; then
            print_info "Package installed in editable mode."
            print_info "You can now test with:"
            echo "  python3 -c 'import pypto; print(\"PyPTO installed successfully\")'"
        else
            # Find and install the wheel file
            WHEEL_DIR="build_out"
            
            if [ ! -d "$WHEEL_DIR" ]; then
                print_warning "Build output directory '$WHEEL_DIR' not found."
                print_info "Package built successfully. Wheel may be in a different location."
                print_info "You can search for wheel files with:"
                echo "  find . -name 'pypto-*.whl' -type f"
            else
                # Find wheel files matching the pattern
                # Pattern: pypto-*-cp*-cp*-*.whl (e.g., pypto-0.1.0-cp39-cp39-linux_x86_64.whl)
                PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
                PYTHON_TAG="cp$(echo $PYTHON_VERSION | tr -d '.')"
                
                # First try to find wheel matching current Python version
                WHEEL_FILE=$(find "$WHEEL_DIR" -name "pypto-*-${PYTHON_TAG}-*.whl" -type f 2>/dev/null | head -1)
                
                # If not found, try any pypto wheel file
                if [ -z "$WHEEL_FILE" ]; then
                    WHEEL_FILE=$(find "$WHEEL_DIR" -name "pypto-*.whl" -type f 2>/dev/null | head -1)
                fi
                
                if [ -n "$WHEEL_FILE" ] && [ -f "$WHEEL_FILE" ]; then
                    print_info "Found wheel file: $WHEEL_FILE"
                    print_info "Installing wheel..."
                    echo ""
                    
                    if python3 -m pip install "$WHEEL_FILE"; then
                        echo ""
                        print_success "Wheel installed successfully!"
                        print_info "You can now test with:"
                        echo "  python3 -c 'import pypto; print(\"PyPTO installed successfully\")'"
                    else
                        echo ""
                        print_error "Failed to install wheel. You can install manually with:"
                        echo "  python3 -m pip install $WHEEL_FILE"
                        exit 1
                    fi
                else
                    print_warning "No wheel file found in $WHEEL_DIR"
                    print_info "Package built successfully. You can search for wheel files with:"
                    echo "  find . -name 'pypto-*.whl' -type f"
                fi
            fi
        fi
    else
        echo ""
        print_error "Build failed. Please check the error messages above."
        exit 1
    fi
}

# Run main function
main
