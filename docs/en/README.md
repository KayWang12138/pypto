# Project Documentation

## Introduction

This directory provides the source files for the [PyPTO Documentation Center](https://pypto.gitcode.com), including environment setup, programming guides, API references, and more.

## Contributing

Contributions to the documentation are welcome! Please refer to the [Documentation Contribution Guide](../docs/CONTRIBUTION_DOC.md) for details. Be sure to follow the documentation writing standards and submit according to the process rules. Once approved, your contributions will appear in the `docs` directory of this project and on the Documentation Center website. If you have any comments or suggestions about the documentation, please submit them in Issues.

## Directory Structure

The key directory structure is as follows:

```txt
├── install                    # Environment setup
├── invocation                 # Running examples
├── tutorials                  # PyPTO Programming Guide
├── api                        # PyPTO API Reference
├── tools                      # PyPTO Toolkit User Guide
└── README
```

## Building the Documentation

Both the PyPTO Programming Guide and API documentation can be generated using Sphinx. After a documentation PR is merged, the build is triggered automatically. Local builds are also supported. Before building locally, install the required modules by following these steps.

1. Clone the PyPTO repository.

   ```bash
   git clone https://gitcode.com/cann/pypto.git
   ```

2. Navigate to the `docs` directory and install the dependencies listed in `requirements.txt`.

   ```bash
   cd docs
   pip install -r requirements.txt
   ```

3. Run the following command in the `docs` directory to build the documentation.

   ```bash
   make html
   ```

4. After the build completes, a `_build/html` directory will be created. Run the following command to start an HTTP server to serve the documentation.

   ```bash
   cd _build/html
   python3 -m http.server 8000
   ```

   The default port is 8000; you may specify a different port if needed.

5. Open `http://localhost:8000` in your browser to view the documentation.
