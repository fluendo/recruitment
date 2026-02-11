# Test for Multimedia Engineer

Fluendo is a leading company in multimedia solutions based on the GStreamer framework. We develop a range of products, from high-end video and audio codecs to sleek end-user applications and middleware powering playback, encoding, and streaming scenarios. Our software is used in millions of devices, and our clients include major thin client manufacturers and world-renowned film production companies.

## Assignment Overview

The task is to develop a decoder plugin for GStreamer. This plugin will accept a stream compressed with **Zstandard (zstd)** and output an uncompressed stream.

### Requirements (Mandatory)

- The plugin must be named `"zstddec"`.
- The plugin must have **one sink pad** (input) and **one source pad** (output).
- The plugin should **decompress Zstandard-compressed streams**.
- The deliverable is the source code, organized so it can be compiled on a Linux system (x86-64 architecture).
- The project must use the **Meson** build system.
- At compilation time, the plugin must depend on the **libzstd** library via `pkg-config`.
- The delivery must be a **git repository**, a **tarball including the full git history**, or a private repository on **GitHub or a similar platform**.

### Testing

Your plugin will be tested using the following GStreamer pipeline:

```sh
gst-launch-1.0 filesrc location=file.txt.zst ! zstddec ! filesink location=file.txt
```

The output `file.txt` **must be identical** to uncompressing the file with the following command:

```sh
zstd -d file.txt.zst > file.txt
```

### Optional (Extra Credit)

- Support decompressing **bzip2** and/or **gzip** streams, in addition to zstd.
    - If you support these formats, your plugin may be named generically (e.g., `multidec`), and should **autodetect the compression format**.
    - For gzip: depend on the zlib library (`pkg-config`).
    - For bzip2: depend on the bzip2 library (`pkg-config`).
- Add continuous integration (CI) tests for your plugin.
- Write documentation/FAQ for your plugin.

### Notes

- The core requirement is full Zstandard support.
- Optional: autodetect and support other formats (gzip, bzip2).
- Code must be clean and well-documented to facilitate review.

If you have any questions or would like further clarification about the requirements, do not hesitate to contact us.
