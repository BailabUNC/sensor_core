---
title: "sensor_core: a Python-based library for real-time acquisition, visualization, processing, and storage of custom sensor data"
authors:
  - name: Arjun Putcha
  - name: Rachit Keyal
  - name: Aastha Sharma
  - name: Grace Sosa
  - name: Luke Piasecki
  - name: Nina Dhillon
  - name: Michael Kosorok
  - name: Wubin Bai
date: 2025-06-11
---

## Summary

sensor_core is an open-source software framework for real-time sensor data acquisition, visualization, processing, and storage designed to support custom and heterogeneous sensor streams. It provides a modular architecture that enables hardware-agnostic interfaces for sensor integration, streaming visualizations for immediate data assessment, configurable processing pipelines for digital filtering, and efficient storage backends for persistent data archiving. The framework emphasizes low-latency operation and extensibility, making it suitable for a broad subset of research applications.

## Statement of Need

Contemporary sensor-driven research—such as robotics, neuroscience, and medical devices—increasingly depends on the real-time collection and interpretation of high-frequency data streams. Many existing solutions are either proprietary, tied to specific hardware, or lack integrated real-time processing and visualization capabilities. This fragmentation creates barriers for researchers who need flexible, extensible, and open frameworks capable of supporting diverse sensors and experimental conditions.

sensor_core fills this gap by providing consolidated infrastructure for:

- Real-time acquisition of data from heterogeneous sensors with reliable time synchronization.
- Interactive visualization using fastplotlib to support exploratory analysis and monitoring workflows.
- Processing pipelines that allow custom digital signal processing within live streams.
- Persistent storage with schema-agnostic backends that facilitate reproducible downstream analysis.

The target audience includes researchers, engineers, and developers who require a lightweight but comprehensive foundation for building sensor-centric research systems. By abstracting common real-time concerns and providing extensible interfaces, sensor_core reduces the engineering overhead of custom integrations and fosters reproducible data workflows across domains.

## Comparison With Related Tools

Existing software frameworks, depicted in Figure 1, vary widely in scope and design:

- **Proprietary acquisition suites**, such as LabVIEW and DASYLab, often support full data pipeline functionality for specific vendor hardware but limit extensibility and introduce cost and licensing barriers.
- **Hardware-specific libraries**, such as nidaqmx, can provide high-fidelity support for specific vendor hardware but are not generalizable to other custom devices.
- **Feature-specific libraries**, such as PySerial, Matplotlib, and SQLite, offer reliable tools for individual pipeline components but do not independently support full custom sensor data management workflows.

sensor_core distinguishes itself by combining end-to-end pipeline support with a modular architecture that developers can adapt to novel sensor classes. Its real-time visualization and storage capabilities are embedded, eliminating the need for third-party tools for core pipeline elements. While default methods are available for data acquisition and processing, custom methods can be integrated into the pipeline to support broader generalizability.

![Overview of the sensor_core data pipeline architecture and comparison with existing tools.](sensor_core/paper/pipeline.jpg)
**Figure 1:** Overview of the sensor_core data pipeline architecture and related tools.

## Features

sensor_core achieves high data throughput through three primary architectural features:

### Multiprocessing and Zero-Copy Buffer Transfers

Separate processes are instantiated to handle data acquisition, visualization, and storage, reducing latency and contention across tasks. With the exception of the plotting process, zero-copy buffer transfers are employed to minimize memory overhead and inter-process communication costs.

### Single-Producer Multiple-Consumer Circular Buffer

A single circular buffer is shared across processes with an enforced fixed lag between write and read pointers, eliminating the need for mutex locks on shared memory. The buffer is implemented in C++ for performance and accessed through a Python adaptor layer.

### Dual-Binary File Streaming

Although SQLite provides a lightweight and tabular storage format, appending high-frequency data can induce significant latency. To mitigate this, sensor_core streams data into one of two alternating binary files while a separate process asynchronously offloads data into a SQLite database. This approach minimizes storage-induced bottlenecks in the real-time pipeline.

sensor_core also leverages fastplotlib, a GPU-accelerated visualization library built on WGPU, to enable efficient rendering of both line-based and image-based sensor data streams.

![Performance benchmarks for line and image data pipelines.](timing.png)
**Figure 2:** Writer, plotter, and ingest performance for line and image data streams.

## Software Use Cases and Impact

sensor_core has been used in multiple photoplethysmography-based applications, including pulse oximetry and near-infrared muscle tracking systems, to enable real-time medical device experimentation. These use cases demonstrate that sensor_core supports demanding real-time workloads while enabling reproducible and transparent research workflows.

## Acknowledgements

The authors acknowledge the contributions of collaborators and beta testers who provided valuable feedback during development. This work was supported by the North Carolina Biotechnology Translational Research Grant (NC Biotech TRG) and the 1789 Student Venture Fund at the University of North Carolina.

## References

References are provided in the accompanying `paper.bib` file.
