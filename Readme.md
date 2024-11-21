# Unique IPv4 Counter
Retrying [this assignment](https://github.com/Ecwid/new-job/blob/master/IP-Addr-Counter.md) with rust.

## Implementation Overview
The Program generally speaking does this steps

- Memory maps the input file
- Spawn threads each loads small chunk parses sets in bitset, then loads next chunk
- Count number of set bits and print

 ### Used Technics For Speedup

| Optimization                | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| Parallelization             | Used as many threads as processors available in the system                                      |
| Small Chunks                | Parsing in small chunks, each thread when available takes next the chunk, 1MB worked best       |
| Memory Mapping              | Memory mapping the file, to reduce unnecessary memory usage                                     |
| Unsafe                      | Using unsafe in some places to remove pointer bound checks when reading from mapped memory      |
| SIMD                        | Using simd instruction to load and convert IP addresses to `u32`                                |
| BitSet                      | Using atomic array as a bitset instead of `HashSet` or `boolean` array                          |
| Thread Priority             | Setting priority to max when `max_thread_priority` feature is enabled (may require root user)   |
| Profile-guided optimization | Compiling with Profile guided optimizations                                                     |
| Using x86_64 intrinsics     | using safe_arch crate for simd instruction, because in some cases portable_simd lacked support  |


### References
- https://questdb.io/blog/billion-row-challenge-step-by-step/
- https://www.jbang.dev/documentation/guide/latest/index.html
- https://perf.wiki.kernel.org/index.php/Main_Page
- https://docs.oracle.com/javase/8/docs/technotes/guides/visualvm/
- https://github.com/async-profiler/async-profiler
- https://questdb.io/blog/1brc-merykittys-magic-swar/
- http://0x80.pl/notesen/2023-04-09-faster-parse-ipv4.html#scalar-conversion
- https://lemire.me/blog/2023/06/08/parsing-ip-addresses-crazily-fast/
