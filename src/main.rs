#![allow(internal_features)]
#![feature(core_intrinsics)]

use std::fs::File;
use std::io;
use std::mem::transmute;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;

/// A set to keep track of unique IPs using a bitmask.
struct MaxBitSet {
    // words: Box<[AtomicU64; 1 << 26]>,
    words: Vec<AtomicU64>,
}

impl MaxBitSet {
    fn new() -> Self {
        // let words = Box::new([const { AtomicU64::new(0) }; 1 << 26]);
        let words = vec![0; 1 << 26].into_iter().map(AtomicU64::new).collect();
        Self { words }
    }

    // // #[inline(never)]
    /// Sets the bit at the specified index.
    fn set(&self, bit_index: u32) {
        let word = unsafe {
            &*self
                .words
                .as_ptr()
                .offset(bit_index.wrapping_shr(6) as isize)
        };
        let mask = 1u64.wrapping_shl(bit_index);
        AtomicU64::fetch_or(word, mask, Ordering::SeqCst);
    }

    // #[inline(never)]
    /// Returns the number of unique bits set.
    fn count(&self) -> u64 {
        self.words
            .iter()
            .map(|word| word.load(Ordering::Relaxed).count_ones() as u64)
            .sum()
    }
}

// #[inline(never)]
fn num(mut value: u128, new_line: u32) -> u32 {
    value = value & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F;
    let mut shift_size = 24;
    let mut current_byte: u8 = 0;
    let mut result = 0;

    for _ in 0..new_line {
        let c = (value & 0xFF) as u8;
        value >>= 8;

        if c == 0x0e {
            result |= (current_byte as u32) << shift_size;
            shift_size -= 8;
            current_byte = 0;
        } else {
            current_byte = current_byte.wrapping_mul(10).wrapping_add(c);
        }
    }

    result | (current_byte as u32)
}

// #[inline(never)]
fn read_wide(bytes: &[u8]) -> u128 {
    if std::intrinsics::likely(bytes.len() >= 16) {
        u128::from_le_bytes(bytes[..16].try_into().unwrap())
    } else {
        let mut wide = 0u128;
        let mut offset = 0;
        for &b in bytes.iter().take(16) {
            wide |= (b as u128) << offset;
            offset += 8;
        }
        wide
    }
}

// #[inline(never)]
fn newline_location(word: u128) -> u32 {
    let reversed = word ^ 0x0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A0A; // xor with \n\n\n...\n\n\n
    let code = (reversed.wrapping_sub(0x01010101010101010101010101010101))
        & (!reversed & 0x80808080808080808080808080808080);

    code.trailing_zeros() >> 3
}

// #[inline(never)]
fn worker(
    memory: &'static [u8],
    chunk_id: &'static AtomicUsize,
    bitset: &'static MaxBitSet,
    file_size: usize,
) {
    const CHUNK_SIZE: usize = 1 * 1024 * 1024;
    loop {
        let id = chunk_id.fetch_add(1, Ordering::SeqCst);
        if std::intrinsics::unlikely(id * CHUNK_SIZE >= memory.len()) {
            break;
        }
        let mut current_mem = &memory[(id * CHUNK_SIZE)..];
        let end = memory[std::cmp::min((id + 1) * CHUNK_SIZE, file_size)..].as_ptr();

        // Adjusting start
        if std::intrinsics::likely(id > 0) {
            let value = read_wide(current_mem);
            current_mem = &current_mem[newline_location(value) as usize..];
        }

        parse_loop(bitset, current_mem, end);
    }
}

// #[inline(never)]
fn parse_loop(bitset: &MaxBitSet, mut current_mem: &[u8], end: *const u8) {
    while current_mem.as_ptr() < end {
        let value = read_wide(current_mem);
        let next_new = newline_location(value) + 1;
        bitset.set(num(value, next_new));
        current_mem = &current_mem[(next_new as usize)..];
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: unique_ip_counter <file_path>");
        return Ok(());
    }
    let file_path = &args[1];

    let file = File::open(file_path)?;

    let count = count_ips(&file);

    println!("Number of unique IP addresses: {}", count);
    Ok(())
}

fn count_ips(file: &File) -> u64 {
    let file_size = file.metadata().unwrap().len() as usize;

    // Create memory-mapped file
    let mmap = Box::leak(Box::new(unsafe { memmap2::Mmap::map(file).unwrap() }));
    let map: &'static [u8] = unsafe { transmute::<&[u8], &'static [u8]>(mmap.as_ref()) };

    let chunk_id = Box::leak(Box::new(AtomicUsize::new(0)));
    let bitset = Box::leak(Box::new(MaxBitSet::new()));

    let mut handles = Vec::new();
    let num_threads = num_cpus::get();
    for _ in 0..num_threads {
        handles.push(thread::spawn(thread(map, chunk_id, bitset, file_size)));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    bitset.count()
}

fn thread(
    map: &'static [u8],
    chunk_id: &'static AtomicUsize,
    bitset: &'static MaxBitSet,
    file_size: usize,
) -> impl FnOnce() {
    move || {
        thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Max).unwrap();
        worker(map, chunk_id, bitset, file_size);
    }
}

#[cfg(test)]
mod tests {
    use crate::{newline_location, num, read_wide};
    use rand::Rng;

    // #[test]
    // fn generate_test_file() {
    //     use rand::Rng;
    //     use std::fs::File;
    //     use std::io::{self, Write};
    //     use std::io::BufWriter;
    //
    //     /// Generates `n` random IP addresses and writes them to the specified file.
    //     fn generate_ips_and_write_to_file(filename: &str, n: usize) -> io::Result<()> {
    //         let mut file = BufWriter::new(File::create(filename)?);
    //         let mut rng = rand::thread_rng();
    //
    //         for _ in 0..n {
    //             // Generate a random IP address by creating four random octets.
    //             let ip = format!(
    //                 "{}.{}.{}.{}",
    //                 rng.gen::<u8>(),
    //                 rng.gen::<u8>(),
    //                 rng.gen::<u8>(),
    //                 rng.gen::<u8>()
    //             );
    //
    //             writeln!(file, "{}", ip)?;
    //         }
    //
    //         Ok(())
    //     }
    //
    //     let filename = "target/ips-100.txt";
    //     let n = 100;
    //
    //     generate_ips_and_write_to_file(filename, n).unwrap();
    //     println!("Generated {} IP addresses and wrote to {}", n, filename);
    // }

    /// Converts IP string to an integer bit index.
    fn ip_to_index(ip: &[u8]) -> Option<u32> {
        let parts: Vec<&[u8]> = ip.split(|&c| c == '.' as u8).collect();
        if parts.len() == 4 {
            let octets: Vec<u8> = parts
                .iter()
                .filter_map(|s| String::from_utf8_lossy(s).parse().ok())
                .collect();
            if octets.len() == 4 {
                Some(
                    ((octets[0] as u32) << 24)
                        | ((octets[1] as u32) << 16)
                        | ((octets[2] as u32) << 8)
                        | (octets[3] as u32),
                )
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Converts IP string to an integer bit index.
    fn ip_to_index_str(ip: &str) -> Option<u32> {
        let parts: Vec<&str> = ip.split('.').collect();
        if parts.len() == 4 {
            let octets: Vec<u8> = parts.iter().filter_map(|s| s.parse().ok()).collect();
            if octets.len() == 4 {
                Some(
                    ((octets[0] as u32) << 24)
                        | ((octets[1] as u32) << 16)
                        | ((octets[2] as u32) << 8)
                        | (octets[3] as u32),
                )
            } else {
                None
            }
        } else {
            None
        }
    }

    #[test]
    fn location_test_1() {
        let location = newline_location(u128::from_le_bytes(*b"1.1.12.12\n123.34"));
        assert_eq!(location, 9);
    }

    #[test]
    fn location_test_2() {
        let location = newline_location(u128::from_le_bytes(*b"1.1.1.12\n123.34."));
        assert_eq!(location, 8);
    }

    #[test]
    fn location_test_3() {
        let location = newline_location(u128::from_le_bytes(*b"\n123.234.123.123"));
        assert_eq!(location, 0);
    }

    #[test]
    fn location_test_4() {
        let location = newline_location(u128::from_le_bytes(*b"1.13.123.123\n123"));
        assert_eq!(location, 12);
    }

    #[test]
    fn location_test_5() {
        let location = newline_location(u128::from_le_bytes(*b"123.123.123.123\n"));
        assert_eq!(location, 15);
    }

    #[test]
    fn location_test_6() {
        let location = newline_location(u128::from_le_bytes(*b"1.1.1.1\n1.1.1.1\n"));
        assert_eq!(location, 7);
    }

    #[test]
    fn location_test_7() {
        let location = newline_location(u128::from_le_bytes(*b"163.162.45.85\n1."));
        assert_eq!(location, 13);
    }

    #[test]
    fn location_test_8() {
        let location = newline_location(u128::from_le_bytes(*b"163.162.45.8521."));
        assert_eq!(location, 16);
    }

    #[test]
    fn parse_test_1() {
        let data = *b"163.162.45.85\n1.";
        let location = newline_location(u128::from_le_bytes(data));
        let number = num(read_wide(&data), location);
        let number_real = ip_to_index(&data[..(location as usize)]).unwrap();
        assert_eq!(number, number_real);
    }

    #[test]
    fn parse_test_2() {
        let data = *b"1.1.1.1\n2.2.2.2\n";
        let location = newline_location(u128::from_le_bytes(data));
        let number = num(read_wide(&data), location);
        assert_eq!(number, 0x01010101u32);
    }

    #[test]
    fn parse_test_3() {
        let data = *b"1.2.3.4\n2.2.2.2\n";
        let location = newline_location(u128::from_le_bytes(data));
        let number = num(read_wide(&data), location);
        assert_eq!(number, 0x01020304u32);
    }

    #[test]
    fn parse_fuzzy_test() {
        let mut rng = rand::thread_rng();
        let mut ip_str = move || {
            format!(
                "{}.{}.{}.{}",
                rng.gen::<u8>(),
                rng.gen::<u8>(),
                rng.gen::<u8>(),
                rng.gen::<u8>()
            )
        };
        for _ in 0..100000 {
            let ip1 = ip_str();
            let ip2 = ip_str();
            let data = format!("{ip1}\n{ip2}\n");
            let data_bytes = data.as_bytes()[0..16].try_into().unwrap();
            let location = newline_location(u128::from_le_bytes(data_bytes));
            assert_eq!(
                ip1.len(),
                location as usize,
                "length mismatch {ip1:?} {location}"
            );
            let number = num(read_wide(&data_bytes), location);
            let number_real = ip_to_index_str(&ip1).unwrap();
            assert_eq!(number, number_real, "parsing this chunk failed '{data:?}'");
        }
    }
}
